"""Checks for the memt_cc training changes (chunked loss, per-document windows, skipped last write)."""
import json

import numpy as np
import torch
import torch.nn.functional as F

from conftest import tiny_qwen3
from lact_qwen import MemoryConfig, attach_memory, freeze_base, memory_parameters, set_attention_window
from lact_qwen.distill import ce_and_kl
from lact_qwen.text_data import document_sequences
from lact_qwen.token_store import TokenStore, document_windows
from lact_qwen.train_loss import StudentWithLoss, chunked_ce_kl


def memt_model(seed=1):
    torch.manual_seed(seed)
    m = tiny_qwen3()
    set_attention_window(m, 8, 0, 8)
    attach_memory(m, MemoryConfig(num_heads=4, head_dim=16, core="matrix", chunk_size=8, qkv_init="attention",
        qkv_act="none", inject="pre_o_proj", out_scale="memt", output_gate_rank=16, forget_rank=16, write_block=4,
        stable_matrix_gates=True))
    for layer in m.model.layers:
        torch.nn.init.normal_(layer.memory.output_gate_up.weight, std=0.5)
    freeze_base(m)
    return m.train()


def test_chunked_loss_matches_full_logits_loss_and_gradients():
    m = memt_model()
    params = memory_parameters(m)
    ids = torch.randint(0, 97, (1, 40))
    full = F.cross_entropy(m(input_ids=ids).logits[:, :-1].reshape(-1, 97), ids[:, 1:].reshape(-1))
    g_full = torch.autograd.grad(full, params)
    ce, kl, n = StudentWithLoss(m, chunk=7)(input_ids=ids, labels=ids)
    g_new = torch.autograd.grad(ce / n, params)
    assert int(n) == 39 and float(kl) == 0
    torch.testing.assert_close(ce / n, full, rtol=1e-5, atol=1e-6)
    assert max(g.abs().max() for g in g_full) > 1e-3
    for a, b in zip(g_full, g_new):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_chunked_kl_matches_reference_with_topk_and_masked_labels():
    torch.manual_seed(3)
    head = torch.nn.Linear(12, 31, bias=False)
    hidden, teacher = torch.randn(2, 9, 12, requires_grad=True), torch.randn(2, 9, 12)
    labels = torch.randint(0, 31, (2, 9))
    labels[0, :5] = -100
    for top_k in (0, 4):
        ce, kl, n = chunked_ce_kl(hidden, head, labels, teacher, head, chunk=3, top_k=top_k, temperature=1.5)
        ref_ce, ref_kl = ce_and_kl(head(hidden), teacher, head, labels, chunk=2, top_k=top_k, temperature=1.5)
        torch.testing.assert_close(ce / n, ref_ce, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(kl / n, ref_kl, rtol=1e-5, atol=1e-6)
    none = torch.full_like(labels, -100)
    ce, kl, n = chunked_ce_kl(hidden, head, none)
    assert int(n) == 0 and float(ce) == 0
    ce.backward()  # still differentiable


def test_last_write_is_skipped_without_changing_outputs_but_kept_in_a_session():
    m = memt_model().eval()
    ids = torch.randint(0, 97, (1, 24))
    calls = []
    branch = m.model.layers[0].memory
    original = branch._matrix_write
    branch._matrix_write = lambda state, x: (calls.append(x.shape[1]), original(state, x))[1]
    with torch.no_grad():
        packed = m(ids).logits
        assert calls == [8, 8]  # three segments, the last one is not written
        calls.clear()
        m.lact_ctx.start_session()
        chunks = [m(ids[:, s:s + 8], position_ids=torch.arange(s, s + 8)[None]).logits for s in range(0, 24, 8)]
        m.lact_ctx.end_session()
        assert calls == [8, 8, 8]  # streaming keeps every write
    torch.testing.assert_close(torch.cat(chunks, 1), packed, atol=2e-5, rtol=2e-5)


class CharTok:
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return type("E", (), {"input_ids": [ord(c) for c in text]})()


def test_document_windows_never_mix_documents(tmp_path):
    path = tmp_path / "d.jsonl"
    docs = ["a" * 25, "b" * 7, "c" * 10]
    path.write_text("".join(json.dumps({"text": d}) + "\n" for d in docs))
    out = list(document_sequences(str(path), CharTok(), seq_len=10, repeat=False, min_doc_tokens=8, min_window=4))
    # "a"*25 + EOS -> 10, 10, 6;  "b" document is too short;  "c"*10 + EOS -> 10 and a 1-token tail (dropped)
    assert [len(w) for w in out] == [10, 10, 6, 10]
    assert all(len(set(w) - {0}) == 1 for w in out)
    assert out[2][-1] == 0 and document_windows([26, 11], 10, 4) == [(0, 0, 10), (0, 10, 20), (0, 20, 26), (1, 0, 10)]


def test_token_store_ranks_are_disjoint_and_resume_is_exact(tmp_path):
    lengths = [30, 12, 50, 9, 41]
    tokens = np.concatenate([np.full(n, i + 1, dtype=np.uint32) for i, n in enumerate(lengths)])
    tokens.tofile(tmp_path / "tokens.u32")
    np.save(tmp_path / "offsets.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
    (tmp_path / "meta.json").write_text("{}")
    store = TokenStore(str(tmp_path))
    n = len(store.windows(16, 8))
    assert n == 10  # 30 -> 16+14, 12, 50 -> 16+16+16 (2-token tail dropped), 9, 41 -> 16+16+9
    take = lambda rank, skip, k: [tuple(w) for _, w in zip(range(k), store.stream(16, 8, rank, 2, seed=5, skip=skip))]
    a, b = take(0, 0, 5), take(1, 0, 5)  # one epoch = 5 rounds of 2 ranks
    assert all(len(set(w)) == 1 for w in a + b)  # a window holds one document only
    order = np.random.default_rng([5, 0]).permutation(n)
    assert sorted(order[0::2].tolist() + order[1::2].tolist()) == list(range(n))  # ranks split the epoch
    windows = store.windows(16, 8)
    assert a == [tuple(store.tokens[s:e].tolist()) for s, e in windows[order[0::2]]]
    assert b == [tuple(store.tokens[s:e].tolist()) for s, e in windows[order[1::2]]]
    assert take(0, 3, 6) == take(0, 0, 9)[3:]  # resume across the epoch boundary
