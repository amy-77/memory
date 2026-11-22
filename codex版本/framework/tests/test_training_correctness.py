import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lact_qwen.checkpointing import latest_checkpoint, save_training_checkpoint, archive_checkpoint, RUNTIME_NAME
from lact_qwen.training_state import capture_runtime
from lact_qwen.distill import ce_and_kl
from lact_qwen.memory_branch import MemoryConfig
from lact_qwen.modeling import attach_memory, freeze_base, memory_parameters, set_attention_window
from lact_qwen.text_data import document_sequences
from lact_qwen.training_data import optimizer_batch
from lact_qwen.training_loss import hidden_ce_and_kl, MemoryTrainingLoss
from lact_qwen.teacher_cache import make_topk_targets, TeacherCache, file_sha256, row_key, token_digest

spec = importlib.util.spec_from_file_location(
    "train_memory", Path(__file__).resolve().parents[1] / "scripts/train_memory.py")
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


class Tokenizer:
    eos_token_id = 96
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[int(i) for i in text.split()])


def test_document_windows_preserve_targets_without_cross_document_context(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('\n'.join(json.dumps({"text": t}) for t in ("1 2 3 4 5 6", "7 8", "9")))
    rows = list(document_sequences(str(path), Tokenizer(), 4, repeat=False))
    assert rows == [[1, 2, 3, 4], [4, 5, 6, 96], [7, 8, 96], [9, 96]]
    assert [x for row in rows for x in row[1:]] == [2, 3, 4, 5, 6, 96, 8, 96, 96]
    args = SimpleNamespace(text_packing="document", seq_len=4, shuffle_buffer=0,
                           min_doc_tokens=0, batch_size=3)
    batch = next(training.text_batches(str(path), Tokenizer(), args, 0, 1, 0))
    assert batch["input_ids"][2].tolist() == [7, 8, 96, 0]
    assert batch["labels"][2].tolist() == [7, 8, 96, -100]
    assert batch["attention_mask"][2].tolist() == [1, 1, 1, 0]
    assert list(document_sequences(str(path), Tokenizer(), 4, min_doc_tokens=999)) == []


def test_schedule_floor_and_legacy_compatibility():
    assert training.lr_at(0, 1., 2, 10, .05) == .5
    assert training.lr_at(2, 1., 2, 10, .05) == 1.
    assert training.lr_at(9, 1., 2, 10, .05) == pytest.approx(.05)
    assert training.lr_at(9, 1., 2, 10, .2) == pytest.approx(.2)
    import math
    for step in range(2, 10):
        old = .1 + .9 * .5 * (1 + math.cos(math.pi * (step - 2) / 8))
        assert training.lr_at(step, 1., 2, 10, legacy=True) == old


@pytest.mark.parametrize("top_k,temperature", [(0, 1.), (5, 1.), (5, 1.7)])
def test_chunked_head_loss_and_gradients_match_full_logits(top_k, temperature):
    torch.manual_seed(7)
    head = torch.nn.Linear(8, 23, bias=False).requires_grad_(False)
    hidden = torch.randn(2, 19, 8, requires_grad=True)
    teacher = torch.randn_like(hidden)
    labels = torch.randint(0, 23, (2, 19))
    labels[0, :4] = -100
    labels[1, -6:] = -100
    reference = ce_and_kl(head(hidden), teacher, head, labels,
                          top_k=top_k, temperature=temperature)
    grad = torch.autograd.grad(reference[0] + .7 * reference[1], hidden)[0]
    actual = hidden_ce_and_kl(hidden, head, labels, teacher_hidden=teacher, teacher_head=head,
                              chunk=3, top_k=top_k, temperature=temperature)
    actual_grad = torch.autograd.grad(actual[0] + .7 * actual[1], hidden)[0]
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(actual_grad, grad, atol=1e-7, rtol=1e-5)
    assert head.weight.grad is None
    masked = hidden_ce_and_kl(hidden, head, torch.full_like(labels, -100), chunk=3)
    assert torch.autograd.grad(sum(masked), hidden)[0].count_nonzero() == 0


def setup_memory(model):
    set_attention_window(model, 4, 0, 4)
    attach_memory(model, MemoryConfig(core="matrix", num_heads=4, head_dim=16,
                  chunk_size=4, write_block=2, qkv_init="attention", qkv_act="none",
                  inject="pre_o_proj", out_scale="memt", output_gate_rank=8,
                  forget_rank=8, stable_matrix_gates=True))
    freeze_base(model)
    return model


def test_training_wrapper_matches_full_model_with_checkpointing_and_padding(tiny_model):
    model = setup_memory(tiny_model).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    x = torch.randint(1, 90, (2, 11))
    mask = torch.ones_like(x)
    mask[1, -3:] = 0
    y = x.clone()
    y[mask == 0] = -100
    logits = model(input_ids=x, attention_mask=mask).logits
    reference = F.cross_entropy(logits[:, :-1].reshape(-1, 97), y[:, 1:].reshape(-1))
    reference.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual, _ = MemoryTrainingLoss(model, chunk=3)(x, y, mask)
    actual.backward()
    torch.testing.assert_close(actual, reference)
    for n, p in model.named_parameters():
        if n in gradients:
            torch.testing.assert_close(p.grad, gradients[n], atol=2e-7, rtol=1e-4)
    # The second document's logits do not depend on the other document in its batch.
    x[0] = torch.randint(1, 90, (11,))
    changed = model(input_ids=x, attention_mask=mask).logits
    torch.testing.assert_close(changed[1, :8], logits[1, :8])
    solo = model(input_ids=x[1:2, :8]).logits
    torch.testing.assert_close(solo[0], changed[1, :8], atol=1e-6, rtol=1e-5)


def test_checkpoint_selection_is_numeric_complete_and_follows_moved(tmp_path, tiny_model):
    root = tmp_path / "run_name_with_underscores" / "ckpt"
    model = setup_memory(tiny_model)
    opt = torch.optim.AdamW(memory_parameters(model))
    for step in (240, 960, 1200, 2160):
        save_training_checkpoint(model, opt, root / f"step_{step}", {"step": step})
    (root / "step_2160" / "optimizer.pt").unlink()
    assert Path(latest_checkpoint(root)).name == "step_1200"
    moved = tmp_path / "remote_store" / "step_1200"
    moved.parent.mkdir()
    (root / "step_1200").rename(moved)
    (root / "step_1200").mkdir()
    (root / "step_1200" / "MOVED.txt").write_text(str(moved))
    assert latest_checkpoint(root) == str(moved)
    (moved / "COMPLETE").unlink()
    assert Path(latest_checkpoint(root)).name == "step_960"
    with pytest.raises(FileExistsError):
        save_training_checkpoint(model, opt, root / "step_960", {"step": 960})


def test_uneven_microbatch_has_zero_label_padding_and_ends_cleanly():
    rows = iter([{"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[1, 2, 3]])}])
    batches, counts = optimizer_batch(rows, 4, "cpu", 0)
    assert counts.tolist() == [2, 3, 0]
    assert len(batches) == 4
    assert all((b["labels"] == -100).all() for b in batches[1:])
    assert optimizer_batch(rows, 4, "cpu", 0)[0] is None


def test_archive_repairs_partial_copy_before_removing_local_files(tmp_path, tiny_model):
    model = setup_memory(tiny_model)
    opt = torch.optim.AdamW(memory_parameters(model))
    source = tmp_path / "local" / "step_123"
    destination = tmp_path / "archive" / "step_123"
    save_training_checkpoint(model, opt, source, {"step": 123})
    expected = (source / "memory_weights.pt").read_bytes()
    destination.mkdir(parents=True)
    (destination / "memory_weights.pt").write_bytes(b"interrupted copy")
    archive_checkpoint(source, destination, move=True)
    assert (destination / "memory_weights.pt").read_bytes() == expected
    assert (destination / "optimizer.pt").is_file()
    assert (destination / "COMPLETE").is_file()
    assert not (source / "memory_weights.pt").exists()
    assert latest_checkpoint(source.parent) == str(destination)


def test_runtime_checkpoint_requires_and_archives_rank_state(tmp_path, tiny_model):
    model = setup_memory(tiny_model)
    opt = torch.optim.AdamW(memory_parameters(model))
    source, destination = tmp_path / "local/step_4", tmp_path / "archive/step_4"
    runtime = {"format_version": 1, "ranks": [capture_runtime("cpu", 7, 42)]}
    save_training_checkpoint(model, opt, source, {"step": 4}, runtime_state=runtime)
    assert (source / "COMPLETE").read_text() == "3\n"
    archive_checkpoint(source, destination, move=True)
    assert (destination / "COMPLETE").read_text() == "3\n"
    loaded = torch.load(destination / RUNTIME_NAME, weights_only=True)
    assert loaded["ranks"][0]["consumed_batches"] == 7
    assert latest_checkpoint(source.parent) == str(destination)
    (destination / RUNTIME_NAME).unlink()
    assert latest_checkpoint(source.parent) is None


@pytest.mark.parametrize("temperature", [1., 1.7])
def test_cached_teacher_matches_online_loss_and_gradient(tmp_path, temperature):
    torch.manual_seed(21)
    head = torch.nn.Linear(8, 23, bias=False).requires_grad_(False)
    teacher = torch.randn(1, 19, 8)
    student = torch.randn(1, 19, 8, requires_grad=True)
    labels = torch.randint(0, 23, (1, 19))
    labels[:, :13] = -100
    targets = make_topk_targets(teacher, head, labels, 5, temperature, chunk=2)
    online = hidden_ce_and_kl(student, head, labels, teacher_hidden=teacher, teacher_head=head,
                              chunk=3, top_k=5, temperature=temperature)
    expected = torch.autograd.grad(sum(online), student)[0]
    cached = hidden_ce_and_kl(student, head, labels, teacher_topk=targets,
                              chunk=3, top_k=5, temperature=temperature)
    actual = torch.autograd.grad(sum(cached), student)[0]
    torch.testing.assert_close(cached, online)
    torch.testing.assert_close(actual, expected)
    row = {"id": "example"}
    batch = {"input_ids": torch.zeros_like(labels), "labels": labels}
    data = tmp_path / "rows.jsonl"
    data.write_text(json.dumps(row))
    directory = tmp_path / "cache"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(dict(format_version=1, model="tiny",
        data_sha256=file_sha256(data), top_k=5, temperature=temperature)))
    torch.save(dict(indices=targets[0], log_probs=targets[1], token_digest=token_digest(batch)),
               directory / (row_key(row) + ".pt"))
    cache = TeacherCache(directory, data, "tiny", 5, temperature)
    torch.testing.assert_close(cache.get(row, batch), targets)
    batch["input_ids"][0, 0] = 1
    with pytest.raises(ValueError, match="alignment"):
        cache.get(row, batch)
    data.write_text('changed data')
    with pytest.raises(ValueError, match="sha256"):
        TeacherCache(directory, data, "tiny", 5, temperature)
