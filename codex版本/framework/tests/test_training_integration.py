"""Offline end-to-end tests with a tiny Qwen3 and real CPU/Gloo DDP workers."""
import json
import os
from pathlib import Path
import re
import subprocess
import socket
import sys

import pytest
import torch

from conftest import tiny_qwen3

TRAIN = Path(__file__).resolve().parents[1] / "scripts/train_memory.py"


@pytest.fixture
def offline_model(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    path = tmp_path / "base_model"
    tiny_qwen3().save_pretrained(path)
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, **{f"t{i}": i + 3 for i in range(94)}}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                        eos_token="<eos>", unk_token="<unk>")
    tokenizer.save_pretrained(path)
    return path


def train(model, data, output, *, world=1, accum=2, extra=(), expected_error=None):
    command = [sys.executable]
    if world > 1:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        command += ["-m", "torch.distributed.run", "--master-addr=127.0.0.1",
                    f"--master-port={port}", f"--nproc-per-node={world}"]
    command += [str(TRAIN), "--model", str(model), "--data", data, "--out", str(output),
        "--device", os.environ.get("MEMT_TEST_DEVICE", "cpu"), "--dtype", "float32",
        "--attn-impl", "eager", "--memory", "new",
        "--seq-len", "16", "--steps", "3", "--warmup", "0", "--lr", "0.002",
        "--window", "4", "--sinks", "0", "--attn-segment", "4", "--mem-chunk", "4",
        "--mem-core", "matrix", "--mem-heads", "4", "--mem-head-dim", "16",
        "--mem-write-block", "2", "--mem-qkv-init", "attention", "--mem-qkv-act", "none",
        "--mem-inject", "pre_o_proj", "--mem-out-scale", "memt", "--mem-output-gate-rank", "8",
        "--mem-forget-rank", "8", "--mem-stable-matrix-gates", "--grad-ckpt",
        "--grad-accum", str(accum), "--save-every", "1", "--log-every", "1",
        "--shuffle-buffer", "0", "--exact-text-resume", "--seed", "42",
        "--loss-chunk-size", "3", "--fail-on-nonfinite", *map(str, extra)]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    if expected_error is None:
        assert result.returncode == 0, result.stdout[-12000:]
    else:
        assert result.returncode != 0 and expected_error in result.stdout, result.stdout[-12000:]
    return result.stdout


def weights(output):
    return torch.load(output / "final" / "memory_weights.pt", map_location="cpu", weights_only=True)


def test_interrupted_text_resume_matches_uninterrupted(offline_model, tmp_path):
    data = tmp_path / "text.jsonl"
    data.write_text('\n'.join(json.dumps({"text": " ".join(f"t{i}" for i in range(n))})
                              for n in (12, 6, 19, 8, 4)))
    uninterrupted, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    train(offline_model, f"text:{data}", uninterrupted)
    train(offline_model, f"text:{data}", resumed, extra=["--stop-at", "1"])
    assert not (resumed / "final").exists()
    train(offline_model, f"text:{data}", resumed, extra=["--resume", resumed / "step_1"])
    for key, expected in weights(uninterrupted).items():
        torch.testing.assert_close(weights(resumed)[key], expected, atol=0, rtol=0)
    final = json.loads((resumed / "final/memory_config.json").read_text())
    assert final["step"] == 3 and final["processed_before_resume_known"]
    assert final["processed_tokens"] == json.loads(
        (uninterrupted / "final/memory_config.json").read_text())["processed_tokens"]


@pytest.mark.parametrize("rows", [5, 7])
def test_ddp_uneven_post_data_matches_global_token_mean(offline_model, tmp_path, rows):
    data = tmp_path / "post.jsonl"
    data.write_text('\n'.join(json.dumps(dict(id=f"row-{i}", history_ids=list(range(3+i, 11+i)),
        query_ids=[20+i], response_ids=list(range(30+i, 31+i+i%3)), teacher_generated=True))
        for i in range(rows)))
    single, distributed = tmp_path / "single", tmp_path / "distributed"
    extra = ["--ce-weight", "0", "--distill-weight", "1", "--distill-top-k", "5"]
    train(offline_model, f"post:{data}", single, accum=4, extra=extra)
    train(offline_model, f"post:{data}", distributed, world=2, accum=2, extra=extra)
    final = json.loads((distributed / "final/memory_config.json").read_text())
    assert final["step"] == 2 and final["data_exhausted"]
    assert final["processed_tokens"] == sum(10+i%3 for i in range(rows))
    actual = weights(distributed)
    for key, expected in weights(single).items():
        torch.testing.assert_close(actual[key], expected, atol=2e-5, rtol=2e-4, msg=key)


def test_offline_teacher_cache_trains_like_online_teacher(offline_model, tmp_path):
    data = tmp_path / "post.jsonl"
    data.write_text('\n'.join(json.dumps(dict(id=f"row-{i}", history_ids=list(range(3+i, 11+i)),
        query_ids=[20+i], response_ids=[30+i, 31+i], teacher_generated=True)) for i in range(3)))
    cache = tmp_path / "cache"
    result = subprocess.run([sys.executable, str(TRAIN.parent / "cache_teacher.py"),
        "--model", str(offline_model), "--data", str(data), "--out", str(cache),
        "--segment", "4", "--max-length", "16", "--top-k", "5", "--temperature", "1.7",
        "--dtype", "float32", "--device", "cpu"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=60,
        env={**os.environ, "OMP_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1"})
    assert result.returncode == 0, result.stdout
    extra = ["--ce-weight", "0", "--distill-weight", "1", "--distill-top-k", "5",
             "--distill-temperature", "1.7"]
    online, cached = tmp_path / "online", tmp_path / "cached"
    train(offline_model, f"post:{data}", online, extra=extra)
    train(offline_model, f"post:{data}", cached,
          extra=[*extra, "--teacher-cache", cache, "--stop-at", 1])
    train(offline_model, f"post:{data}", cached,
          extra=[*extra, "--teacher-cache", cache, "--resume", cached / "step_1"])
    actual = weights(cached)
    for key, expected in weights(online).items():
        torch.testing.assert_close(actual[key], expected, atol=2e-6, rtol=2e-5, msg=key)


def post_rows(path, count):
    path.write_text('\n'.join(json.dumps(dict(id=f"row-{i}", history_ids=list(range(3+i, 11+i)),
        query_ids=[20+i], response_ids=[30+i, 31+i], teacher_generated=True)) for i in range(count)))


POST_ARGS = ["--ce-weight", "0", "--distill-weight", "1", "--distill-top-k", "5"]


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_ddp_post_resume_preserves_dropout_rng_cursor_optimizer_and_logs(offline_model, tmp_path):
    config_file = offline_model / "config.json"
    config = json.loads(config_file.read_text())
    config["attention_dropout"] = 0.2
    config_file.write_text(json.dumps(config))
    data = tmp_path / "post.jsonl"
    post_rows(data, 7)
    uninterrupted, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    expected_log = train(offline_model, f"post:{data}", uninterrupted, world=2, extra=POST_ARGS)
    train(offline_model, f"post:{data}", resumed, world=2, extra=[*POST_ARGS, "--stop-at", 1])
    actual_log = train(offline_model, f"post:{data}", resumed, world=2,
                       extra=[*POST_ARGS, "--resume", resumed / "step_1", "--log-every", 2])
    actual = weights(resumed)
    for name, expected in weights(uninterrupted).items():
        torch.testing.assert_close(actual[name], expected, atol=0, rtol=0, msg=name)
    for filename in ("optimizer.pt", "training_state.pt"):
        left = torch.load(uninterrupted / "final" / filename, weights_only=True)
        right = torch.load(resumed / "final" / filename, weights_only=True)
        assert_state_equal(left, right)
    runtime = torch.load(resumed / "final/training_state.pt", weights_only=True)
    assert [s["consumed_batches"] for s in runtime["ranks"]] == [4, 3]
    assert not torch.equal(runtime["ranks"][0]["torch_rng"], runtime["ranks"][1]["torch_rng"])
    metadata = json.loads((resumed / "final/memory_config.json").read_text())
    assert metadata["training_version"] == 3 and metadata["data_exhausted"]
    pattern = r"step 2 loss ([\d.]+) kl ([\d.]+)"
    assert re.search(pattern, actual_log).groups() == re.search(pattern, expected_log).groups()


def test_post_resume_rejects_changed_source_and_empty_post_never_succeeds(offline_model, tmp_path):
    data, output = tmp_path / "post.jsonl", tmp_path / "run"
    post_rows(data, 3)
    train(offline_model, f"post:{data}", output, extra=[*POST_ARGS, "--stop-at", 1])
    post_rows(data, 4)
    train(offline_model, f"post:{data}", output,
          extra=[*POST_ARGS, "--resume", output / "step_1"], expected_error="source SHA-256 changed")
    assert not (output / "final").exists()
    data.write_text("")
    empty_output = tmp_path / "empty_run"
    train(offline_model, f"post:{data}", empty_output, extra=POST_ARGS,
          expected_error="contains no supervised examples")
    assert not (empty_output / "final").exists()


def test_ddp_checkpoint_writer_failure_reaches_every_rank(offline_model, tmp_path):
    data, output = tmp_path / "post.jsonl", tmp_path / "run"
    post_rows(data, 5)
    (output / "step_1").mkdir(parents=True)
    log = train(offline_model, f"post:{data}", output, world=2, extra=POST_ARGS,
                expected_error="checkpoint publication failed on a rank")
    assert log.count("RuntimeError: checkpoint publication failed on a rank") == 2
    assert not (output / "final").exists()
