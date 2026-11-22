import random

import numpy as np
import pytest
import torch

from lact_qwen.training_state import capture_runtime, restore_rng, replay_iterator


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for the device RNG snapshot"))])
def test_rng_snapshot_roundtrip_is_safe_for_weights_only_loading(tmp_path, device):
    original = capture_runtime(device, 0, 0)
    try:
        random.seed(71)
        np.random.seed(72)
        torch.manual_seed(73)
        state = capture_runtime(device, 17, 5)
        def draws():
            return random.random(), np.random.standard_normal(7), torch.randn(7, device=device)
        expected = draws()
        target = tmp_path / "runtime.pt"
        torch.save(state, target)
        draws()
        loaded = torch.load(target, weights_only=True)
        assert loaded["consumed_batches"] == 17 and loaded["data_seed"] == 5
        restore_rng(loaded, device)
        actual = draws()
        assert actual[0] == expected[0]
        np.testing.assert_array_equal(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    finally:
        restore_rng(original, device)


def test_cursor_replay_retains_remaining_examples_and_rejects_short_data():
    source = ["first", "second", "third", "fourth"]
    stream = replay_iterator(iter(source), 2)
    assert next(stream) == "third" and stream.consumed == 3
    assert list(stream) == ["fourth"] and stream.consumed == 4
    with pytest.raises(RuntimeError, match="cursor exceeds"):
        replay_iterator(iter(source), 5)
    with pytest.raises(RuntimeError, match="invalid checkpoint data cursor"):
        replay_iterator(iter(source), -1)
