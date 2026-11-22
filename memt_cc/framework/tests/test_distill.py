import torch
import torch.nn.functional as F

from lact_qwen.distill import ce_and_kl


def test_ce_matches_cross_entropy_and_kl_is_zero_for_identical_models():
    torch.manual_seed(0)
    head = torch.nn.Linear(8, 23, bias=False)
    hidden = torch.randn(2, 37, 8)
    logits = head(hidden).detach().requires_grad_(True)
    labels = torch.randint(0, 23, (2, 37))
    labels[0, :5] = -100
    ce, kl = ce_and_kl(logits, hidden, head, labels, chunk=7)
    ref = F.cross_entropy(logits[:, :-1].reshape(-1, 23), labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(ce, ref, rtol=1e-5, atol=1e-5)
    assert abs(float(kl)) < 1e-5


def test_kl_is_positive_has_gradient_and_ignores_unlabelled_positions():
    torch.manual_seed(1)
    head = torch.nn.Linear(8, 23, bias=False)
    hidden = torch.randn(1, 20, 8)
    logits = torch.randn(1, 20, 23, requires_grad=True)
    labels = torch.full((1, 20), -100)
    labels[0, 10:] = torch.randint(0, 23, (10,))
    ce, kl = ce_and_kl(logits, hidden, head, labels, chunk=4)
    assert float(kl) > 0
    kl.backward()
    assert logits.grad[0, 9:19].abs().sum() > 0  # positions predicting labelled tokens
    assert logits.grad[0, :9].abs().sum() == 0 and logits.grad[0, 19].abs().sum() == 0
    assert head.weight.grad is None  # nothing flows into the teacher
