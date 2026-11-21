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


def test_topk_kl_matches_direct_computation():
    torch.manual_seed(2)
    head = torch.nn.Linear(8, 23, bias=False)
    hidden = torch.randn(1, 12, 8)
    logits = torch.randn(1, 12, 23)
    labels = torch.randint(0, 23, (1, 12))
    _, kl = ce_and_kl(logits, hidden, head, labels, chunk=5, topk=4)
    top, idx = head(hidden)[:, :-1].topk(4, dim=-1)
    p = top.softmax(-1)
    log_s = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, idx)
    torch.testing.assert_close(kl, (p * (p.log() - log_s)).sum() / 11, rtol=1e-4, atol=1e-5)
    # all tokens kept: the same as no truncation
    _, full = ce_and_kl(logits, hidden, head, labels, chunk=5)
    _, all_k = ce_and_kl(logits, hidden, head, labels, chunk=5, topk=23)
    torch.testing.assert_close(all_k, full, rtol=1e-4, atol=1e-5)


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
