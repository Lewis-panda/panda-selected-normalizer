from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panda import (  # noqa: E402
    build_standard_blank_target_set,
    dense_selected_reference,
    selected_forward_backward,
    tensor_metrics,
)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _random_tensors(n: int, h: int, c: int, seed: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = (0.07 * torch.randn((n, h), generator=generator, dtype=torch.float32)).to(device)
    weight = (0.05 * torch.randn((c, h), generator=generator, dtype=torch.float32)).to(device)
    bias = (0.01 * torch.randn((c,), generator=generator, dtype=torch.float32)).to(device)
    return generator, hidden, weight, bias


def _assert_close_result(panda, dense, selected_mask):
    active = selected_mask
    assert torch.isfinite(panda.selected_logp[active]).all()
    assert torch.isfinite(panda.logZ).all()
    assert torch.isfinite(panda.grad_hidden).all()
    assert torch.isfinite(panda.grad_weight).all()
    assert torch.isfinite(panda.grad_bias).all()
    assert float((panda.selected_logp[active] - dense.selected_logp[active]).abs().max().cpu()) <= 2e-5
    assert float((panda.logZ - dense.logZ).abs().max().cpu()) <= 2e-5
    assert float((panda.loss - dense.loss).abs().cpu()) <= 2e-5
    for got, ref in (
        (panda.grad_hidden, dense.grad_hidden),
        (panda.grad_weight, dense.grad_weight),
        (panda.grad_bias, dense.grad_bias),
    ):
        metrics = tensor_metrics(got, ref)
        assert metrics["rel_L2"] <= 2e-5
        assert metrics["cosine"] >= 0.999999


def test_standard_blank_target_selected_set_parity():
    device = _device()
    n, h, c, blank_id = 9, 16, 37, 2
    generator, hidden, weight, bias = _random_tensors(n, h, c, 1001, device)
    target_ids = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long, device=device)
    dtarget = (0.2 * torch.randn((n,), generator=generator, dtype=torch.float32)).to(device)
    dblank = (0.2 * torch.randn((n,), generator=generator, dtype=torch.float32)).to(device)
    selected_ids, selected_mask = build_standard_blank_target_set(target_ids, blank_id, num_classes=c)
    adjoints = torch.stack([dtarget, dblank], dim=1)

    panda = selected_forward_backward(hidden, weight, bias, selected_ids, selected_mask, adjoints, tile_c=8)
    dense = dense_selected_reference(hidden, weight, bias, selected_ids, selected_mask, adjoints)
    _assert_close_result(panda, dense, selected_mask)
    assert panda.persistent_full_logits is False
    assert panda.persistent_full_grad_logits is False


def test_invalid_target_omitted_and_zero_adjoint():
    device = _device()
    n, h, c, blank_id = 8, 16, 41, 2
    generator, hidden, weight, bias = _random_tensors(n, h, c, 1002, device)
    target_ids = torch.tensor([3, -1, 5, -1, 7, 8, -1, 10], dtype=torch.long, device=device)
    dtarget = (0.2 * torch.randn((n,), generator=generator, dtype=torch.float32)).to(device)
    dblank = (0.2 * torch.randn((n,), generator=generator, dtype=torch.float32)).to(device)
    dtarget = torch.where(target_ids >= 0, dtarget, torch.zeros_like(dtarget))
    selected_ids, selected_mask = build_standard_blank_target_set(target_ids, blank_id, num_classes=c)
    adjoints = torch.stack([dtarget, dblank], dim=1)
    adjoints = torch.where(selected_mask, adjoints, torch.zeros_like(adjoints))

    assert selected_mask[:, 0].tolist() == [True, False, True, False, True, True, False, True]
    assert torch.count_nonzero(adjoints[target_ids < 0, 0]).item() == 0

    panda = selected_forward_backward(hidden, weight, bias, selected_ids, selected_mask, adjoints, tile_c=8)
    dense = dense_selected_reference(hidden, weight, bias, selected_ids, selected_mask, adjoints)
    _assert_close_result(panda, dense, selected_mask)


def test_target_equal_blank_rejected():
    target_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    try:
        build_standard_blank_target_set(target_ids, blank_id=2, num_classes=8)
    except ValueError as exc:
        assert "distinct from blank_id" in str(exc)
    else:
        raise AssertionError("target_id == blank_id should be rejected")


def test_multiselected_set_parity():
    device = _device()
    n, h, c, blank_id = 10, 16, 47, 2
    generator, hidden, weight, bias = _random_tensors(n, h, c, 1003, device)
    selected_ids = torch.full((n, 5), -1, dtype=torch.long)
    selected_mask = torch.zeros((n, 5), dtype=torch.bool)
    adjoints = torch.zeros((n, 5), dtype=torch.float32)
    for i in range(n):
        selected = [3 + i, blank_id, 20 + i % 7]
        if i % 2 == 0:
            selected.append(30 + i % 5)
        if i % 3 == 0:
            selected.append(40 + i % 3)
        assert len(selected) == len(set(selected))
        assert len(selected) >= 3
        for k, class_id in enumerate(selected):
            selected_ids[i, k] = class_id
            selected_mask[i, k] = True
            adjoints[i, k] = 0.2 * torch.randn((), generator=generator, dtype=torch.float32)
    selected_ids = selected_ids.to(device)
    selected_mask = selected_mask.to(device)
    adjoints = adjoints.to(device)

    sizes = selected_mask.sum(dim=1)
    assert int(sizes.min().cpu()) >= 3
    assert int(sizes.max().cpu()) <= 5

    panda = selected_forward_backward(hidden, weight, bias, selected_ids, selected_mask, adjoints, tile_c=8)
    dense = dense_selected_reference(hidden, weight, bias, selected_ids, selected_mask, adjoints)
    _assert_close_result(panda, dense, selected_mask)


if __name__ == "__main__":
    test_standard_blank_target_selected_set_parity()
    test_invalid_target_omitted_and_zero_adjoint()
    test_target_equal_blank_rejected()
    test_multiselected_set_parity()
    print("selected_normalizer_parity tests passed")
