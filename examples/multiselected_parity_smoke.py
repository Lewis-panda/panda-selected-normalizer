#!/usr/bin/env python3
"""Synthetic |S_i|=3..5 multi-selected parity smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panda import cuda_peak_mib, dense_selected_reference, reset_cuda_peak, selected_forward_backward, tensor_metrics  # noqa: E402


MAX_SELECTED = 5


def make_multiselected_case(n: int, h: int, c: int, blank_id: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(20260603 + c + n + h)
    hidden = (0.07 * torch.randn((n, h), generator=generator, dtype=torch.float32)).to(device)
    weight = (0.05 * torch.randn((c, h), generator=generator, dtype=torch.float32)).to(device)
    bias = (0.01 * torch.randn((c,), generator=generator, dtype=torch.float32)).to(device)
    selected_ids = torch.full((n, MAX_SELECTED), -1, dtype=torch.long)
    selected_mask = torch.zeros((n, MAX_SELECTED), dtype=torch.bool)
    adjoints = torch.zeros((n, MAX_SELECTED), dtype=torch.float32)
    duplicate_repairs = 0

    def add_distinct(existing: list[int], candidate: int) -> tuple[int, int]:
        repairs = 0
        candidate %= c
        while candidate in existing or candidate == blank_id:
            candidate = (candidate + 1) % c
            repairs += 1
            if repairs > c:
                raise RuntimeError("could not build a distinct selected set")
        return candidate, repairs

    invalid_target_sites = 0
    for i in range(n):
        invalid_target = i % 29 == 0
        selected: list[int] = []
        if not invalid_target:
            target, repairs = add_distinct([blank_id], 7 + ((i * 37 + 11) % (c - 7)))
            duplicate_repairs += repairs
            selected.append(target)
        else:
            invalid_target_sites += 1
        selected.append(blank_id)
        num_extra = 1 + (i % 3)
        if invalid_target:
            num_extra = max(2, num_extra)
        for j in range(num_extra):
            extra, repairs = add_distinct(selected, 3 + ((i * 131 + j * 17 + 5) % (c - 3)))
            duplicate_repairs += repairs
            selected.append(extra)
        if len(selected) <= 2:
            raise AssertionError("|S_i| must be greater than 2")
        for k, class_id in enumerate(selected):
            selected_ids[i, k] = class_id
            selected_mask[i, k] = True
            adjoints[i, k] = 0.2 * torch.randn((), generator=generator, dtype=torch.float32)

    return (
        hidden,
        weight,
        bias,
        selected_ids.to(device),
        selected_mask.to(device),
        adjoints.to(device),
        invalid_target_sites,
        duplicate_repairs,
    )


def fmt_metric(metrics: dict[str, float]) -> str:
    return (
        f"max_abs={metrics['max_abs']:.6e}, "
        f"rel_L2={metrics['rel_L2']:.6e}, "
        f"cosine={metrics['cosine']:.12f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--C", type=int, default=2048, help="output class count / vocabulary size")
    parser.add_argument("--blank-id", type=int, default=2)
    parser.add_argument("--tile-c", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    hidden, weight, bias, selected_ids, selected_mask, adjoints, invalid_target_sites, duplicate_repairs = make_multiselected_case(
        args.N,
        args.H,
        args.C,
        args.blank_id,
        device,
    )

    reset_cuda_peak()
    panda = selected_forward_backward(hidden, weight, bias, selected_ids, selected_mask, adjoints, tile_c=args.tile_c)
    panda_peak = cuda_peak_mib()

    reset_cuda_peak()
    dense = dense_selected_reference(hidden, weight, bias, selected_ids, selected_mask, adjoints)
    dense_peak = cuda_peak_mib()

    active = selected_mask
    selected_sizes = selected_mask.sum(dim=1).detach().cpu()
    selected_logp_max_abs = float((panda.selected_logp[active] - dense.selected_logp[active]).abs().max().cpu().item())
    logz_max_abs = float((panda.logZ - dense.logZ).abs().max().cpu().item())
    loss_abs = float((panda.loss - dense.loss).abs().cpu().item())
    loss_rel = loss_abs / max(abs(float(dense.loss.cpu().item())), 1e-12)

    print("PANDA multi-selected parity smoke")
    print(f"C={args.C} (output class count / vocabulary size), N={args.N}, H={args.H}, device={device}")
    print("contract=synthetic multi-selected / multi-blank-style second instance")
    print("production_multi_blank_or_tdt_support=false")
    print(f"selected_size_min_max={int(selected_sizes.min())}..{int(selected_sizes.max())}")
    print(f"invalid_target_sites={invalid_target_sites}")
    print(f"duplicate_policy=reject_and_regenerate, duplicate_repairs={duplicate_repairs}")
    print("denominator=exact full-C")
    print(f"panda_largest_logits_tile_shape={panda.largest_logits_tile_shape}")
    print(f"dense_full_logits_shape={dense.full_logits_shape}")
    print(f"selected_logp_max_abs={selected_logp_max_abs:.6e}")
    print(f"logZ_max_abs={logz_max_abs:.6e}")
    print(f"loss_abs={loss_abs:.6e}, loss_rel={loss_rel:.6e}")
    print(f"grad_hidden: {fmt_metric(tensor_metrics(panda.grad_hidden, dense.grad_hidden))}")
    print(f"grad_weight: {fmt_metric(tensor_metrics(panda.grad_weight, dense.grad_weight))}")
    print(f"grad_bias: {fmt_metric(tensor_metrics(panda.grad_bias, dense.grad_bias))}")
    print(f"panda_peak_allocated_reserved_MiB={panda_peak['allocated']} / {panda_peak['reserved']}")
    print(f"dense_peak_allocated_reserved_MiB={dense_peak['allocated']} / {dense_peak['reserved']}")
    print(f"panda_persistent_full_logits={str(panda.persistent_full_logits).lower()}")
    print(f"panda_persistent_full_grad_logits={str(panda.persistent_full_grad_logits).lower()}")
    print("training_decode_wer_cer_run=false")


if __name__ == "__main__":
    main()
