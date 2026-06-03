#!/usr/bin/env python3
"""Minimal PANDA selected-normalizer smoke on random tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panda import (  # noqa: E402
    build_standard_blank_target_set,
    cuda_peak_mib,
    dense_selected_reference,
    reset_cuda_peak,
    selected_forward_backward,
    tensor_metrics,
)


def make_random_standard_case(n: int, h: int, c: int, blank_id: int, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(1234 + c + n + h)
    hidden = (0.07 * torch.randn((n, h), generator=generator, dtype=torch.float32)).to(device)
    weight = (0.05 * torch.randn((c, h), generator=generator, dtype=torch.float32)).to(device)
    bias = (0.01 * torch.randn((c,), generator=generator, dtype=torch.float32)).to(device)
    target_cpu = torch.randint(low=0, high=c - 1, size=(n,), generator=generator, dtype=torch.long)
    target_cpu = torch.where(target_cpu >= blank_id, target_cpu + 1, target_cpu)
    if n >= 8:
        target_cpu[::17] = -1
    target_ids = target_cpu.to(device)
    dtarget = (0.2 * torch.randn((n,), generator=generator, dtype=torch.float32)).to(device)
    dblank = (0.2 * torch.randn((n,), generator=generator, dtype=torch.float32)).to(device)
    dtarget = torch.where(target_ids >= 0, dtarget, torch.zeros_like(dtarget))
    selected_ids, selected_mask = build_standard_blank_target_set(target_ids, blank_id, num_classes=c)
    selected_adjoints = torch.stack([dtarget, dblank], dim=1)
    selected_adjoints = torch.where(selected_mask, selected_adjoints, torch.zeros_like(selected_adjoints))
    return hidden, weight, bias, selected_ids, selected_mask, selected_adjoints


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

    hidden, weight, bias, selected_ids, selected_mask, selected_adjoints = make_random_standard_case(
        args.N,
        args.H,
        args.C,
        args.blank_id,
        device,
    )

    reset_cuda_peak()
    panda = selected_forward_backward(
        hidden,
        weight,
        bias,
        selected_ids,
        selected_mask,
        selected_adjoints,
        tile_c=args.tile_c,
    )
    panda_peak = cuda_peak_mib()

    reset_cuda_peak()
    dense = dense_selected_reference(hidden, weight, bias, selected_ids, selected_mask, selected_adjoints)
    dense_peak = cuda_peak_mib()

    active = selected_mask
    selected_logp_max_abs = float((panda.selected_logp[active] - dense.selected_logp[active]).abs().max().cpu().item())
    logz_max_abs = float((panda.logZ - dense.logZ).abs().max().cpu().item())
    loss_abs = float((panda.loss - dense.loss).abs().cpu().item())
    loss_rel = loss_abs / max(abs(float(dense.loss.cpu().item())), 1e-12)

    print("PANDA minimal selected-normalizer smoke")
    print(f"C={args.C} (output class count / vocabulary size), N={args.N}, H={args.H}, device={device}")
    print("contract=standard blank/target selected set")
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
