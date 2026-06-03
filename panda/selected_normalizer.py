"""Exact full-C selected normalization without persistent full logits.

This module is intentionally small and readable. It uses PyTorch tensor
operations in C tiles and a manual backward formula so users can inspect the
PANDA boundary without depending on the research repository or a Triton build.

The implementation computes the exact full-C denominator by scanning every
class tile. It may allocate temporary tile logits shaped [N, tile_C], but it
does not keep persistent [N, C] logits or persistent [N, C] grad_logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SelectedNormalizerResult:
    selected_logp: torch.Tensor
    logZ: torch.Tensor
    largest_logits_tile_shape: tuple[int, int]
    persistent_full_logits: bool = False
    persistent_full_grad_logits: bool = False


@dataclass(frozen=True)
class SelectedBackwardResult:
    grad_hidden: torch.Tensor
    grad_weight: torch.Tensor
    grad_bias: torch.Tensor
    largest_logits_tile_shape: tuple[int, int]
    persistent_full_logits: bool = False
    persistent_full_grad_logits: bool = False


@dataclass(frozen=True)
class SelectedForwardBackwardResult:
    selected_logp: torch.Tensor
    logZ: torch.Tensor
    loss: torch.Tensor
    grad_hidden: torch.Tensor
    grad_weight: torch.Tensor
    grad_bias: torch.Tensor
    largest_logits_tile_shape: tuple[int, int]
    persistent_full_logits: bool = False
    persistent_full_grad_logits: bool = False


def _check_2d(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape {tuple(value.shape)}")


def validate_selected_set(
    selected_ids: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    num_classes: int,
    reject_duplicates: bool = True,
) -> None:
    """Validate compact selected-emission ids.

    Inactive entries may contain any value, but active entries must be in
    [0, num_classes). Active selected classes within each site must be distinct
    by default because duplicate selected emissions make compact adjoints
    ambiguous.
    """

    _check_2d("selected_ids", selected_ids)
    _check_2d("selected_mask", selected_mask)
    if selected_ids.shape != selected_mask.shape:
        raise ValueError(
            "selected_ids and selected_mask must have the same shape, "
            f"got {tuple(selected_ids.shape)} and {tuple(selected_mask.shape)}"
        )
    if selected_mask.dtype != torch.bool:
        raise TypeError(f"selected_mask must be bool, got {selected_mask.dtype}")
    active = selected_mask
    if active.any():
        active_ids = selected_ids[active]
        if bool(((active_ids < 0) | (active_ids >= num_classes)).any().item()):
            raise ValueError("active selected ids must be in [0, C)")
    if reject_duplicates:
        ids_cpu = selected_ids.detach().cpu()
        mask_cpu = selected_mask.detach().cpu()
        for row, mask in zip(ids_cpu.tolist(), mask_cpu.tolist()):
            active_row = [int(item) for item, keep in zip(row, mask) if keep]
            if len(active_row) != len(set(active_row)):
                raise ValueError("active selected ids must be distinct within each site")


def build_standard_blank_target_set(
    target_ids: torch.Tensor,
    blank_id: int,
    *,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build selected set S_i={target_i, blank} with invalid targets omitted.

    The first column is the target edge and the second column is the blank edge.
    If target_ids[i] < 0, the target edge is inactive. Valid target ids must not
    equal blank_id; standard RNN-T treats blank and transcript labels as
    distinct emission classes.
    """

    if target_ids.ndim != 1:
        raise ValueError(f"target_ids must be rank-1, got {tuple(target_ids.shape)}")
    if not 0 <= int(blank_id) < int(num_classes):
        raise ValueError("blank_id must be in [0, C)")

    valid = target_ids >= 0
    if valid.any():
        valid_targets = target_ids[valid]
        if bool((valid_targets == int(blank_id)).any().item()):
            raise ValueError("valid target ids must be distinct from blank_id")
        if bool(((valid_targets < 0) | (valid_targets >= num_classes)).any().item()):
            raise ValueError("valid target ids must be in [0, C)")

    selected_ids = torch.empty((target_ids.numel(), 2), dtype=torch.long, device=target_ids.device)
    selected_ids[:, 0] = torch.where(valid, target_ids, torch.zeros_like(target_ids))
    selected_ids[:, 1] = int(blank_id)
    selected_mask = torch.empty((target_ids.numel(), 2), dtype=torch.bool, device=target_ids.device)
    selected_mask[:, 0] = valid
    selected_mask[:, 1] = True
    return selected_ids, selected_mask


def selected_log_probs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_mask: torch.Tensor,
    *,
    tile_c: int = 1024,
) -> SelectedNormalizerResult:
    """Compute selected log-probabilities and exact full-C logZ.

    Args:
        hidden: [N, H] float tensor.
        weight: [C, H] float tensor.
        bias: [C] float tensor.
        selected_ids: [N, S_max] compact selected class ids.
        selected_mask: [N, S_max] active selected-entry mask.
        tile_c: class tile size for exact C scan.
    """

    _check_2d("hidden", hidden)
    _check_2d("weight", weight)
    if bias.ndim != 1:
        raise ValueError(f"bias must be rank-1, got {tuple(bias.shape)}")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError(f"H mismatch: hidden has {hidden.shape[1]}, weight has {weight.shape[1]}")
    if bias.shape[0] != weight.shape[0]:
        raise ValueError(f"C mismatch: weight has {weight.shape[0]}, bias has {bias.shape[0]}")
    if selected_ids.shape[0] != hidden.shape[0]:
        raise ValueError("selected_ids first dimension must match hidden N")
    if tile_c <= 0:
        raise ValueError("tile_c must be positive")
    validate_selected_set(selected_ids, selected_mask, num_classes=weight.shape[0])

    n_sites = hidden.shape[0]
    num_classes = weight.shape[0]
    with torch.no_grad():
        logZ = torch.full((n_sites,), -torch.inf, dtype=torch.float32, device=hidden.device)
        largest_tile = 0
        hidden_f = hidden.to(torch.float32)
        weight_f = weight.to(torch.float32)
        bias_f = bias.to(torch.float32)
        for start in range(0, num_classes, tile_c):
            end = min(start + tile_c, num_classes)
            largest_tile = max(largest_tile, end - start)
            logits_tile = hidden_f @ weight_f[start:end].T + bias_f[start:end]
            logZ = torch.logaddexp(logZ, torch.logsumexp(logits_tile, dim=1))

        safe_ids = selected_ids.clamp(min=0)
        selected_logits = (hidden_f[:, None, :] * weight_f[safe_ids]).sum(dim=2) + bias_f[safe_ids]
        selected_logp = selected_logits - logZ[:, None]
        selected_logp = torch.where(selected_mask, selected_logp, torch.zeros_like(selected_logp))

    return SelectedNormalizerResult(
        selected_logp=selected_logp,
        logZ=logZ,
        largest_logits_tile_shape=(int(n_sites), int(largest_tile)),
    )


def selected_backward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_mask: torch.Tensor,
    selected_adjoints: torch.Tensor,
    logZ: torch.Tensor,
    *,
    tile_c: int = 1024,
) -> SelectedBackwardResult:
    """Manual backward from compact selected-emission adjoints.

    selected_adjoints[i, s] is d loss / d selected_logp[i, s]. Inactive entries
    are ignored by selected_mask. The backward recomputes logits tile-by-tile
    and accumulates directly into hidden/head gradients.
    """

    _check_2d("hidden", hidden)
    _check_2d("weight", weight)
    _check_2d("selected_adjoints", selected_adjoints)
    if selected_adjoints.shape != selected_ids.shape:
        raise ValueError("selected_adjoints must have the same shape as selected_ids")
    if logZ.shape != (hidden.shape[0],):
        raise ValueError(f"logZ must have shape [N], got {tuple(logZ.shape)}")
    validate_selected_set(selected_ids, selected_mask, num_classes=weight.shape[0])

    n_sites = hidden.shape[0]
    num_classes = weight.shape[0]
    with torch.no_grad():
        hidden_f = hidden.to(torch.float32)
        weight_f = weight.to(torch.float32)
        bias_f = bias.to(torch.float32)
        adjoints_f = torch.where(selected_mask, selected_adjoints.to(torch.float32), torch.zeros_like(selected_adjoints, dtype=torch.float32))
        selected_total = adjoints_f.sum(dim=1)

        grad_hidden = torch.zeros_like(hidden_f)
        grad_weight = torch.zeros_like(weight_f)
        grad_bias = torch.zeros_like(bias_f)
        largest_tile = 0
        for start in range(0, num_classes, tile_c):
            end = min(start + tile_c, num_classes)
            largest_tile = max(largest_tile, end - start)
            logits_tile = hidden_f @ weight_f[start:end].T + bias_f[start:end]
            probs_tile = torch.exp(logits_tile - logZ[:, None])
            grad_logits_tile = -selected_total[:, None] * probs_tile
            for k in range(selected_ids.shape[1]):
                ids = selected_ids[:, k]
                active = selected_mask[:, k] & (ids >= start) & (ids < end)
                if bool(active.any().item()):
                    rows = torch.nonzero(active, as_tuple=False).squeeze(1)
                    cols = ids[rows] - start
                    grad_logits_tile[rows, cols] += adjoints_f[rows, k]
            grad_hidden += grad_logits_tile @ weight_f[start:end]
            grad_weight[start:end] += grad_logits_tile.T @ hidden_f
            grad_bias[start:end] += grad_logits_tile.sum(dim=0)

    return SelectedBackwardResult(
        grad_hidden=grad_hidden,
        grad_weight=grad_weight,
        grad_bias=grad_bias,
        largest_logits_tile_shape=(int(n_sites), int(largest_tile)),
    )


def selected_forward_backward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_mask: torch.Tensor,
    selected_adjoints: torch.Tensor,
    *,
    tile_c: int = 1024,
) -> SelectedForwardBackwardResult:
    forward = selected_log_probs(
        hidden,
        weight,
        bias,
        selected_ids,
        selected_mask,
        tile_c=tile_c,
    )
    selected_adjoints_f = torch.where(
        selected_mask,
        selected_adjoints.to(torch.float32),
        torch.zeros_like(selected_adjoints, dtype=torch.float32),
    )
    loss = (selected_adjoints_f * forward.selected_logp).sum()
    backward = selected_backward(
        hidden,
        weight,
        bias,
        selected_ids,
        selected_mask,
        selected_adjoints,
        forward.logZ,
        tile_c=tile_c,
    )
    largest_tile = max(forward.largest_logits_tile_shape[1], backward.largest_logits_tile_shape[1])
    return SelectedForwardBackwardResult(
        selected_logp=forward.selected_logp,
        logZ=forward.logZ,
        loss=loss,
        grad_hidden=backward.grad_hidden,
        grad_weight=backward.grad_weight,
        grad_bias=backward.grad_bias,
        largest_logits_tile_shape=(int(hidden.shape[0]), int(largest_tile)),
    )


def standard_blank_target_forward_backward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    target_ids: torch.Tensor,
    blank_id: int,
    dtarget: torch.Tensor,
    dblank: torch.Tensor,
    *,
    tile_c: int = 1024,
) -> SelectedForwardBackwardResult:
    selected_ids, selected_mask = build_standard_blank_target_set(
        target_ids,
        blank_id,
        num_classes=weight.shape[0],
    )
    selected_adjoints = torch.stack([dtarget, dblank], dim=1).to(torch.float32)
    selected_adjoints = torch.where(selected_mask, selected_adjoints, torch.zeros_like(selected_adjoints))
    return selected_forward_backward(
        hidden,
        weight,
        bias,
        selected_ids,
        selected_mask,
        selected_adjoints,
        tile_c=tile_c,
    )


def tensor_metrics(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    got_f = got.detach().reshape(-1).to(dtype=torch.float64)
    ref_f = ref.detach().reshape(-1).to(dtype=torch.float64)
    diff = got_f - ref_f
    ref_norm = torch.linalg.vector_norm(ref_f)
    got_norm = torch.linalg.vector_norm(got_f)
    diff_norm = torch.linalg.vector_norm(diff)
    return {
        "max_abs": float(diff.abs().max().cpu().item()),
        "rel_L2": float((diff_norm / torch.clamp(ref_norm, min=1e-12)).cpu().item()),
        "cosine": float((torch.dot(got_f, ref_f) / torch.clamp(got_norm * ref_norm, min=1e-12)).cpu().item()),
    }


def cuda_peak_mib() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"allocated": None, "reserved": None}
    torch.cuda.synchronize()
    return {
        "allocated": float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0),
        "reserved": float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0),
    }


def reset_cuda_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
