"""Dense full-C PyTorch oracle for PANDA smoke tests.

This module intentionally materializes logits[N, C] and uses autograd. It is
only a correctness oracle for small random-tensor examples and tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .selected_normalizer import build_standard_blank_target_set, validate_selected_set


@dataclass(frozen=True)
class DenseReferenceResult:
    selected_logp: torch.Tensor
    logZ: torch.Tensor
    loss: torch.Tensor
    grad_hidden: torch.Tensor
    grad_weight: torch.Tensor
    grad_bias: torch.Tensor
    full_logits_shape: tuple[int, int]
    materializes_full_logits: bool = True
    uses_autograd: bool = True


def dense_selected_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    selected_ids: torch.Tensor,
    selected_mask: torch.Tensor,
    selected_adjoints: torch.Tensor,
) -> DenseReferenceResult:
    """Materialized full-C selected-logprob reference."""

    validate_selected_set(selected_ids, selected_mask, num_classes=weight.shape[0])
    if selected_adjoints.shape != selected_ids.shape:
        raise ValueError("selected_adjoints must have the same shape as selected_ids")

    hidden_ref = hidden.detach().clone().to(torch.float32).requires_grad_(True)
    weight_ref = weight.detach().clone().to(torch.float32).requires_grad_(True)
    bias_ref = bias.detach().clone().to(torch.float32).requires_grad_(True)
    logits = hidden_ref @ weight_ref.T + bias_ref
    logp = torch.log_softmax(logits, dim=1)
    logZ = torch.logsumexp(logits, dim=1)
    safe_ids = selected_ids.clamp(min=0)
    selected_logp = torch.gather(logp, dim=1, index=safe_ids)
    selected_logp = torch.where(selected_mask, selected_logp, torch.zeros_like(selected_logp))
    adjoints = torch.where(
        selected_mask,
        selected_adjoints.to(torch.float32),
        torch.zeros_like(selected_adjoints, dtype=torch.float32),
    )
    loss = (adjoints * selected_logp).sum()
    loss.backward()
    return DenseReferenceResult(
        selected_logp=selected_logp.detach(),
        logZ=logZ.detach(),
        loss=loss.detach(),
        grad_hidden=hidden_ref.grad.detach(),
        grad_weight=weight_ref.grad.detach(),
        grad_bias=bias_ref.grad.detach(),
        full_logits_shape=(int(logits.shape[0]), int(logits.shape[1])),
    )


def dense_standard_blank_target_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    target_ids: torch.Tensor,
    blank_id: int,
    dtarget: torch.Tensor,
    dblank: torch.Tensor,
) -> DenseReferenceResult:
    selected_ids, selected_mask = build_standard_blank_target_set(
        target_ids,
        blank_id,
        num_classes=weight.shape[0],
    )
    selected_adjoints = torch.stack([dtarget, dblank], dim=1).to(torch.float32)
    selected_adjoints = torch.where(selected_mask, selected_adjoints, torch.zeros_like(selected_adjoints))
    return dense_selected_reference(
        hidden,
        weight,
        bias,
        selected_ids,
        selected_mask,
        selected_adjoints,
    )
