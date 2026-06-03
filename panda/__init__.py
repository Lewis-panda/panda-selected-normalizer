"""PANDA selected-normalizer research prototype package."""

from .reference import dense_selected_reference, dense_standard_blank_target_reference
from .selected_normalizer import (
    SelectedBackwardResult,
    SelectedForwardBackwardResult,
    SelectedNormalizerResult,
    build_standard_blank_target_set,
    cuda_peak_mib,
    reset_cuda_peak,
    selected_backward,
    selected_forward_backward,
    selected_log_probs,
    standard_blank_target_forward_backward,
    tensor_metrics,
    validate_selected_set,
)

__all__ = [
    "SelectedBackwardResult",
    "SelectedForwardBackwardResult",
    "SelectedNormalizerResult",
    "build_standard_blank_target_set",
    "cuda_peak_mib",
    "dense_selected_reference",
    "dense_standard_blank_target_reference",
    "reset_cuda_peak",
    "selected_backward",
    "selected_forward_backward",
    "selected_log_probs",
    "standard_blank_target_forward_backward",
    "tensor_metrics",
    "validate_selected_set",
]
