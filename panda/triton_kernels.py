"""Optional Triton extension boundary.

The GitHub-ready draft intentionally keeps the default backend as the readable
PyTorch tiled implementation in :mod:`panda.selected_normalizer`.

Separate Triton experiments exist outside this minimal package, but they are
not vendored here as a production/default backend. This wrapper gives future
contributors a stable place to add a Triton implementation after it has separate
packaging and validation.
"""

from __future__ import annotations

from .selected_normalizer import selected_forward_backward, selected_log_probs


try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


def triton_available() -> bool:
    return TRITON_AVAILABLE


def selected_log_probs_triton(*args, **kwargs):
    """Placeholder for a future packaged Triton forward path.

    For this draft package, use :func:`panda.selected_normalizer.selected_log_probs`.
    """

    raise NotImplementedError(
        "Packaged Triton kernels are not enabled in this GitHub-ready draft. "
        "Use panda.selected_normalizer.selected_log_probs for the readable "
        "tiled smoke path."
    )


def selected_forward_backward_torch_tiled(*args, **kwargs):
    """Explicit alias for the current default smoke backend."""

    return selected_forward_backward(*args, **kwargs)


__all__ = [
    "TRITON_AVAILABLE",
    "triton_available",
    "selected_log_probs",
    "selected_log_probs_triton",
    "selected_forward_backward_torch_tiled",
]
