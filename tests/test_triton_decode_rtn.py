"""Smoke tests for the Triton fused dequant-attention module.

The legacy per-block online-softmax decode path (triton_decode_attention) was
removed — the fused kernels (fused_decode_attention_packed / _k2b) plus the
chunked PyTorch fallback now cover every config (Wave 2 debloat; see
docs/2026-06-24-decode-path-debloat-removal.md addendum). Bit-exact oracle
coverage for the fused kernels lives in the fused-kernel test modules.

These two tests run on ANY machine (including this AMD/no-CUDA dev box):
  - the module must import cleanly with TRITON_AVAILABLE=False.
  - _require_triton() must fail loud (RuntimeError) when TRITON_AVAILABLE=False.
"""

import pytest


def test_triton_module_imports_with_available_flag():
    """Module must import cleanly on AMD/no-CUDA with TRITON_AVAILABLE=False."""
    from bmx.cache.triton_dequant_attention import TRITON_AVAILABLE  # noqa: F401

    # On non-CUDA hosts TRITON_AVAILABLE is False — that is the correct state.
    # On a CUDA host with Triton installed it should be True.
    # Either way, the import must not raise.
    assert isinstance(TRITON_AVAILABLE, bool)


def test_require_triton_raises_without_cuda():
    """_require_triton() must raise RuntimeError when TRITON_AVAILABLE is False."""
    import bmx.cache.triton_dequant_attention as mod

    original = mod.TRITON_AVAILABLE
    try:
        mod.TRITON_AVAILABLE = False
        with pytest.raises(RuntimeError, match="TRITON_AVAILABLE"):
            mod._require_triton()
    finally:
        mod.TRITON_AVAILABLE = original
