# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jetson compatibility helpers for Alpamayo inference."""

from __future__ import annotations

from typing import Any

import torch


def patch_typing_self() -> None:
    """Provide typing.Self on Python 3.10 for dependencies that import it."""
    import typing

    if hasattr(typing, "Self"):
        return

    from typing_extensions import Self

    typing.Self = Self


def disable_transformers_cuda_allocator_warmup() -> None:
    """Disable Transformers' CUDA allocator warmup to avoid large Jetson preallocations."""
    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError:
        return

    modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None


def _is_jetson_linalg_symbol_error(exc: BaseException) -> bool:
    text = str(exc)
    return "libtorch_cuda_linalg.so" in text or "cusolverDnXsyevBatched_bufferSize" in text


def patch_linalg_cpu_fallback(verbose: bool = True) -> None:
    """Fallback selected CUDA linalg calls to CPU for Jetson libcusolver symbol gaps."""
    if getattr(torch.linalg.cholesky, "_alpamayo_jetson_patched", False):
        return

    orig_cholesky = torch.linalg.cholesky
    orig_cholesky_solve = torch.cholesky_solve

    def safe_cholesky(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        try:
            return orig_cholesky(input, *args, **kwargs)
        except RuntimeError as exc:
            if input.is_cuda and _is_jetson_linalg_symbol_error(exc):
                if verbose:
                    print("fallback cholesky on cpu")
                return orig_cholesky(input.cpu(), *args, **kwargs).to(input.device)
            raise

    def safe_cholesky_solve(
        input: torch.Tensor,
        input2: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        try:
            return orig_cholesky_solve(input, input2, *args, **kwargs)
        except RuntimeError as exc:
            if input.is_cuda and _is_jetson_linalg_symbol_error(exc):
                if verbose:
                    print("fallback cholesky_solve on cpu")
                return orig_cholesky_solve(input.cpu(), input2.cpu(), *args, **kwargs).to(
                    input.device
                )
            raise

    safe_cholesky._alpamayo_jetson_patched = True
    torch.linalg.cholesky = safe_cholesky
    torch.cholesky_solve = safe_cholesky_solve


def apply_jetson_inference_compat(verbose: bool = True) -> None:
    """Apply the compatibility tweaks used by the Jetson Orin inference path."""
    patch_typing_self()
    disable_transformers_cuda_allocator_warmup()
    patch_linalg_cpu_fallback(verbose=verbose)
