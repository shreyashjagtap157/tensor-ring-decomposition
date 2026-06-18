"""Contraction path computation and execution for tensor ring decomposition."""

from __future__ import annotations

from typing import List, Optional

import torch

from .factorization import compute_mixed_radix_strides


def gather_vocab_cores(
    flat_indices: torch.Tensor,
    vocab_cores: List[torch.Tensor],
    factor_sizes: List[int],
    strides: Optional[List[int]] = None,
    *,
    raise_oob: bool = False,
) -> torch.Tensor:
    """Decompose flat indices via mixed-radix, gather core slices, chain via bmm.

    Args:
        flat_indices: (B,) flat token indices.
        vocab_cores: List of (V_i, R_i, R_{i+1}) core tensors.
        factor_sizes: List of vocab factor sizes.
        strides: Precomputed strides. If None, computed from factor_sizes.
        raise_oob: When True, raise IndexError if any factor index is out of
            range. When False (default), apply a (no-op for in-range ids)
            clamp to keep behavior backward-compatible with v0.4.0.

    Returns:
        (B, R_0, R_k) chained result.
    """
    k = len(vocab_cores)
    if strides is None:
        strides = compute_mixed_radix_strides(factor_sizes)
    factor_indices: List[torch.Tensor] = []
    remaining = flat_indices
    for i in range(k):
        if i < k - 1:
            fi = remaining // strides[i]
            remaining = remaining % strides[i]
        else:
            fi = remaining
        if raise_oob:
            if flat_indices.numel() and (
                fi.min().item() < 0 or fi.max().item() >= factor_sizes[i]
            ):
                raise IndexError(
                    f"Vocabulary factor index out of range: factor {i} has "
                    f"size {factor_sizes[i]} but factor indices in "
                    f"[{fi.min().item()}, {fi.max().item()}] encountered."
                )
        else:
            # Hot path. Clamp is a no-op on valid token ids, and historically
            # silently coerced malformed ids to the last factor row. API users
            # can opt into ``raise_oob=True`` to surface this loudly.
            fi = fi.clamp(0, factor_sizes[i] - 1)
        factor_indices.append(fi)

    gathered = [vocab_cores[i][factor_indices[i]] for i in range(k)]
    result = gathered[0]
    for cg in gathered[1:]:
        result = torch.bmm(result, cg)
    return result



def compute_emb_precontraction(emb_cores: List[torch.Tensor]) -> torch.Tensor:
    """Compute embedding cores precontraction via sequential bmm.

    Each emb core i has shape (D_i, R, R).
    Result: (R, D_0*D_1*...*D_{m-1}, R) reshaped to (R, D, R).
    """
    if len(emb_cores) == 0:
        raise ValueError("emb_cores must not be empty")
    if len(emb_cores) == 1:
        core = emb_cores[0]
        # (D, R, R) -> (R, D, R)
        return core.permute(1, 0, 2)

    # Sequential bmm: contract last dim of result with middle dim of core
    result = emb_cores[0]
    for core in emb_cores[1:]:
        # result: (D_prev, R, R), core: (D_next, R, R)
        # Contract over R (last dim of result, middle dim of core)
        # einsum: result(a,b,c) core(d,c,e) -> (a,b,d,e) -> permute to (a,d,b,e) -> reshape to (a*d,b,e)
        result = torch.einsum("abc,dce->adbe", result, core)
        result = result.reshape(-1, result.shape[2], result.shape[3])

    # result: (D_total, R, R) -> (R, D_total, R)
    return result.permute(1, 0, 2)


def ring_closure(
    vocab_result: torch.Tensor,
    emb_contraction: torch.Tensor,
    use_efficient: bool = False,
) -> torch.Tensor:
    """Combine vocab chain result with precontracted emb contraction.

    Args:
        vocab_result: (B, R, R)
        emb_contraction: (R, D, R)
        use_efficient: When True, use the Python-loop implementation. This is
            primarily kept for benchmarking and backward compatibility;
            the einsum path is faster for the typical rank range (R <= 256)
            because it dispatches a single fused BLAS contraction instead
            of R Python-level torch.mm calls.

    Returns:
        (B, D) output embeddings
    """
    if use_efficient:
        return _ring_closure_efficient(vocab_result, emb_contraction)
    return _ring_closure_einsum(vocab_result, emb_contraction)


def _ring_closure_efficient(
    vocab_result: torch.Tensor,
    emb_contraction: torch.Tensor,
) -> torch.Tensor:
    """Memory-efficient ring closure via R-dimension loop."""
    R = vocab_result.shape[1]
    B = vocab_result.shape[0]
    D = emb_contraction.shape[1]
    device = vocab_result.device
    dtype = vocab_result.dtype

    output = torch.zeros(B, D, device=device, dtype=dtype)
    for r in range(R):
        output += torch.mm(vocab_result[:, r, :], emb_contraction[:, :, r])

    return output


def _ring_closure_einsum(
    vocab_result: torch.Tensor,
    emb_contraction: torch.Tensor,
) -> torch.Tensor:
    """Einsum-based ring closure.

    Ring closure formula:
        result[b, d] = sum_{r₀, r₁} voc[b, r₀, r₁] * emb[r₁, d, r₀]

    Args:
        vocab_result: (B, R, R) — vocabulary chain result
        emb_contraction: (R, D, R) — embedding precontraction

    Returns:
        (B, D) compressed embeddings
    """
    return torch.einsum('bri,idr->bd', vocab_result, emb_contraction)
