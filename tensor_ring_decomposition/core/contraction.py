"""Contraction path computation and execution using opt_einsum."""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Tuple

import opt_einsum as oe
import torch

class ShapeView:
    """Lightweight wrapper so opt_einsum accepts shape tuples as tensors."""

    __slots__ = ("shape",)

    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape


class ContractionPathCache:
    """Thread-safe, deterministic cache for contraction paths.

    Stores paths keyed by (equation string, shape tuples).
    Path is computed once with dummy tensors and reused forever.
    """

    _cache: Dict[
        Tuple[str, Tuple[Tuple[int, ...], ...]],
        Tuple[List[Tuple[int, int]], str],
    ] = {}
    _lock = threading.Lock()

    @classmethod
    def get_path(
        cls,
        eq: str,
        tensors_or_shapes: List[torch.Tensor | Tuple[int, ...]],
        optimize: str = "greedy",
    ) -> Tuple[List[Tuple[int, int]], str]:
        """Get cached contraction path or compute and cache it.

        Accepts either actual tensors or shape tuples.  Shape tuples are
        wrapped so opt_einsum can read the ``shape`` attribute.

        Returns:
            (path, path_info_str) pair.
        """
        # Normalise every entry to a plain shape tuple
        shapes: List[Tuple[int, ...]] = []
        operands: list = []
        for entry in tensors_or_shapes:
            if isinstance(entry, torch.Tensor):
                shapes.append(tuple(entry.shape))
                operands.append(entry)
            else:
                shapes.append(tuple(entry))
                operands.append(ShapeView(entry))

        key = (eq, tuple(shapes))

        with cls._lock:
            if key not in cls._cache:
                path, path_info = oe.contract_path(eq, *operands, optimize=optimize)
                cls._cache[key] = (path, str(path_info))

        return cls._cache[key]

    @classmethod
    def clear(cls) -> None:
        """Clear all cached paths."""
        with cls._lock:
            cls._cache.clear()


def compute_vocab_chain_expression(
    vocab_core_shapes: List[Tuple[int, int, int]],
    rank: int,
) -> oe.contract_expression:
    """Precompute contraction path for vocab cores.
    Result: (B, R, R) chain result.
    """
    k = len(vocab_core_shapes)
    # Use distinct single characters for indices.
    # Batch: 'b', Ranks: 'a', 'c', 'e', 'g'... (odd alphabet)
    rank_chars = "acegikmoqsuwy"
    operands = [f"b{rank_chars[i]}{rank_chars[i+1]}" for i in range(k)]
    eq = ",".join(operands) + f"->b{rank_chars[0]}{rank_chars[k]}"

    shapes = [(1, s[1], s[2]) for s in vocab_core_shapes]
    return oe.contract_expression(eq, *shapes)


def compute_emb_precontraction_expression(
    emb_core_shapes: List[Tuple[int, int, int]],
) -> oe.contract_expression:
    """Precompute contraction path for embedding cores.
    Result: (R0, D, Rm) tensor.
    """
    m = len(emb_core_shapes)
    # Dim indices: 'f', 'h', 'j', 'l'... (even alphabet)
    # Rank indices: 'a', 'c', 'e', 'g'... (odd alphabet)
    dim_chars = "fhjlnprstvwxyz"
    rank_chars = "acegikmoqsuwy"
    
    operands = [f"{dim_chars[i]}{rank_chars[i]}{rank_chars[i+1]}" for i in range(m)]
    eq = ",".join(operands) + f"{rank_chars[0]}" + "".join(dim_chars[:m]) + f"{rank_chars[m]}"

    shapes = [s for s in emb_core_shapes]
    return oe.contract_expression(eq, *shapes)

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
    use_efficient: bool = True,
) -> torch.Tensor:
    """Combine vocab chain result with precontracted emb contraction.

    Args:
        vocab_result: (B, R, R)
        emb_contraction: (R, D, R)
        use_efficient: Whether to use memory-efficient loop

    Returns:
        (B, D) output embeddings
    """
    if use_efficient:
        return _ring_closure_efficient(vocab_result, emb_contraction)
    else:
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
