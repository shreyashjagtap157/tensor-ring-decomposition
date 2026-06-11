"""Dimension factorization and ring structure computation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


def factorize_dimension(dim: int, n_factors: int) -> List[int]:
    """Split dim into n_factors parts whose product = dim.

    Strategy: Greedy near-equal product.

    Examples:
        factorize_dimension(50000, 4) -> [9, 9, 9, 68]
        factorize_dimension(768, 4) -> [3, 4, 8, 8]
        factorize_dimension(100, 2) -> [10, 10]

    Raises:
        ValueError: If dim < n_factors (cannot factorize).
    """
    if dim < n_factors:
        raise ValueError(
            f"Cannot factor {dim} into {n_factors} factors (each must be >= 1)"
        )
    if n_factors == 1:
        return [dim]

    factors: List[int] = []
    remaining = dim

    for i in range(n_factors - 1):
        factors_left = n_factors - i
        target = remaining ** (1.0 / factors_left)
        factor = max(1, int(round(target)))
        # Ensure factor divides remaining evenly
        while remaining % factor != 0 and factor > 1:
            factor -= 1
        if factor < 1:
            factor = 1
        factors.append(factor)
        remaining //= factor

    factors.append(remaining)
    return factors


def find_best_factorization(n: int, k: int) -> List[int]:
    """Find the most balanced factorization of n into k factors."""
    if k == 0:
        return []
    if k == 1:
        return [n]

    # Find all divisors of n
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i * i != n:
                divisors.append(n // i)
    divisors.sort()

    target = n ** (1.0 / k)
    best_factors = None
    best_max = float("inf")

    # Greedy / recursive search for balanced factors
    for d in divisors:
        # Pruning: if d itself is already worse than the best max, skip
        if d >= best_max:
            continue
        sub = find_best_factorization(n // d, k - 1)
        if sub:
            factors = [d] + sub
            max_f = max(factors)
            if max_f < best_max:
                best_max = max_f
                best_factors = factors

    if best_factors is None:
        # Fallback to standard factorization
        return factorize_dimension(n, k)
    return sorted(best_factors)


def find_highly_factorable_dim(
    dim: int, n_factors: int, max_padding_pct: float = 0.10
) -> Tuple[int, List[int]]:
    """Find the smallest padded dimension >= dim that has highly balanced factors.

    Solves the prime/near-prime vocabulary bottleneck by searching for a slightly
    padded dimension with much smaller maximum factors.

    Args:
        dim: The original dimension.
        n_factors: Number of factors required.
        max_padding_pct: Maximum allowed padding percentage.

    Returns:
        A tuple of (padded_dim, factors_list).
    """
    if dim < n_factors:
        raise ValueError(f"dim {dim} must be >= n_factors {n_factors}")
    if n_factors == 0:
        return dim, []
    if n_factors == 1:
        return dim, [dim]

    max_padded = int(math.ceil(dim * (1.0 + max_padding_pct)))
    best_dim = dim
    best_factors = factorize_dimension(dim, n_factors)
    best_max_factor = max(best_factors)

    # If the current dimension is already reasonably factorable, return it
    target = dim ** (1.0 / n_factors)
    if best_max_factor <= 2.0 * target:
        return dim, best_factors

    # Search candidates from dim up to max_padded
    for candidate in range(dim, max_padded + 1):
        factors = find_best_factorization(candidate, n_factors)
        max_f = max(factors)
        # Score candidates preferring smaller max factor, and smaller candidate (less padding)
        # We give a strong weight to reducing the max factor
        if max_f < best_max_factor:
            best_max_factor = max_f
            best_dim = candidate
            best_factors = factors

            # If we've hit a nearly perfect balance, we can stop early
            if best_max_factor <= 1.2 * target:
                break

    return best_dim, best_factors


@dataclass
class RingStructure:
    """Complete specification of a tensor ring decomposition."""

    vocab_factor_sizes: List[int]
    emb_factor_sizes: List[int]
    ranks: List[int]
    rank: int
    ring_components: int
    n_vocab_cores: int
    n_emb_cores: int
    original_vocab_size: Optional[int] = None
    original_embedding_dim: Optional[int] = None
    padded_vocab_size: Optional[int] = None
    padded_embedding_dim: Optional[int] = None


def compute_ring_structure(
    vocab_size: int,
    embedding_dim: int,
    ring_components: int = 4,
    rank: int = 8,
    split_mode: str = "balanced",
    ranks: Optional[List[int]] = None,
    auto_pad: bool = True,
    max_padding_pct: float = 0.15,
) -> RingStructure:
    """Compute the complete ring structure.

    Args:
        vocab_size: Original vocabulary size.
        embedding_dim: Original embedding dimension.
        ring_components: Total number of ring cores.
        rank: Target compression rank.
        split_mode: How to split components between vocab and embedding ("balanced" or "proportional").
        ranks: Optional explicit ranks per boundary.
        auto_pad: If True, pads vocab_size and embedding_dim to highly factorable numbers
                  to avoid prime/near-prime bottlenecks.
        max_padding_pct: Maximum allowed padding percentage.
    """
    if ring_components < 2:
        raise ValueError(f"ring_components must be >= 2, got {ring_components}")
    if split_mode not in ("balanced", "proportional", "manual"):
        raise ValueError(f"Unknown split_mode: {split_mode}")

    if split_mode == "balanced":
        k = ring_components // 2
        m = ring_components - k
    elif split_mode == "proportional":
        total = vocab_size + embedding_dim
        k = max(1, int(round(ring_components * vocab_size / total)))
        m = ring_components - k
    elif split_mode == "manual":
        if ranks is None:
            raise ValueError("split_mode='manual' requires explicit ranks")
        k = ring_components // 2
        m = ring_components - k

    padded_v = vocab_size
    padded_d = embedding_dim

    if auto_pad:
        padded_v, vocab_factors = find_highly_factorable_dim(vocab_size, k, max_padding_pct)
        padded_d, emb_factors = find_highly_factorable_dim(embedding_dim, m, max_padding_pct)
    else:
        vocab_factors = factorize_dimension(vocab_size, k)
        emb_factors = factorize_dimension(embedding_dim, m)

    if ranks is not None and len(ranks) < ring_components + 1:
        raise ValueError(
            f"ranks length ({len(ranks)}) must be at least "
            f"ring_components+1 ({ring_components + 1})"
        )
    if ranks is None:
        ranks = [rank] * (k + m + 1)

    return RingStructure(
        vocab_factor_sizes=vocab_factors,
        emb_factor_sizes=emb_factors,
        ranks=ranks,
        rank=rank,
        ring_components=ring_components,
        n_vocab_cores=k,
        n_emb_cores=m,
        original_vocab_size=vocab_size,
        original_embedding_dim=embedding_dim,
        padded_vocab_size=padded_v,
        padded_embedding_dim=padded_d,
    )
