"""Dimension factorization and ring structure computation."""

from __future__ import annotations

import functools
import math
import logging
logger = logging.getLogger(__name__)
from dataclasses import dataclass
from typing import List, Optional, Tuple


def factorize_dimension(dim: int, n_factors: int) -> List[int]:
    """Split dim into n_factors parts whose product = dim.

    Strategy: Greedy near-equal product.

    Examples:
        factorize_dimension(50000, 4) -> [10, 10, 20, 25]
        factorize_dimension(768, 4) -> [4, 6, 4, 8]
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


@functools.lru_cache(maxsize=1024)
def _find_best_factors(n: int, k: int) -> Tuple[int, ...]:
    """Memoized recursive search for most balanced factorization.
    
    Returns tuple of factors sorted in their computed order (preserving
    positional semantics for TR cores).
    """
    if k == 0:
        return ()
    if k == 1:
        return (n,)

    # Find divisors up to sqrt(n)
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i * i != n:
                divisors.append(n // i)
    divisors.sort()

    best_factors: Optional[Tuple[int, ...]] = None
    best_max = float("inf")

    for d in divisors:
        if d >= best_max:
            continue
        sub = _find_best_factors(n // d, k - 1)
        if sub:
            factors = (d,) + sub
            max_f = max(factors)
            if max_f < best_max:
                best_max = max_f
                best_factors = factors

    if best_factors is None:
        return tuple(factorize_dimension(n, k))
    return best_factors


def find_best_factorization(n: int, k: int) -> List[int]:
    """Find the most balanced factorization of n into k factors.
    
    Uses memoized recursive search over divisors. Results are cached
    across calls for the same (n, k) pair.
    """
    return list(_find_best_factors(n, k))


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

    # Search candidates using divisor-based stepping.
    # The key insight: for each candidate, the best factorization's max factor
    # divides the candidate. We only need to check candidates that have small
    # divisors, skipping prime numbers and nearly-prime numbers.
    for candidate in range(dim, max_padded + 1):
        # Quick pre-filter: skip primes and near-primes
        sqrt_c = int(math.isqrt(candidate))
        has_small_divisor = any(candidate % d == 0 for d in range(2, min(sqrt_c, 50) + 1))
        if not has_small_divisor and candidate > dim:
            continue

        factors = find_best_factorization(candidate, n_factors)
        max_f = max(factors)
        if max_f < best_max_factor:
            best_max_factor = max_f
            best_dim = candidate
            best_factors = factors

            if best_max_factor <= 1.2 * target:
                break

    return best_dim, best_factors


def compute_mixed_radix_strides(factors: List[int]) -> List[int]:
    """Compute strides for mixed-radix decomposition.

    strides[i] = product of factors[i+1:]
    Uses single reverse pass (O(n) instead of O(n²)).
    """
    n = len(factors)
    strides = [1] * n
    for i in range(n - 2, -1, -1):
        strides[i] = strides[i + 1] * factors[i + 1]
    return strides


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


def compute_tapered_ranks(
    ring_components: int,
    base_rank: int,
    taper_factor: float = 0.5,
    min_rank: int = 2,
) -> List[int]:
    """Compute tapered ranks that decrease toward the middle of the ring.
    
    Tapering reduces computational cost while maintaining expressiveness at
    boundaries. The first and last ranks (boundaries) stay at base_rank,
    while middle ranks taper down by taper_factor.
    
    Args:
        ring_components: Total number of ring components.
        base_rank: Rank at boundaries (first and last).
        taper_factor: Factor to multiply by distance from boundary (0 < taper_factor < 1).
        min_rank: Minimum allowed rank.
        
    Returns:
        List of ranks of length ring_components + 1.
        
    Example:
        compute_tapered_ranks(4, 8, 0.5) -> [8, 4, 2, 4, 8]
        compute_tapered_ranks(5, 8, 0.5) -> [8, 4, 2, 2, 4, 8]
    """
    total_boundaries = ring_components + 1

    ranks = [base_rank] * total_boundaries
    middle_ranks = total_boundaries - 2
    if middle_ranks > 0:
        for i in range(1, total_boundaries - 1):
            dist_from_boundary = min(i, total_boundaries - 1 - i)
            taper = taper_factor ** dist_from_boundary
            new_rank = max(min_rank, int(base_rank * taper))
            ranks[i] = new_rank

    return ranks


def _compute_param_balanced_split(
    vocab_size: int, embedding_dim: int, ring_components: int
) -> Tuple[int, int]:
    """Compute split that balances parameter count between vocab and emb sides.
    
    For param-balanced split:
        vocab_size * n_vocab_cores ~ embedding_dim * n_emb_cores
    
    Args:
        vocab_size: Vocabulary size.
        embedding_dim: Embedding dimension.
        ring_components: Total number of ring components.
        
    Returns:
        Tuple of (n_vocab_cores, n_emb_cores).
    """
    best_k = ring_components // 2
    best_m = ring_components - best_k
    best_diff = float("inf")

    for k in range(1, ring_components):
        m = ring_components - k
        vocab_params = vocab_size * k
        emb_params = embedding_dim * m
        diff = abs(vocab_params - emb_params)
        if diff < best_diff:
            best_diff = diff
            best_k = k
            best_m = m

    return best_k, best_m


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
        split_mode: How to split components between vocab and embedding 
                   ("balanced", "proportional", "param_balanced", or "manual").
        ranks: Optional explicit ranks per boundary.
        auto_pad: If True, pads vocab_size and embedding_dim to highly factorable numbers
                  to avoid prime/near-prime bottlenecks.
        max_padding_pct: Maximum allowed padding percentage.
    """
    if ring_components < 2:
        raise ValueError(f"ring_components must be >= 2, got {ring_components}")
    if split_mode not in ("balanced", "proportional", "param_balanced", "manual"):
        raise ValueError(f"Unknown split_mode: {split_mode}")

    if split_mode == "balanced":
        k = ring_components // 2
        m = ring_components - k
    elif split_mode == "proportional":
        total = vocab_size + embedding_dim
        k = max(1, int(round(ring_components * vocab_size / total)))
        m = ring_components - k
    elif split_mode == "param_balanced":
        k, m = _compute_param_balanced_split(vocab_size, embedding_dim, ring_components)
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

    if ranks is not None:
        if len(ranks) < ring_components + 1:
            raise ValueError(
                f"ranks length ({len(ranks)}) must be at least "
                f"ring_components+1 ({ring_components + 1})"
            )
        if ranks[0] != ranks[-1]:
            raise ValueError(
                f"ranks[0] ({ranks[0]}) must equal ranks[-1] ({ranks[-1]}) "
                f"for tensor ring closure"
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
