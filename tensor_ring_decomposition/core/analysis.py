"""Analysis utilities for tensor ring decomposition.

Extracts common analysis patterns for reuse across modules.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def compute_svd(
    matrix: torch.Tensor,
    q: int = 200,
    full_matrices: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute SVD of a matrix using optimal method based on size.

    Uses randomized SVD (svd_lowrank) for large matrices to avoid OOM,
    and standard SVD for smaller matrices.

    Args:
        matrix: Input matrix.
        q: Number of random vectors for randomized SVD (only used if matrix
           has min(dim) > 200).
        full_matrices: Whether to compute full U, Vh matrices.

    Returns:
        Tuple of (U, S, Vh) singular values and vectors.
    """
    matrix_f = matrix.to(torch.float32)
    Vm, Dm = matrix_f.shape

    if min(Vm, Dm) > 200:
        U, S, Vh = torch.svd_lowrank(matrix_f, q=min(min(Vm, Dm), q))
    else:
        U, S, Vh = torch.linalg.svd(matrix_f, full_matrices=full_matrices)

    return U, S, Vh


def compute_svdvals(
    matrix: torch.Tensor,
    q: int = 200,
) -> torch.Tensor:
    """Compute only the singular values of a matrix.

    Args:
        matrix: Input matrix.
        q: Number of random vectors for randomized SVD.

    Returns:
        Tensor of singular values.
    """
    matrix_f = matrix.to(torch.float32)
    Vm, Dm = matrix_f.shape

    if min(Vm, Dm) > 200:
        return torch.svd_lowrank(matrix_f, q=min(min(Vm, Dm), q))[1]
    return torch.linalg.svdvals(matrix_f)


def compute_variance_explained(
    S: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cumulative variance explained by singular values.

    Args:
        S: Singular values from SVD.

    Returns:
        Tuple of (cumsum_variance_normalized, total_variance) where
        cumsum_variance_normalized has shape (len(S),) with values in [0, 1].
    """
    S_sq = S ** 2
    total_var = S_sq.sum()
    cum_var = torch.cumsum(S_sq, dim=0) / total_var
    return cum_var, total_var


def find_knee_point(
    values: torch.Tensor,
    slope_threshold: float = 0.1,
) -> int:
    """Find knee point in a monotonically increasing curve.

    Uses the distance from the line connecting first and last points.

    Args:
        values: Monotonically increasing values.
        slope_threshold: Minimum slope to consider a knee.

    Returns:
        Index of knee point.
    """
    n = len(values)
    if n < 2:
        return 0

    first = values[0]
    last = values[-1]

    line_vals = torch.linspace(first, last, n)

    distances = torch.abs(values - line_vals)
    knee_idx = torch.argmax(distances).item()

    slopes = (values[1:] - values[:-1])
    if knee_idx < n - 1 and slopes[knee_idx] < slope_threshold * slopes[:knee_idx].mean():
        knee_idx = 0

    return knee_idx


def spectral_gap_analysis(
    S: torch.Tensor,
    significance_factor: float = 3.0,
) -> Tuple[int, int]:
    """Analyze singular values for spectral gap detection.

    Args:
        S: Singular values.
        significance_factor: Number of standard deviations for significance.

    Returns:
        Tuple of (rank_statistical, rank_gap) - two rank estimates.
    """
    S_avg = S.mean()
    S_std = S.std()

    significance_threshold = significance_factor * S_std

    significant_indices = torch.where(S > significance_threshold)[0]

    if len(significant_indices) == 0:
        rank_stat = len(S) // 4
    else:
        rank_stat = significant_indices[-1].item() + 1

    rel_drops = torch.abs((S[:-1] - S[1:]) / (S[:-1] + 1e-8))
    max_gap_idx = torch.argmax(rel_drops)
    rank_gap = max_gap_idx.item() + 1

    return rank_stat, rank_gap


def compute_covariance(
    matrix: torch.Tensor,
    chunk_size: int = 1024,
    mean: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute covariance matrix in chunks to avoid OOM for large matrices.

    Computes X^T @ X / N efficiently using chunked matrix multiplication.
    For a matrix of shape (V, D), this computes the (D, D) covariance matrix.

    Args:
        matrix: Input matrix of shape (V, D).
        chunk_size: Number of rows to process at a time.
        mean: Optional precomputed mean of shape (D,). If None, computed in chunks.

    Returns:
        Covariance matrix of shape (D, D).

    Example:
        >>> matrix = torch.randn(10000, 768)
        >>> cov = compute_covariance(matrix, chunk_size=512)
        >>> assert cov.shape == (768, 768)
    """
    V, D = matrix.shape

    if mean is None:
        mean = torch.zeros(D, dtype=matrix.dtype, device=matrix.device)
        count = 0
        for start in range(0, V, chunk_size):
            end = min(start + chunk_size, V)
            chunk = matrix[start:end]
            mean += chunk.sum(dim=0)
            count += chunk.shape[0]
        mean = mean / count

    cov = torch.zeros(D, D, dtype=matrix.dtype, device=matrix.device)

    for start in range(0, V, chunk_size):
        end = min(start + chunk_size, V)
        chunk = matrix[start:end] - mean
        cov += chunk.T @ chunk

    cov = cov / V
    return cov


def compute_covariance_chunked(
    matrix: torch.Tensor,
    n_chunks: int = 4,
) -> torch.Tensor:
    """Compute covariance matrix by splitting into n chunks.

    Alternative to compute_covariance that splits rows evenly.

    Args:
        matrix: Input matrix of shape (V, D).
        n_chunks: Number of chunks to split into.

    Returns:
        Covariance matrix of shape (D, D).
    """
    V, D = matrix.shape
    chunk_size = (V + n_chunks - 1) // n_chunks
    return compute_covariance(matrix, chunk_size=chunk_size)