"""Gauge fixing utilities for Tensor Ring cores."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class GaugeFixer:
    """Orthogonalize TR cores to eliminate gauge freedom."""

    @staticmethod
    def fix_left(cores: nn.ParameterList) -> None:
        """QR-based left gauge fix: orthogonalize from left to right.
        
        The R matrix from the last core is absorbed into the first core
        (wrap-around) to preserve the exact tensor ring reconstruction.
        """
        for i in range(len(cores)):
            old_shape = cores[i].shape
            flat = cores[i].data.reshape(-1, old_shape[2])

            Q, R_mat = torch.linalg.qr(flat)

            diag_signs = torch.sign(torch.diag(R_mat))
            diag_signs = torch.where(
                diag_signs == 0, torch.ones_like(diag_signs), diag_signs
            )
            Q = Q * diag_signs.unsqueeze(0)
            R_mat = torch.diag(diag_signs) @ R_mat

            cores[i].data = Q.reshape(old_shape)

            if i < len(cores) - 1:
                next_core = cores[i + 1]
                R_expanded = R_mat.unsqueeze(0).expand(next_core.shape[0], -1, -1)
                cores[i + 1].data = torch.bmm(R_expanded, next_core.data)
            elif len(cores) > 1:
                # Wrap-around: absorb final R into first core
                first_core = cores[0]
                R_expanded = R_mat.unsqueeze(0).expand(first_core.shape[0], -1, -1)
                cores[0].data = torch.bmm(first_core.data, R_expanded)

    @staticmethod
    def fix_right(cores: nn.ParameterList) -> None:
        """RQ-based right gauge fix: orthogonalize from right to left.
        
        The R matrix from the first core is absorbed into the last core
        (wrap-around) to preserve the exact tensor ring reconstruction.
        """
        for i in range(len(cores) - 1, -1, -1):
            old_shape = cores[i].shape
            R_dim = old_shape[1]

            flat_right = cores[i].data.permute(1, 0, 2).reshape(R_dim, -1)
            flat_right_t = flat_right.T

            Q_r, R_r = torch.linalg.qr(flat_right_t)
            diag_signs = torch.sign(torch.diag(R_r))
            diag_signs = torch.where(
                diag_signs == 0, torch.ones_like(diag_signs), diag_signs
            )
            Q_r = Q_r * diag_signs.unsqueeze(0)
            R_r = torch.diag(diag_signs) @ R_r

            Q_reshaped = Q_r.reshape(old_shape[0], old_shape[2], old_shape[1])
            cores[i].data = Q_reshaped.permute(0, 2, 1)

            if i > 0:
                prev_core = cores[i - 1]
                R_expanded = R_r.unsqueeze(0).expand(prev_core.shape[0], -1, -1)
                cores[i - 1].data = torch.bmm(prev_core.data, R_expanded)
            elif len(cores) > 1:
                # Wrap-around: absorb first R into last core
                last_core = cores[-1]
                R_expanded = R_r.unsqueeze(0).expand(last_core.shape[0], -1, -1)
                cores[-1].data = torch.bmm(last_core.data, R_expanded)

    @staticmethod
    def spectral_norms(cores: nn.ParameterList) -> List[float]:
        """Compute spectral norm (largest singular value) of each core.

        Uses power iteration (∼ O(m·n)) instead of full SVD (O(m·n²)).
        Typically 10-50× faster for the largest singular value only.
        """
        norms: List[float] = []
        for core in cores:
            flat = core.data.reshape(-1, core.shape[-1])
            s = _power_iteration_svd(flat, n_iter=15)
            norms.append(s.item())
        return norms


def _power_iteration_svd(A: torch.Tensor, n_iter: int = 15) -> torch.Tensor:
    """Estimate largest singular value via power iteration.

    Computes σ_max(A) ≈ ||A·v|| where v converges to the right singular
    vector through repeated multiplication Aᵀ·A·v.

    Args:
        A: (m, n) matrix.
        n_iter: Number of power iterations (default 15, sufficient for
                1e-5 relative error).

    Returns:
        Scalar estimate of σ_max(A).
    """
    if A.numel() == 0:
        return torch.tensor(0.0, device=A.device, dtype=A.dtype)

    v = torch.randn(A.shape[1], 1, device=A.device, dtype=A.dtype)
    v = v / v.norm()

    for _ in range(n_iter):
        u = A @ v
        u_norm = u.norm()
        if u_norm > 1e-12:
            u = u / u_norm
        v = A.T @ u

    # Normalize v before Rayleigh quotient to get σ (not σ²)
    v = v / v.norm()
    sigma = (u * (A @ v)).sum().abs()
    return sigma
