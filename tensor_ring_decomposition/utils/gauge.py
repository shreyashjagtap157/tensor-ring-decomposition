"""Gauge fixing utilities for Tensor Ring cores."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class GaugeFixer:
    """Orthogonalize TR cores to eliminate gauge freedom."""

    @staticmethod
    def fix_left(cores: nn.ParameterList) -> None:
        """QR-based left gauge fix: orthogonalize from left to right."""
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

    @staticmethod
    def fix_right(cores: nn.ParameterList) -> None:
        """RQ-based right gauge fix: orthogonalize from right to left."""
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

    @staticmethod
    def spectral_norms(cores: nn.ParameterList) -> List[float]:
        """Compute spectral norm (largest singular value) of each core."""
        norms: List[float] = []
        for core in cores:
            flat = core.data.reshape(-1, core.shape[-1])
            s = torch.linalg.svdvals(flat)
            norms.append(s[0].item())
        return norms
