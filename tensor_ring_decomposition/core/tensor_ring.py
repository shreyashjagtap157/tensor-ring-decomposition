"""Tensor Ring tensor representation and reconstruction."""

from __future__ import annotations

from typing import List

import torch


class TRTensor:
    """Tensor Ring tensor representation."""

    def __init__(
        self,
        vocab_cores: List[torch.Tensor],
        emb_cores: List[torch.Tensor],
    ):
        self.vocab_cores = vocab_cores
        self.emb_cores = emb_cores

    def to_tensor(self) -> torch.Tensor:
        """Reconstruct full V x D matrix from ring factors.

        Each vocab core  has shape (V_i, R_curr, R_next).
        Each emb core    has shape (D_i, R_curr, R_next).
        On every step we contract the shared rank (last dim of accumulator,
        middle dim of the new core) while keeping the *ring-opening* rank
        (dim 1 of the very first core) alive.

        WARNING: Materialises the full (V, D) matrix.  Debugging only.
        """
        # ----------------------- embedding chain -------------------------
        emb = self.emb_cores[0]                     # (D_0, R_0, R_1)
        for core in self.emb_cores[1:]:             # (D_i, R_i, R_{i+1})
            # contract over R_i  (last dim of emb / middle dim of core)
            # `dab,ebc->deac`  means: emb[d, a, b] * core[e, b, c] → (d,e,a,c)
            # reshape merges the first two (d*e) into the new vocab dim:
            emb = torch.einsum("dab,ebc->deac", emb, core)
            emb = emb.reshape(-1, emb.shape[2], emb.shape[3])
        # emb: (D_total, R_0, R_last)
        R0 = emb.shape[1]
        emb = emb.permute(1, 0, 2)                  # → (R_0, D_total, R_last)

        # ------------------------- vocab chain ---------------------------
        voc = self.vocab_cores[0]                   # (V_0, R_0, R_1)
        for core in self.vocab_cores[1:]:           # (V_i, R_i, R_{i+1})
            # contract over R_i  (last dim of voc / middle dim of core)
            voc = torch.einsum("vab,wbc->vwac", voc, core)
            voc = voc.reshape(-1, voc.shape[2], voc.shape[3])
        # voc: (V_total, R_0, R_last)

        # ------------------------ ring closure ---------------------------
        # result[v, d] = Σ_{r₀, r_last} voc[v, r₀, r_last] * emb[r_last, d, r₀]
        result = torch.einsum("vri,idr->vd", voc, emb)
        return result.reshape(result.shape[0], -1)

    def parameter_count(self) -> int:
        """Total parameters."""
        return sum(c.numel() for c in self.vocab_cores + self.emb_cores)
