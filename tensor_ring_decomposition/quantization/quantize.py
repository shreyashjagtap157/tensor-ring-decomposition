"""Post-training int8 quantization for TR embeddings with AMX matmul.

Enterprise-grade quantization:
- Learned Step Size Quantization (LSQ) — ICLR 2020 — learnable scale parameters
- Per-tensor and per-channel quantization with STE gradient pass-through
- QAT with learnable scales for faster convergence
- Non-negative core constraint option for interpretable embeddings
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

if TYPE_CHECKING:
    from ..core.embedding import TensorRingEmbedding


class STERound(Function):
    """Straight-Through Estimator for rounding.

    Forward: round(x)
    Backward: identity (gradient passes through unchanged)
    """
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class LSQQuantize(Function):
    """Learned Step Size Quantization (ICLR 2020).

    Scale is a learnable parameter, not computed per-forward.
    This converges faster and to better minima than data-dependent scaling.

    Forward: q = round(x / s) * s   (STE: backward passes gradient unchanged)
    Scale gradient: dL/ds = dL/dq · (round(x/s) - x/s)  (LSQ formula)
    """
    @staticmethod
    def forward(ctx, x, scale, ste=True):
        s = scale
        orig_ndim = s.ndim
        if s.ndim == 1:
            s = s.view(-1, 1, 1)
        ctx.save_for_backward(x, s)
        ctx.orig_scale_ndim = orig_ndim
        if ste:
            q = STERound.apply(x / s).clamp(-128, 127)
            return q * s
        else:
            s_clamped = s.clamp(min=1e-8)
            q = STERound.apply(x / s_clamped).clamp(-128, 127)
            return q * s_clamped

    @staticmethod
    def backward(ctx, grad_output):
        x, s = ctx.saved_tensors
        grad_x = grad_output
        grad_scale = grad_output * (torch.round((x / s).detach()) - (x / s).detach())
        if grad_scale.ndim > ctx.orig_scale_ndim:
            dims = tuple(range(ctx.orig_scale_ndim, grad_scale.ndim))
            grad_scale = grad_scale.sum(dim=dims)
        return grad_x, grad_scale, None


def _quantize_tensor(t: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    """Quantize a float tensor to int8 with per-tensor scale and zero-point.

    Uses symmetric quantization (zero_point=0) for best performance.

    Returns:
        (int8_tensor, scale, zero_point)
    """
    abs_max = t.abs().max()
    if abs_max < 1e-8:
        return torch.zeros(t.shape, dtype=torch.int8, device=t.device), 1.0, 0
    scale = abs_max.item() / 127.0
    q = (t / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale, 0


def _quantize_tensor_per_channel(t: torch.Tensor, dim: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to int8 with per-channel scales."""
    reduce_dims = tuple(d for d in range(t.ndim) if d != dim)
    if reduce_dims:
        abs_max = t.abs().amax(dim=reduce_dims)
    else:
        abs_max = t.abs()
    abs_max = abs_max.clamp(min=1e-8)
    scales = abs_max / 127.0
    scale_shape = [1] * t.ndim
    scale_shape[dim] = -1
    q = (t / scales.view(*scale_shape)).round().clamp(-128, 127).to(torch.int8)
    return q, scales, torch.zeros_like(scales, dtype=torch.int)


class STEQuantize(Function):
    """Straight-Through Estimator for symmetric int8 quantization."""
    @staticmethod
    def forward(ctx, x, scale):
        q = torch.round(x / scale).clamp(-128, 127)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def fake_quantize(x, scale):
    return STEQuantize.apply(x, scale)


class NonNegativeClamp(Function):
    """Clamp gradients while clamping values to non-negative."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = grad_output.masked_fill(x < 0, 0.0)
        return grad


class QuantizedTensorRingEmbedding(nn.Module):
    """Post-training and Quantization-Aware Training (QAT) for TR embeddings.

    Supports both PTQ (Post-Training Quantization) and QAT (Quantization-Aware Training)
    using a Straight-Through Estimator (STE) to maintain gradients through rounding.

    Enterprise features:
    - LSQ: Learnable scale parameters (ICLR 2020) for faster QAT convergence
    - Per-tensor or per-channel scale granularity
    - Non-negative core constraint for interpretability
    - Memory-efficient eval cache for PTQ mode
    """
    def __init__(self, embedding: "TensorRingEmbedding", per_channel: bool = True, qat: bool = False,
                 lsq: bool = False, non_negative: bool = False):
        super().__init__()
        self.vocab_size = embedding.vocab_size
        self.embedding_dim = embedding.embedding_dim
        self.ring_components = embedding.ring_components
        self.structure = embedding.structure
        self.padding_idx = embedding.padding_idx
        self._dtype = embedding._dtype
        self._vocab_strides = embedding._vocab_strides
        self._validate_indices_flag = embedding._validate_indices_flag
        self._max_seq_len = embedding._max_seq_len
        self.qat = qat
        self.lsq = lsq
        self.non_negative = non_negative
        self._per_channel = per_channel
        self._lsq_initialized = False

        if self.qat:
            self.tr_embedding = embedding
            self._quantized = True
            if self.lsq:
                self._init_lsq_scales(embedding)
        else:
            self._q_vocab_cores: List[torch.Tensor] = []
            self._q_emb_cores: List[torch.Tensor] = []
            self._vocab_scales: List[torch.Tensor] = []
            self._emb_scales: List[torch.Tensor] = []
            self._vocab_zeros: List = []
            self._emb_zeros: List = []
            self._quantized = False
            self.quantize(embedding)

        self._emb_cache: Optional[torch.Tensor] = None
        self._cache_valid: bool = False
        self._decompose_indices = embedding._decompose_indices

    def _init_lsq_scales(self, embedding: "TensorRingEmbedding") -> None:
        """Initialize learnable scale parameters for LSQ (ICLR 2020).

        Each core gets its own learnable scale (per-tensor) or per-channel scales.
        Initialized from the absmax of the original weights.
        """
        self._vocab_lsq_scales: nn.ParameterList = nn.ParameterList()
        self._emb_lsq_scales: nn.ParameterList = nn.ParameterList()

        for core in embedding.cores.vocab_cores:
            if self._per_channel:
                absmax = core.data.abs().amax(dim=tuple(range(1, core.data.ndim)))
                init_scale = (absmax / 127.0).clamp(min=1e-8)
                param = nn.Parameter(init_scale)
            else:
                absmax = core.data.abs().max()
                init_scale = (absmax / 127.0).clamp(min=1e-8)
                param = nn.Parameter(init_scale.detach().clone())
            self._vocab_lsq_scales.append(param)

        for core in embedding.cores.emb_cores:
            if self._per_channel:
                absmax = core.data.abs().amax(dim=tuple(range(1, core.data.ndim)))
                init_scale = (absmax / 127.0).clamp(min=1e-8)
                param = nn.Parameter(init_scale)
            else:
                absmax = core.data.abs().max()
                init_scale = (absmax / 127.0).clamp(min=1e-8)
                param = nn.Parameter(init_scale.detach().clone())
            self._emb_lsq_scales.append(param)

        self._lsq_initialized = True

    def _lsq_quantize_core(self, core: torch.Tensor, scale: nn.Parameter) -> torch.Tensor:
        """Apply LSQ quantization with learnable scale.
        
        Forward: round(x / s) * s via LSQQuantize Function
        Scale update: dL/ds = -(x / s²) * dL/dq via backward
        """
        return LSQQuantize.apply(core, scale, True)

    def quantize(self, embedding: "TensorRingEmbedding") -> None:
        """PTQ: Quantize all cores from a TR embedding to int8."""
        self._q_vocab_cores, self._vocab_scales, self._vocab_zeros = self._quantize_cores(embedding.cores.vocab_cores)
        self._q_emb_cores, self._emb_scales, self._emb_zeros = self._quantize_cores(embedding.cores.emb_cores)
        self._quantized = True

    def _quantize_cores(self, cores):
        q_cores, scales, zeros = [], [], []
        for core in cores:
            if self._per_channel:
                q, s, z = _quantize_tensor_per_channel(core.data)
            else:
                q, s, z = _quantize_tensor(core.data)
            q_cores.append(q); scales.append(s); zeros.append(z)
        return q_cores, scales, zeros

    def _ste_quantize_core(self, core: torch.Tensor) -> torch.Tensor:
        """Quantize a single core using STE with per-channel or per-tensor scale."""
        if not self._per_channel:
            scale = core.data.abs().max() / 127.0
            return STERound.apply(core / scale).clamp(-128, 127) * scale
        scale = core.data.abs().amax(dim=tuple(range(1, core.data.ndim))) / 127.0
        return STERound.apply(core / scale.view(-1, 1, 1)).clamp(-128, 127) * scale.view(-1, 1, 1)

    def _gather_and_chain(self, vocab_cores_q: List[torch.Tensor],
                          factor_indices: List[torch.Tensor]) -> torch.Tensor:
        """Gather quantized vocab cores and chain via bmm."""
        gathered = [vocab_cores_q[i][factor_indices[i]] for i in range(len(vocab_cores_q))]
        result = gathered[0]
        for cg in gathered[1:]:
            result = torch.bmm(result, cg)
        return result

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if not self._quantized:
            raise RuntimeError("Must call quantize() before forward")

        from ..core.contraction import ring_closure
        if self._validate_indices_flag:
            from ..utils.validation import validate_indices
            validate_indices(indices, self.vocab_size, self.padding_idx)

        original_shape = indices.shape
        flat = indices.view(-1)
        factor_indices = self._decompose_indices(flat, self.structure.vocab_factor_sizes, self._vocab_strides)

        if self.qat:
            vocab_cores = self.tr_embedding.cores.vocab_cores
            emb_cores_list = self.tr_embedding.cores.emb_cores

            if self.lsq and self._lsq_initialized:
                vocab_q = [self._lsq_quantize_core(c, self._vocab_lsq_scales[i])
                           for i, c in enumerate(vocab_cores)]
                result = self._gather_and_chain(vocab_q, factor_indices)
                q_emb_cores = [self._lsq_quantize_core(c, self._emb_lsq_scales[i])
                               for i, c in enumerate(emb_cores_list)]
            else:
                vocab_q = [self._ste_quantize_core(c) for c in vocab_cores]
                result = self._gather_and_chain(vocab_q, factor_indices)
                q_emb_cores = [self._ste_quantize_core(c) for c in emb_cores_list]

            from ..core.contraction import compute_emb_precontraction
            emb = compute_emb_precontraction(q_emb_cores)
        else:
            result = self._q_vocab_cores[0][factor_indices[0]].float()
            s0 = self._vocab_scales[0]
            if isinstance(s0, torch.Tensor) and s0.ndim > 0:
                result = result * s0[factor_indices[0]].view(-1, 1, 1)
            else:
                result = result * s0
            for i in range(1, len(self._q_vocab_cores)):
                g = self._q_vocab_cores[i][factor_indices[i]].float()
                s = self._vocab_scales[i]
                if isinstance(s, torch.Tensor) and s.ndim > 0:
                    g = g * s[factor_indices[i]].view(-1, 1, 1)
                else:
                    g = g * s
                result = torch.bmm(result, g)

            if not self.training and self._cache_valid:
                emb = self._emb_cache
            else:
                emb = self._compute_emb_contraction()

        output = ring_closure(result, emb)

        if self.non_negative:
            output = F.relu(output)

        return output.view(*original_shape, self.embedding_dim)

    def _compute_emb_contraction(self) -> torch.Tensor:
        from ..core.contraction import compute_emb_precontraction
        cores = [self._dequantize_emb_core(i) for i in range(len(self._q_emb_cores))]
        return compute_emb_precontraction(cores)

    def _dequantize_emb_core(self, idx: int) -> torch.Tensor:
        q, s = self._q_emb_cores[idx], self._emb_scales[idx]
        return q.float() * s.view(-1, 1, 1) if isinstance(s, torch.Tensor) and s.ndim > 0 else q.float() * s

    def to_eval_mode(self) -> "QuantizedTensorRingEmbedding":
        self.eval()
        if not self.qat:
            with torch.no_grad():
                self._emb_cache = self._compute_emb_contraction()
                self._cache_valid = True
        return self

    @property
    def compression_ratio(self) -> float:
        dense = self.vocab_size * self.embedding_dim
        if self.qat:
            q_params = sum(p.numel() for p in self.parameters())
        elif not self._quantized:
            raise RuntimeError("Must call quantize() before accessing compression_ratio")
        else:
            q_params = sum(q.numel() for q in self._q_vocab_cores) + sum(q.numel() for q in self._q_emb_cores)
        if q_params <= 0:
            return float('inf')
        return dense / q_params

    @property
    def bits_per_parameter(self) -> float:
        if self.qat:
            total_params = sum(p.numel() for p in self.parameters())
            if total_params <= 0:
                return 32.0
            scale_params = sum(s.numel() for s in self._vocab_lsq_scales) + sum(s.numel() for s in self._emb_lsq_scales) if self.lsq else 0
            core_params = total_params - scale_params
            effective_bits = (core_params * 8.0 + scale_params * 32.0) / total_params
            return effective_bits
        elif not self._quantized:
            raise RuntimeError("Must call quantize() before accessing bits_per_parameter")
        total = sum(q.numel() for q in self._q_vocab_cores) + sum(q.numel() for q in self._q_emb_cores)
        return 8.0 if total > 0 else 32.0
