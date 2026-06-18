"""High-level convenience API: one-call model compression."""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch

from .core.embedding import TensorRingEmbedding
from .models.registry import ModelProfile, ModelRegistry
from .loaders.loaders import load_embedding_matrix

logger = logging.getLogger(__name__)


def compress(
    source: Union[str, torch.Tensor, ModelProfile],
    rank: Optional[int] = None,
    target_compression: Optional[float] = None,
    target_params: Optional[int] = None,
    target_mse: Optional[float] = None,
    ring_components: int = 4,
    init_method: str = "svd",
    device: Optional[torch.device] = None,
    autotune: bool = False,
    **kwargs,
) -> TensorRingEmbedding:
    """Compress an embedding matrix using Tensor Ring decomposition.

    This is the single-entry-point convenience function that integrates the
    model registry, multi-format loaders, and TR embedding into one call.

    Args:
        source: One of:
            - A file path (``.bin``, ``.pt``, ``.safetensors``, ``.npy``, ``.gguf``)
            - A HuggingFace model name (e.g., ``"bert-base-uncased"``)
            - A ``(V, D)`` tensor
            - A ``ModelProfile`` (to create from scratch)
        rank: TR rank. If not set, inferred from ``target_compression``,
            ``target_params``, ``target_mse``, or the model's default rank.
        target_compression: Desired compression ratio (e.g., 10x).
        target_params: Desired parameter budget.
        target_mse: Desired relative mean-squared error.
        ring_components: Number of ring components (default 4).
        init_method: ``"svd"`` (1000-step training) or ``"tr_svd"`` (700-step fast).
        device: Torch device.
        autotune: If True, run ``autotune`` to find optimal rank.
        **kwargs: Additional args for ``TensorRingEmbedding``.

    Returns:
        TensorRingEmbedding compressing the source embeddings.

    Raises:
        ValueError: If no rank or target constraint is given and no default
                    can be inferred.
        FileNotFoundError: If a file path does not exist.
        ImportError: If required package is missing.
    """
    # ── Resolve source to a matrix and optional profile ──────
    profile: Optional[ModelProfile] = None
    matrix: Optional[torch.Tensor] = None

    if isinstance(source, ModelProfile):
        profile = source
    elif isinstance(source, str):
        profile = ModelRegistry.get(source)
        if profile is not None:
            # If a matrix-based init is requested, we MUST load the matrix from HF
            if init_method in ("svd", "tr_svd", "als", "distribution_aware"):
                try:
                    matrix = load_embedding_matrix(source, device=device)
                except Exception as e:
                    if init_method in ("svd", "tr_svd", "als", "distribution_aware"):
                        logger.warning(f"Failed to load matrix for {source}: {e}. Falling back to uniform init. "
                                       f"This may produce lower-quality embeddings. "
                                       f"To avoid this, ensure the model is accessible or provide a tensor directly.")
                
        if profile is None:
            # Assume it's a file path or direct HF model name
            try:
                matrix = load_embedding_matrix(source, device=device)
            except Exception as e:
                raise FileNotFoundError(f"Could not load embeddings from {source}: {e}")
    elif isinstance(source, torch.Tensor):
        matrix = source
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # ── Determine V, D and rank ──────────────────────────────
    if profile is not None and matrix is None:
        V, D = profile.vocab_size, profile.embedding_dim
    elif matrix is not None:
        V, D = matrix.shape
    else:
        raise ValueError("Could not determine vocab size / embedding dim.")

    if rank is None:
        rank = _resolve_rank(
            V, D, profile, ring_components,
            target_compression, target_params, target_mse,
        )

    # ── Autotune if requested ────────────────────────────────
    if autotune and matrix is not None:
        result = TensorRingEmbedding.autotune(
            matrix, ring_components=ring_components,
            target_compression=target_compression,
            target_params=target_params,
            target_mse=target_mse,
            verbose=False,
            device=device,
        )
        rank = result.rank
        logger.info(f"Autotune selected rank={rank}")

    # ── Build the embedding ──────────────────────────────────
    if matrix is not None:
        # Matrix available: use requested init_method
        emb = TensorRingEmbedding.from_pretrained(
            matrix, rank, ring_components=ring_components,
            init_method=init_method, device=device, **kwargs,
        )
    elif profile is not None:
        # Profile only: must use random init
        if init_method in ("svd", "tr_svd", "als", "distribution_aware"):
            logger.info(
                f"Matrix not available for '{source}'; using uniform init instead of {init_method}. "
                f"Pass a tensor or use from_huggingface for matrix-based init."
            )
        actual_init = init_method if init_method not in ("svd", "tr_svd", "als", "distribution_aware") else "uniform"
        emb = TensorRingEmbedding.from_profile(
            profile, rank=rank, target_compression=target_compression,
            init_method=actual_init, device=device, **kwargs,
        )
    else:
        raise RuntimeError("Unexpected state: neither profile nor matrix available.")

    logger.info(
        f"Compressed {V}×{D} → rank={rank}, "
        f"{emb.compression_ratio:.1f}x compression, "
        f"{emb.num_parameters:,} params"
    )
    return emb


def _resolve_rank(
    V: int,
    D: int,
    profile: Optional[ModelProfile],
    ring_components: int,
    target_compression: Optional[float],
    target_params: Optional[int],
    target_mse: Optional[float],
) -> int:
    """Resolve rank from constraints or profile defaults."""
    if target_params is not None:
        return TensorRingEmbedding.optimal_rank(V, D, ring_components, target_params=target_params)
    if target_compression is not None:
        return TensorRingEmbedding.optimal_rank(V, D, ring_components, target_compression=target_compression)
    if target_mse is not None:
        return TensorRingEmbedding.optimal_rank(V, D, ring_components, target_mse=target_mse)
    if profile is not None:
        return profile.default_rank
    return 8  # reasonable default


def list_models(family: Optional[str] = None) -> str:
    """List all registered models, optionally filtered by family.

    Args:
        family: Optional family name (e.g., ``"llama"``, ``"bert"``).

    Returns:
        Formatted string summary.
    """
    if family:
        from .models.registry import ModelFamily
        try:
            fam = ModelFamily(family.lower())
        except ValueError:
            available = [e.value for e in ModelFamily]
            return f"Unknown family '{family}'. Available: {available}"
        profiles = ModelRegistry.list_by_family(fam)
        return f"Models in family '{family}':\n" + "\n".join(
            f"  {p.name:30s} V={p.vocab_size:<6} D={p.embedding_dim:<4} "
            f"comp={p.compression_at_rank(p.default_rank):.0f}x@R{p.default_rank}"
            for p in profiles
        )
    return ModelRegistry.summary()
