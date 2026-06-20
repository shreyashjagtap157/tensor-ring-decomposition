"""HuggingFace integration for Tensor Ring Embeddings."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch.nn as nn

from ..core.embedding import TensorRingEmbedding
from ..utils.validation import validate_compatibility

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class HuggingFaceTensorRingEmbedding:
    """Replace nn.Embedding in HuggingFace models with TensorRingEmbedding."""

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        rank: int,
        ring_components: int = 4,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        max_model_size_gb: float = 5.0,
        download_timeout: int = 300,
        **kwargs,
    ) -> TensorRingEmbedding:
        """Load HF model embedding, decompose via TR.

        Delegates to ``load_from_transformers`` for efficient model loading
        (low_cpu_mem_usage, cache management), then creates the TR embedding.

        Args:
            model_name: HF model identifier.
            rank: TR rank.
            ring_components: Number of ring components.
            cache_dir: HF cache directory.
            trust_remote_code: Whether to allow execution of remote code. Default False.
            max_model_size_gb: Max model size in GB to prevent OOM.
            download_timeout: Download timeout in seconds.
            **kwargs: Additional args passed to TensorRingEmbedding.
        """
        from ..loaders.loaders import load_from_transformers

        weight = load_from_transformers(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            max_model_size_gb=max_model_size_gb,
            download_timeout=download_timeout,
        )
        if weight.device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

        tr_emb = TensorRingEmbedding.from_pretrained(
            weight, rank, ring_components, **kwargs
        )

        return tr_emb

    @classmethod
    def replace_in_model(
        cls,
        model: "PreTrainedModel",
        tr_embedding: TensorRingEmbedding,
    ) -> "PreTrainedModel":
        """Replace input embedding layers with TensorRingEmbedding.

        Only replaces the model's word/input embedding,
        not position embeddings, token type embeddings, or other nn.Embedding layers.
        Validates compatibility before replacement.
        """
        validate_compatibility(tr_embedding, model)

        input_emb = model.get_input_embeddings()
        if input_emb is None:
            raise ValueError(f"Model {type(model).__name__} has no input embedding")

        skip_patterns = ("position", "token_type", "segment")

        replaced_names = []
        for name, module in model.named_modules():
            if module is not input_emb:
                continue
            if any(p in name.lower() for p in skip_patterns):
                continue
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            setattr(parent, child_name, tr_embedding)
            replaced_names.append(name)

        if not replaced_names:
            raise ValueError(
                f"Could not find input embedding layer in model {type(model).__name__}"
            )

        if hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings:
            logger.warning(
                "Model has tied word embeddings. The input embedding has been replaced "
                "with TensorRingEmbedding, but lm_head.weight still references the "
                "original dense embedding weight. Consider updating lm_head manually."
            )

        return model
