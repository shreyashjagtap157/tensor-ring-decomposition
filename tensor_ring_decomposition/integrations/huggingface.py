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
        **kwargs,
    ) -> TensorRingEmbedding:
        """Load HF model embedding, decompose via TR.

        Delegates to ``load_from_transformers`` for efficient model loading
        (low_cpu_mem_usage, cache management), then creates the TR embedding.
        """
        from ..loaders.loaders import load_from_transformers

        weight = load_from_transformers(model_name)
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

        Only replaces the model's input embedding (and tied lm_head if applicable),
        not position embeddings, token type embeddings, or other nn.Embedding layers.
        Validates compatibility before replacement.
        """
        validate_compatibility(tr_embedding, model)

        input_emb = model.get_input_embeddings()
        if input_emb is None:
            raise ValueError(f"Model {type(model).__name__} has no input embedding")

        replaced_names = []
        for name, module in model.named_modules():
            if module is input_emb:
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

        # Note: lm_head tied embedding weights are NOT replaced here.
        # Replacing lm_head.weight with tr_embedding.weight would materialize
        # the full V×D dense matrix, defeating compression. The model's output
        # probabilities after replacement will be computed using the TR embedding's
        # forward path through the linear head. For full tied-weight behavior,
        # consider wrapping lm_head in a custom module that uses tr_embedding.
        if hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings:
            logger.warning(
                "Model has tied word embeddings. The input embedding has been replaced "
                "with TensorRingEmbedding, but lm_head.weight still references the "
                "original dense embedding weight. Consider updating lm_head manually."
            )

        return model
