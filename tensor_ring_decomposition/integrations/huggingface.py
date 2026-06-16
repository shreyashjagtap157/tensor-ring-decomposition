"""HuggingFace integration for Tensor Ring Embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch.nn as nn

from ..core.embedding import TensorRingEmbedding
from ..utils.validation import validate_compatibility

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

        Uses a lightweight loading approach to avoid downloading the entire
        model just for the embedding table. Loads with low_cpu_mem_usage=True
        and immediately discards the model after extracting the embedding weight.
        """
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)
        original_emb = model.get_input_embeddings()
        weight = original_emb.weight.data.clone()

        del model
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
                new_tr_emb = TensorRingEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    rank=tr_embedding.rank,
                )
                new_tr_emb.cores.initialize("svd", module.weight.data)

                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                setattr(parent, child_name, new_tr_emb)
                replaced_names.append(name)

        if not replaced_names:
            raise ValueError(
                f"Could not find input embedding layer in model {type(model).__name__}"
            )

        # Also replace tied output embeddings if they point to the same weight
        if hasattr(model.config, "tie_word_embeddings") and model.config.tie_word_embeddings:
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and hasattr(lm_head, "weight"):
                lm_head.weight = new_tr_emb.weight

        return model
