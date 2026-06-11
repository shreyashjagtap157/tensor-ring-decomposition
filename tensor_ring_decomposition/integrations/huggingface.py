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
        """Load HF model, extract input embedding, decompose via TR."""
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name)
        original_emb = model.get_input_embeddings()

        tr_emb = TensorRingEmbedding.from_pretrained(
            original_emb.weight.data, rank, ring_components, **kwargs
        )

        return tr_emb

    @classmethod
    def replace_in_model(
        cls,
        model: "PreTrainedModel",
        tr_embedding: TensorRingEmbedding,
    ) -> "PreTrainedModel":
        """Replace all nn.Embedding layers with TensorRingEmbedding.

        Validates compatibility before replacement.
        """
        validate_compatibility(tr_embedding, model)

        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
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

        return model
