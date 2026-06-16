"""Input validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..core.embedding import TensorRingEmbedding


def validate_indices(
    indices: torch.Tensor,
    vocab_size: int,
    padding_idx: Optional[int] = None,
) -> None:
    """Validate token indices are within bounds.

    Handles empty tensors gracefully (always passes for 0-element tensors).
    """
    if indices.numel() == 0:
        return
    if indices.min() < 0:
        if padding_idx is not None:
            if indices.min() < -1:
                raise IndexError(
                    f"Indices contain values < -1 (min={indices.min().item()}). "
                    f"Only -1 is allowed as a padding marker when padding_idx is set."
                )
        else:
            raise IndexError(
                f"Indices contain negative values (min={indices.min().item()}). "
                f"Set padding_idx if negative indices are intentional."
            )
    if indices.max() >= vocab_size:
        raise IndexError(
            f"Index {indices.max().item()} out of range for vocab_size={vocab_size}"
        )


def validate_compatibility(
    embedding: "TensorRingEmbedding",
    downstream_module: nn.Module,
) -> None:
    """Check that TR embedding output dimension matches downstream module input."""
    for name, module in downstream_module.named_modules():
        if isinstance(module, nn.Linear):
            if module.in_features != embedding.embedding_dim:
                raise ValueError(
                    f"Downstream module '{name}' expects input dim "
                    f"{module.in_features}, but TR embedding outputs dim "
                    f"{embedding.embedding_dim}"
                )
            break
