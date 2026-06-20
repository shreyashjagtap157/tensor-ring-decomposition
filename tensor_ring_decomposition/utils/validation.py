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
    Skips validation during TorchScript tracing to avoid TracerWarnings.

    Args:
        indices: Tensor of token indices.
        vocab_size: Maximum valid index (exclusive upper bound).
        padding_idx: Optional padding index. When set, ``-1`` is allowed
            as a padding marker in addition to ``[0, vocab_size)``.
    """
    if torch.jit.is_tracing():
        return
    if indices.numel() == 0:
        return

    # Compute min/max once to avoid duplicate reductions and to keep the
    # negative-index branch consistent with the upper-bound branch below.
    min_val = indices.min()
    max_val = indices.max()

    if min_val < 0:
        if padding_idx is not None:
            # Allow -1 (the standard padding sentinel) but nothing smaller.
            if min_val < -1:
                raise IndexError(
                    f"Indices contain values < -1 (min={min_val.item()}). "
                    f"Only -1 is allowed as a padding marker when padding_idx is set."
                )
        else:
            raise IndexError(
                f"Indices contain negative values (min={min_val.item()}). "
                f"Set padding_idx if negative indices are intentional."
            )

    if max_val >= vocab_size:
        raise IndexError(
            f"Index {max_val.item()} out of range for vocab_size={vocab_size}"
        )


def validate_compatibility(
    embedding: "TensorRingEmbedding",
    downstream_module: nn.Module,
) -> None:
    """Check that TR embedding output dimension matches every downstream Linear.

    Iterates *all* ``nn.Linear`` sub-modules of ``downstream_module`` and
    validates that each one's ``in_features`` matches the embedding's output
    dimension. Raises ``ValueError`` on the first mismatch (with the module
    path so users can locate the offending layer).

    Args:
        embedding: The tensor ring embedding providing output vectors.
        downstream_module: Container module (e.g. the rest of a model) that
            receives the embedding's output.
    """
    mismatches = []
    for name, module in downstream_module.named_modules():
        if isinstance(module, nn.Linear):
            if module.in_features != embedding.embedding_dim:
                mismatches.append(
                    f"  - '{name}' expects {module.in_features}, "
                    f"got {embedding.embedding_dim}"
                )
    if mismatches:
        joined = "\n".join(mismatches)
        raise ValueError(
            f"Downstream module has {len(mismatches)} Linear layer(s) whose "
            f"in_features does not match the TR embedding output dim "
            f"({embedding.embedding_dim}):\n{joined}"
        )
