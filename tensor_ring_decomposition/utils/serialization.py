"""Save/load with safetensors + HMAC verification."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from ..core.embedding import TensorRingEmbedding

CHECKPOINT_SCHEMA_VERSION = "1.0"


class SecurityError(Exception):
    """Raised when checkpoint hash verification fails."""


def save(
    embedding: "TensorRingEmbedding",
    path: str,
    secret_key: Optional[bytes] = None,
    extra_metadata: Optional[dict] = None,
) -> None:
    """Save TR embedding with safetensors + HMAC-verified manifest.

    Creates two files:
    - {path}.safetensors: Weights
    - {path}.json: Manifest with metadata and hash

    Weights are serialized to bytes once; the same bytes are written to
    disk and used for hash computation (eliminates double I/O).
    """
    import safetensors.torch as sf

    # Collect weights
    weights = {name: param.data for name, param in embedding.named_parameters()}

    # Serialize to bytes, write to disk, compute hash from same bytes
    weights_bytes = sf.save(weights)
    weights_path = Path(path).with_suffix(".safetensors")
    weights_path.write_bytes(weights_bytes)

    if secret_key:
        core_hash = hmac.new(secret_key, weights_bytes, hashlib.sha256).hexdigest()
        hash_type = "hmac-sha256"
    else:
        core_hash = hashlib.sha256(weights_bytes).hexdigest()
        hash_type = "sha256"

    # Build manifest
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "tr_config": embedding.config(),
        "weights_file": weights_path.name,
        "core_hash": core_hash,
        "hash_type": hash_type,
        "weight_count": len(weights),
        "weight_shapes": {
            name: list(param.shape)
            for name, param in embedding.named_parameters()
        },
        "compression_metrics": {
            "compression_ratio": embedding.compression_ratio,
            "num_parameters": embedding.num_parameters,
            "dense_parameters": embedding.vocab_size * embedding.embedding_dim,
            "params_saved": embedding.vocab_size * embedding.embedding_dim - embedding.num_parameters,
        },
    }
    if extra_metadata:
        manifest.update(extra_metadata)

    # Save manifest
    manifest_path = Path(path).with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))


def load(
    path: str,
    secret_key: Optional[bytes] = None,
    device: Optional[torch.device] = None,
) -> "TensorRingEmbedding":
    """Load TR embedding from safetensors + manifest.

    Verifies hash before loading. Uses safetensors (not torch.load).
    Weights file is read once; bytes are reused for hash verification.
    """
    import safetensors.torch as sf

    from ..core.embedding import TensorRingEmbedding

    # Load manifest
    manifest_path = Path(path).with_suffix(".json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())

    # Verify hash (single read of weights bytes)
    weights_path = Path(path).parent / manifest["weights_file"]
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    weights_bytes = weights_path.read_bytes()
    if secret_key:
        expected = hmac.new(secret_key, weights_bytes, hashlib.sha256).hexdigest()
    else:
        expected = hashlib.sha256(weights_bytes).hexdigest()

    if expected != manifest["core_hash"]:
        raise SecurityError(
            f"Hash mismatch! Expected {expected[:16]}..., "
            f"got {manifest['core_hash'][:16]}... "
            f"This checkpoint may have been tampered with."
        )

    # Load weights from bytes (no second file read)
    weights = sf.load(weights_bytes)
    if device is not None:
        weights = {k: v.to(device) for k, v in weights.items()}

    # Reconstruct embedding with full config roundtrip
    config = manifest["tr_config"]
    embedding = TensorRingEmbedding(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        rank=config["rank"] if config.get("rank") is not None else max(config.get("ranks", [4])),
        ring_components=config.get("ring_components", 4),
        split_mode=config.get("split_mode", "balanced"),
        init_method="uniform",
        gauge_fix=config.get("gauge_fix", "left"),
        gauge_fix_interval=config.get("gauge_fix_interval", 1000),
        padding_idx=config.get("padding_idx"),
        max_seq_len=config.get("max_seq_len"),
        spectral_reg_coeff=config.get("spectral_reg_coeff", 0.0),
    )
    embedding.load_state_dict(weights)

    return embedding
