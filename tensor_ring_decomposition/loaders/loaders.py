"""Multi-format embedding matrix loader.

Supports PyTorch, safetensors, NumPy, GGUF, and HuggingFace Transformers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)


def guess_format(path: str) -> str:
    """Guess file format from extension."""
    ext = Path(path).suffix.lower()
    fmt_map = {
        ".bin": "torch",
        ".pt": "torch",
        ".pth": "torch",
        ".safetensors": "safetensors",
        ".npy": "numpy",
        ".npz": "numpy",
        ".gguf": "gguf",
    }
    if ext in fmt_map:
        return fmt_map[ext]
    return "unknown"


def load_from_torch(
    path: str,
    key: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Load embedding weight from a PyTorch checkpoint (.bin/.pt/.pth).

    Args:
        path: Path to the checkpoint file.
        key: Key to extract from state dict. If None, auto-detect.
        device: Target device.

    Returns:
        Tensor of shape (vocab_size, embedding_dim).
    """
    raw = torch.load(path, map_location=device, weights_only=True)

    if isinstance(raw, torch.Tensor):
        return raw

    if isinstance(raw, dict):
        if key is not None:
            return _resolve_key(raw, key)

        auto_keys = ["embeddings.word_embeddings.weight", "embeddings.token_embedding.weight",
                      "embed_tokens.weight", "shared.weight", "decoder.embed_tokens.weight",
                      "wte.weight", "embedding.word_embeddings.weight", "input_embeddings.weight"]

        for ak in auto_keys:
            if ak in raw:
                logger.info(f"Auto-detected key: '{ak}'")
                return raw[ak]

        tensor_keys = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor) and v.ndim == 2}
        if not tensor_keys:
            raise ValueError(f"No 2D tensor found. Available keys: {list(raw.keys())[:10]}")
        largest = max(tensor_keys, key=lambda k: tensor_keys[k].shape[0] * tensor_keys[k].shape[1])
        logger.warning(f"No standard key found. Using largest 2D tensor: '{largest}' "
                       f"shape={tuple(tensor_keys[largest].shape)}")
        return tensor_keys[largest]

    raise TypeError(f"Unsupported checkpoint type: {type(raw)}")


def load_from_safetensors(
    path: str,
    key: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Load embedding weight from a safetensors file.

    Args:
        path: Path to the .safetensors file.
        key: Specific key to extract. If None, auto-detect.
        device: Target device.

    Returns:
        Tensor of shape (vocab_size, embedding_dim).
    """
    import safetensors.torch as sf

    tensors = sf.load_file(str(path), device=device)

    if isinstance(tensors, torch.Tensor):
        return tensors

    if key is not None:
        return tensors[key]

    auto_keys = ["embeddings.word_embeddings.weight", "embeddings.token_embedding.weight",
                  "embed_tokens.weight", "shared.weight", "wte.weight",
                  "model.embed_tokens.weight", "lm_head.weight"]

    for ak in auto_keys:
        if ak in tensors:
            logger.info(f"Auto-detected key: '{ak}'")
            return tensors[ak]

    tensor_keys = {k: v for k, v in tensors.items() if isinstance(v, torch.Tensor) and v.ndim == 2}
    if not tensor_keys:
        raise ValueError(f"No 2D tensor found in safetensors file. Keys: {list(tensors.keys())[:10]}")
    largest = max(tensor_keys, key=lambda k: tensor_keys[k].shape[0] * tensor_keys[k].shape[1])
    logger.warning(f"No standard key found. Using largest 2D tensor: '{largest}' "
                   f"shape={tuple(tensor_keys[largest].shape)}")
    return tensor_keys[largest]


def load_from_numpy(path: str, device: Optional[torch.device] = None) -> torch.Tensor:
    """Load embedding weight from a NumPy file (.npy or .npz).

    Args:
        path: Path to the .npy or .npz file.
        device: Target device.

    Returns:
        Tensor of shape (vocab_size, embedding_dim).
    """
    import numpy as np

    p = Path(path)
    if p.suffix.lower() == ".npz":
        archive = np.load(path)
        keys = list(archive.keys())
        if len(keys) == 1:
            arr = archive[keys[0]]
        else:
            largest = max(keys, key=lambda k: archive[k].size)
            logger.warning(f"Multiple arrays in .npz, using largest: '{largest}'")
            arr = archive[largest]
    else:
        arr = np.load(path)

    return torch.from_numpy(arr).to(device)


def load_from_gguf(
    path: str,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Load embedding weight from a GGUF file (quantized model format).

    Uses the 'gguf' Python package to read tensor metadata and extract
    the token embedding weight. Falls back to reading raw tensors from
    supported GGUF files.

    Args:
        path: Path to the .gguf file.
        device: Target device.

    Returns:
        Tensor of shape (vocab_size, embedding_dim).
    """
    try:
        import gguf
    except ImportError:
        raise ImportError(
            "gguf Python package not found. Install with: pip install gguf"
        )

    reader = gguf.GGUFReader(str(path))

    token_emb_name = None
    for candidate in ["token_embd.weight", "tok_embeddings.weight",
                       "embed_tokens.weight", "wte.weight"]:
        try:
            tensor = reader.get_tensor(candidate)
            token_emb_name = candidate
            break
        except (ValueError, KeyError, AttributeError):
            continue

    if token_emb_name is None:
        available = [t.name for t in reader.tensors]
        raise KeyError(
            f"No embedding tensor found in GGUF file. "
            f"Available tensors (sample): {available[:10]}"
        )

    raw_tensor = reader.get_tensor(token_emb_name)
    arr = raw_tensor.data

    if hasattr(arr, "astype"):
        arr = arr.astype("float32")

    tensor = torch.from_numpy(arr)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    logger.info(f"Loaded GGUF tensor '{token_emb_name}': shape={tuple(tensor.shape)}")
    return tensor.to(device)


def load_from_transformers(
    model_name: str,
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """Load embedding weight from a HuggingFace Transformers model.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'bert-base-uncased').
        device: Target device.
        cache_dir: Optional cache directory for HF downloads. If None, defaults to '.hf_cache' in project root.
    
    Returns:
        Tensor of shape (vocab_size, embedding_dim).
    """
    if cache_dir is None:
        import os
        # Ensure all downloads are contained within the project folder
        project_root = os.getcwd()
        cache_dir = os.path.join(project_root, ".hf_cache")
        os.makedirs(cache_dir, exist_ok=True)

    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "transformers package not found. Install with: pip install transformers"
        )
    
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, low_cpu_mem_usage=True)
    emb = model.get_input_embeddings()
    weight = emb.weight.data
    
    logger.info(
        f"Loaded '{model_name}' from HF: shape={tuple(weight.shape)} (cached at {cache_dir})"
    )
    return weight.to(device) if device else weight


def _resolve_key(state_dict: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key in state_dict:
        return state_dict[key]
    # Try suffix match: "embed_tokens.weight" matches "...embed_tokens.weight"
    suffix_matches = [k for k in state_dict if k.endswith(key)]
    if suffix_matches:
        return state_dict[suffix_matches[0]]
    # Fallback to substring match (log warning)
    close_matches = [k for k in state_dict if key in k and isinstance(state_dict[k], torch.Tensor)]
    if close_matches:
        logger.warning(f"No exact/suffix match for '{key}'. Using substring match: '{close_matches[0]}'")
        return state_dict[close_matches[0]]
    raise KeyError(f"Key '{key}' not found. Available keys: {list(state_dict.keys())[:20]}")


def load_embedding_matrix(
    source: str,
    format: Optional[str] = None,
    key: Optional[str] = None,
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None,
) -> torch.Tensor:
    """Universal loader: detect format and load embedding matrix.

    Args:
        source: File path or HuggingFace model name.
        format: Explicit format ('torch', 'safetensors', 'numpy', 'gguf', 'transformers').
                If None, auto-detect from path extension or try transformers.
        key: Specific key for torch/safetensors state dicts.
        device: Target device.
        cache_dir: Cache dir for HuggingFace downloads.

    Returns:
        Tensor of shape (vocab_size, embedding_dim).

    Raises:
        FileNotFoundError: If path does not exist and can't be loaded from HF.
        ImportError: If required package not installed.
    """
    fmt = format or guess_format(source)

    loaders = {
        "torch": lambda: load_from_torch(source, key, device),
        "safetensors": lambda: load_from_safetensors(source, key, device),
        "numpy": lambda: load_from_numpy(source, device),
        "gguf": lambda: load_from_gguf(source, device),
    }

    if fmt != "unknown" and fmt != "transformers":
        if not Path(source).exists():
            raise FileNotFoundError(f"File not found: {source}")
        return loaders[fmt]()

    if Path(source).exists() and fmt == "unknown":
        for try_fmt in ["torch", "safetensors", "gguf", "numpy"]:
            try:
                loader = loaders[try_fmt]
                result = loader()
                logger.info(f"Auto-detected format '{try_fmt}' for {source}")
                return result
            except Exception:
                continue

    return load_from_transformers(source, device, cache_dir)
