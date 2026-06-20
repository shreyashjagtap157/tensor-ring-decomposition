"""Multi-format embedding matrix loader.

Supports PyTorch, safetensors, NumPy, GGUF, and HuggingFace Transformers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

# Security constants
DEFAULT_MAX_MODEL_SIZE_GB = 5
DEFAULT_DOWNLOAD_TIMEOUT = 300
KNOWN_SAFE_MODEL_PREFIXES = (
    "bert-", "roberta-", "distilbert-", "albert-", "electra-", "xlnet-",
    "gpt2", "llama-", "mistral-", "falcon-", "opt-", "t5-", "bart-",
    "deberta-", "mpnet-", "camembert-", "bloom-", "codellama-",
    "qwen2-", "gemma-", "phi-", "mixtral-", "starcoder2-", "cohere-",
    "dbrx-", "phi3-", "qwen-", "yi-", "deepseek-", "baichuan-", "chatglm-"
)


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
    trust_remote_code: bool = False,
    max_model_size_gb: float = DEFAULT_MAX_MODEL_SIZE_GB,
    download_timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
) -> torch.Tensor:
    """Load embedding weight from a HuggingFace Transformers model.
    
    Args:
        model_name: HuggingFace model identifier (e.g., 'bert-base-uncased').
        device: Target device.
        cache_dir: Optional cache directory for HF downloads. If None, defaults to HF cache.
        trust_remote_code: Whether to allow execution of remote code from model repo.
                          Default False for security. Set True only for trusted models.
        max_model_size_gb: Maximum allowed model size in GB. Prevents OOM on huge models.
        download_timeout: Download timeout in seconds.
    
    Returns:
        Tensor of shape (vocab_size, embedding_dim).
    
    Raises:
        ValueError: If model is not in allowlist and trust_remote_code=False.
        RuntimeError: If model size exceeds max_model_size_gb.
    """
    # Security: Validate model name against allowlist
    if not trust_remote_code:
        model_lower = model_name.lower()
        if not any(model_lower.startswith(prefix) for prefix in KNOWN_SAFE_MODEL_PREFIXES):
            raise ValueError(
                f"Model '{model_name}' not in known safe model allowlist. "
                f"Set trust_remote_code=True to load anyway (only for trusted sources)."
            )
    
    if cache_dir is None:
        # Use the user-aware HF cache (~/.cache/huggingface) by default;
        # os.getcwd() is process-stable but varies per invocation and is
        # rarely what library users expect. Falling back to the cwd-derived
        # directory only as a last resort preserves old behavior.
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        if not os.path.isdir(cache_dir):
            cache_dir = os.path.join(os.getcwd(), ".hf_cache")
            os.makedirs(cache_dir, exist_ok=True)

    weight = _load_embedding_only(model_name, cache_dir=cache_dir, device=device, download_timeout=download_timeout)
    if weight is not None:
        return weight

    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError(
            "transformers package not found. Install with: pip install transformers"
        )
    
    # Check model size before loading
    _check_model_size(model_name, cache_dir, max_model_size_gb, download_timeout)
    
    model = AutoModel.from_pretrained(
        model_name, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    emb = model.get_input_embeddings()
    if emb is None:
        raise ValueError(f"Model '{model_name}' has no input embeddings")
    weight = emb.weight.data

    if device:
        weight = weight.to(device)
    
    logger.info(
        f"Loaded '{model_name}' from HF: shape={tuple(weight.shape)} (cached at {cache_dir})"
    )
    return weight


def _check_model_size(
    model_name: str,
    cache_dir: str,
    max_size_gb: float,
    timeout: int,
) -> None:
    """Check model size before downloading to prevent OOM."""
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        logger.warning("huggingface_hub not available, skipping size check")
        return
    
    try:
        files = list_repo_files(model_name, repo_type="model")
    except Exception as e:
        logger.warning(f"Could not list repo files for size check: {e}")
        return
    
    # Find safetensors files
    safetensor_files = [f for f in files if f.endswith(".safetensors") and "index" not in f]
    index_files = [f for f in files if f.endswith("model.safetensors.index.json")]
    
    total_size = 0
    if index_files:
        # Sharded model - check index
        try:
            index_path = hf_hub_download(model_name, index_files[0], cache_dir=cache_dir)
            import json
            with open(index_path) as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})
            # Sum sizes from weight map
            for safetensor_file in set(weight_map.values()):
                file_path = hf_hub_download(model_name, safetensor_file, cache_dir=cache_dir)
                total_size += os.path.getsize(file_path)
        except Exception as e:
            logger.warning(f"Could not check sharded model size: {e}")
    elif safetensor_files:
        try:
            for sf in safetensor_files:
                file_path = hf_hub_download(model_name, sf, cache_dir=cache_dir)
                total_size += os.path.getsize(file_path)
        except Exception as e:
            logger.warning(f"Could not check model size: {e}")
    
    size_gb = total_size / (1024 ** 3)
    if size_gb > max_size_gb:
        raise RuntimeError(
            f"Model '{model_name}' size ({size_gb:.1f} GB) exceeds limit ({max_size_gb} GB). "
            f"Increase max_model_size_gb or use a smaller model."
        )
    
    logger.info(f"Model '{model_name}' size check passed: {size_gb:.2f} GB")


# Mapping from model type to the expected embedding weight key in safetensors
_EMBEDDING_KEYS = {
    "bert": "embeddings.word_embeddings.weight",
    "roberta": "roberta.embeddings.word_embeddings.weight",
    "gpt2": "wte.weight",
    "llama": "model.embed_tokens.weight",
    "mistral": "model.embed_tokens.weight",
    "falcon": "transformer.word_embeddings.weight",
    "gptj": "transformer.wte.weight",
    "opt": "model.decoder.embed_tokens.weight",
    "bloom": "word_embeddings.weight",
    "t5": "shared.weight",
    "deberta": "embeddings.word_embeddings.weight",
    "electra": "electra.embeddings.word_embeddings.weight",
    "xlm_roberta": "roberta.embeddings.word_embeddings.weight",
    "albert": "albert.embeddings.word_embeddings.weight",
    "camembert": "roberta.embeddings.word_embeddings.weight",
    "distilbert": "distilbert.embeddings.word_embeddings.weight",
    "mpnet": "mpnet.embeddings.word_embeddings.weight",
    "qwen2": "model.embed_tokens.weight",
    "gemma": "model.embed_tokens.weight",
    "phi3": "model.embed_tokens.weight",
    "starcoder2": "model.embed_tokens.weight",
    "cohere": "model.embed_tokens.weight",
    "dbrx": "wte.weight",
}

# Alternative key suffixes for fallback detection
_EMBEDDING_KEY_SUFFIXES = [
    ".word_embeddings.weight",
    ".embed_tokens.weight",
    "wte.weight",
    ".word_embeddings",
    "shared.weight",
    "embed_tokens.weight",
    "tok_embeddings.weight",
    "input_embeddings.weight",
]


def _load_embedding_only(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    download_timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
) -> Optional[torch.Tensor]:
    """Load only the embedding weight from a HF model without loading the full model.
    
    Uses the safetensors index to find and download only the weight file
    containing the embedding matrix. Falls back to None if detection fails.
    """
    import json
    import os

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        return None

    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), ".hf_cache")

    try:
        files = list_repo_files(model_name, repo_type="model")
    except Exception:
        return None

    index_files = [f for f in files if f.endswith("model.safetensors.index.json")]
    single_files = [f for f in files if f.endswith(".safetensors") and "index" not in f]

    if index_files:
        index_path = hf_hub_download(model_name, index_files[0], cache_dir=cache_dir)
        with open(index_path) as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})

        target_key = _EMBEDDING_KEYS.get(model_name.split("/")[-1].split("-")[0].lower(), None)
        if target_key is None:
            for suffix in _EMBEDDING_KEY_SUFFIXES:
                matches = {k: v for k, v in weight_map.items() if k.endswith(suffix)}
                if matches:
                    target_key = list(matches.keys())[0]
                    break
        if target_key is None:
            # Compute file sizes once (was O(n²) disk access in max() comparator).
            index_dir = os.path.dirname(index_path)
            def _file_sz(rel: str) -> int:
                p = os.path.join(index_dir, rel)
                try:
                    return os.path.getsize(p)
                except OSError:
                    return 0
            largest_key = max(
                weight_map,
                key=lambda k: _file_sz(weight_map[k]),
            )
            target_key = largest_key

        if target_key and target_key in weight_map:
            safetensor_file = weight_map[target_key]
            safetensor_path = hf_hub_download(model_name, safetensor_file, cache_dir=cache_dir)
            return load_from_safetensors(safetensor_path, key=target_key, device=device)
    elif single_files:
        safetensor_path = hf_hub_download(model_name, single_files[0], cache_dir=cache_dir)
        weight = load_from_safetensors(safetensor_path, device=device)
        if weight is not None:
            return weight

    return None


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
    trust_remote_code: bool = False,
    max_model_size_gb: float = DEFAULT_MAX_MODEL_SIZE_GB,
    download_timeout: int = DEFAULT_DOWNLOAD_TIMEOUT,
) -> torch.Tensor:
    """Universal loader: detect format and load embedding matrix.

    Args:
        source: File path or HuggingFace model name.
        format: Explicit format ('torch', 'safetensors', 'numpy', 'gguf', 'transformers').
                If None, auto-detect from path extension or try transformers.
        key: Specific key for torch/safetensors state dicts.
        device: Target device.
        cache_dir: Cache dir for HuggingFace downloads.
        trust_remote_code: Whether to allow execution of remote code from model repo.
                          Default False for security. Set True only for trusted models.
        max_model_size_gb: Maximum allowed model size in GB. Prevents OOM on huge models.
        download_timeout: Download timeout in seconds.

    Returns:
        Tensor of shape (vocab_size, embedding_dim).

    Raises:
        FileNotFoundError: If path does not exist and can't be loaded from HF.
        ImportError: If required package not installed.
        ValueError: If model not in allowlist and trust_remote_code=False.
        RuntimeError: If model size exceeds max_model_size_gb.
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
            except Exception as e:
                logger.debug(f"Format '{try_fmt}' failed for {source}: {e}")
                continue

    return load_from_transformers(
        source,
        device,
        cache_dir,
    )
