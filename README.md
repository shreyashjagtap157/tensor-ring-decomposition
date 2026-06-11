# Tensor Ring Decomposition for Embeddings

Drop-in replacement for `nn.Embedding` using Tensor Ring Decomposition (TRD), achieving 10–100× parameter reduction with minimal accuracy loss.

## Quick Start

```python
from tensor_ring_decomposition import TensorRingEmbedding

# Explicit rank
emb = TensorRingEmbedding(50000, 768, rank=8)

# Target compression ratio
emb = TensorRingEmbedding.from_compression_ratio(50000, 768, ratio=50)

# From pretrained dense embedding
emb = TensorRingEmbedding.from_pretrained(dense_weight, rank=8)

# Use like nn.Embedding
indices = torch.randint(0, 50000, (4, 16))
output = emb(indices)  # (4, 16, 768)

# High-level compress API
from tensor_ring_decomposition import compress
emb = compress("bert-base-uncased", target_compression=50)

# List available models
from tensor_ring_decomposition import list_models
print(list_models(family="llama"))
```

## Key Features

- **10-100x compression** of embedding tables via Tensor Ring Decomposition
- **Distribution-aware training** (NeurIPS 2025): minimizes output distribution shift via `||(W-Ŵ)Σ^{1/2}||_F`
- **Gradient-safe eval cache**: `to_eval_mode()` precomputes embedding contraction; `train_mode()` clears it
- **Multiple init methods**: uniform, normal, kaiming, SVD, TR-SVD, distribution-aware
- **Gauge fixing**: QR-based orthogonalization to prevent scale drift
- **Safe serialization**: safetensors + HMAC-verified manifest (no `torch.load`)
- **HuggingFace integration**: replace embeddings in transformer models
- **Int8 quantization**: post-training per-channel quantization
- **Autotuning**: automatic rank selection from compression / parameter / MSE budget
- **Spectral regularization**: spectral norm monitoring and regularization
- **Intrinsic evaluation**: eigenspace overlap, distribution-aware reconstruction error
- **DDP support**: distributed data parallel wrapper with gradient sync
- **Export**: ONNX and TorchScript export for production deployment
- **63+ model profiles**: built-in profiles for BERT, LLaMA, GPT, Mistral, Gemma, Phi, and more

## Installation

```bash
pip install tensor-ring-decomposition
```

Optional dependencies:
```bash
pip install tensor-ring-decomposition[hf]   # HuggingFace integration
pip install tensor-ring-decomposition[gguf]  # GGUF format support
pip install tensor-ring-decomposition[all]   # Everything
```

Requires PyTorch 2.0+, opt_einsum 3.3+, and safetensors 0.4+.

## Documentation

### Constructor

```python
TensorRingEmbedding(
    vocab_size, embedding_dim,
    rank=None,                     # TR rank (exactly one of these four)
    ranks=None,                    # Per-core ranks
    target_compression=None,       # Target compression ratio
    target_params=None,            # Target parameter count
    ring_components=4,             # Total number of cores
    split_mode="balanced",         # "balanced", "proportional", or "manual"
    init_method="uniform",         # "uniform", "normal", "kaiming", "svd", "tr_svd", "distribution_aware"
    gauge_fix="left",              # "none", "left", "right", or "both"
    gauge_fix_interval=1000,       # Steps between gauge fixes
    padding_idx=None,              # Padding token index
    max_seq_len=None,              # Max sequence length
    spectral_reg_coeff=0.0,        # Spectral regularization coefficient
    validate_indices=True,         # Validate indices in forward pass
    device=None,                   # Torch device
    dtype=torch.float32,           # Torch dtype (use bf16, not fp16)
)
```

### Methods

| Method | Description |
|--------|-------------|
| `forward(indices)` | Compressed embedding lookup |
| `to_eval_mode()` | Precompute and cache embedding contraction |
| `train_mode()` | Clear cache, enable gradient flow |
| `config()` | Return construction parameters |
| `spectral_norms()` | Per-core spectral norms |
| `reconstruction_error(matrix)` | Relative Frobenius error vs. dense |
| `distribution_aware_reconstruction_error(matrix)` | NeurIPS 2025 style: `||(W-Ŵ)Σ^{1/2}||_F` |
| `eigenspace_overlap_score(matrix, k=10)` | EOSk — top-k subspace preservation measure |
| `reconstruct()` | Full V×D matrix (debugging only) |
| `reset_parameters()` | Re-initialize cores |
| `from_compression_ratio(v, d, ratio)` | Create targeting compression |
| `from_target_params(v, d, params)` | Create targeting parameter count |
| `from_pretrained(matrix, rank)` | Initialize from pretrained weights |
| `from_huggingface(model_name, rank)` | Load HF model → TR embedding |
| `from_profile(profile, rank)` | Create from a ModelProfile |
| `export(path, format)` | Export to ONNX or TorchScript |
| `autotune(matrix, ...)` | Find optimal rank given constraints |
| `suggest_rank(model_name, ...)` | Look up recommended rank |

### Properties

- `compression_ratio`: Dense params / compressed params
- `num_parameters`: Total factor core parameters
- `rank`: TR rank
- `weight`: Reconstructed full matrix (V×D)
- `num_embeddings`: Vocabulary size

## Distributed Training

```python
from tensor_ring_decomposition import TensorRingDDP
from torch.nn.parallel import DistributedDataParallel

emb = TensorRingEmbedding(50000, 768, rank=8)
ddp_emb = TensorRingDDP(emb)
ddp_emb = DistributedDataParallel(ddp_emb, device_ids=[local_rank])
ddp_emb.to_eval_mode()  # cross-rank cache sync
```

## Quantization

```python
from tensor_ring_decomposition import QuantizedTensorRingEmbedding

q_emb = QuantizedTensorRingEmbedding(emb, per_channel=True)
output = q_emb(indices)  # int8 gather + float dequant
print(f"{q_emb.compression_ratio:.1f}x at {q_emb.bits_per_parameter:.1f} bpp")
```

## Model Registry

63+ built-in profiles across all major architectures:

```python
from tensor_ring_decomposition import list_models, ModelRegistry

print(list_models())                    # full list
print(list_models(family="llama"))      # filtered by family
profile = ModelRegistry.get("bert-base-uncased")
```

## Testing

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

## Benchmarks

```bash
python benchmarks/benchmark_tr_vs_dense.py
```

## License

MIT
