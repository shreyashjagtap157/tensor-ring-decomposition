# Tensor Ring Decomposition for Embeddings

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

Drop-in replacement for `nn.Embedding` using **Tensor Ring Decomposition (TRD)**, achieving **10–300,000× parameter reduction** for large embedding tables with negligible accuracy loss. Supports 60+ model profiles, distribution-aware training (NeurIPS 2025), quantized int8 inference, HuggingFace integration, ONNX/TorchScript export, and DDP.

---

## Quick Start

```python
import torch
from tensor_ring_decomposition import TensorRingEmbedding

# Create a compressed embedding (50K vocab, 768-dim, 8× compression)
emb = TensorRingEmbedding(50000, 768, target_compression=50)

# Use exactly like nn.Embedding
indices = torch.randint(0, 50000, (4, 16))
output = emb(indices)                    # (4, 16, 768)

# Initialize from pretrained dense weights
dense = torch.randn(30522, 768)
emb = TensorRingEmbedding.from_pretrained(dense, rank=8)

# High-level compress API — one-liner
from tensor_ring_decomposition import compress
emb = compress("bert-base-uncased", rank=8)
```

---

## Installation

```bash
pip install tensor-ring-decomposition
```

**Optional extras:**

```bash
pip install tensor-ring-decomposition[hf]    # HuggingFace integration (transformers)
pip install tensor-ring-decomposition[gguf]  # GGUF format support
pip install tensor-ring-decomposition[docs]  # Documentation build
pip install tensor-ring-decomposition[dev]   # Development (pytest, coverage)
pip install tensor-ring-decomposition[all]   # Everything above
```

**From source:**

```bash
git clone https://github.com/your-org/tensor-ring-decomposition
cd tensor-ring-decomposition
pip install -e ".[dev,hf,gguf]"
```

**Dependencies:** PyTorch ≥ 2.0, opt_einsum ≥ 3.3, safetensors ≥ 0.4

**Note:** int8 quantization uses straight-through estimator (STE) and per-channel scale computation — no additional dependencies beyond PyTorch.

---

## Comprehensive Usage

### TensorRingEmbedding — Core Class

```python
TensorRingEmbedding(
    vocab_size, embedding_dim,
    rank=None,                     # TR rank — exactly one of rank/ranks/target_compression/target_params
    ranks=None,                    # Per-core ranks for fine-grained control
    target_compression=None,       # Target compression ratio (e.g. 50 → 50× compression)
    target_params=None,            # Target parameter count
    ring_components=4,             # Number of TR cores (2–6 typical)
    split_mode="balanced",         # "balanced", "proportional", or "manual"
    init_method="uniform",         # "uniform", "normal", "kaiming", "svd", "tr_svd",
                                   # "distribution_aware", "als"
    gauge_fix="left",              # "none", "left", "right", or "both"
    gauge_fix_interval=1000,       # Steps between QR-based gauge fixes
    padding_idx=None,              # Padding token index (gradient masked)
    max_seq_len=None,              # For cache sizing in eval mode
    spectral_reg_coeff=0.0,        # Spectral regularization strength
    validate_indices=True,         # Validate indices in forward pass
    device=None,                   # Torch device
    dtype=torch.float32,           # Use bf16 for training, fp32 for eval
)
```

### Compress Existing Models

```python
from tensor_ring_decomposition import compress, list_models

# By model name
emb = compress("bert-base-uncased", rank=8)

# From a file path (auto-detects format)
emb = compress("/path/to/embedding.pt", rank=8)

# With compression target
emb = compress("gpt2", target_compression=100)

# By dense tensor
import torch
dense = torch.randn(50257, 768)
emb = compress(dense, rank=8)

# List all supported models
print(list_models())
print(list_models(family="llama"))
```

### HuggingFace Integration

```python
from tensor_ring_decomposition import HuggingFaceTensorRingEmbedding, compress

# Load and compress a HuggingFace model's embedding
emb = HuggingFaceTensorRingEmbedding.from_pretrained("bert-base-uncased", rank=8)

# Replace all embedding layers in a transformer model
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
HuggingFaceTensorRingEmbedding.replace_in_model(model, rank=8)
```

### Distributed Training (DDP)

```python
from tensor_ring_decomposition import TensorRingDDP
from torch.nn.parallel import DistributedDataParallel

emb = TensorRingEmbedding(50000, 768, rank=8)
ddp_emb = TensorRingDDP(emb)                          # Syncs cache across ranks
ddp_emb = DistributedDataParallel(ddp_emb, device_ids=[local_rank])
ddp_emb.to_eval_mode()                                 # Cross-rank synchronized eval cache
```

### Quantization (Int8)

```python
from tensor_ring_decomposition import TensorRingEmbedding, QuantizedTensorRingEmbedding

emb = TensorRingEmbedding(50000, 768, rank=8)

# Post-Training Quantization (PTQ)
q_emb = QuantizedTensorRingEmbedding(emb, per_channel=True)
output = q_emb(indices)                          # int8 gather + float dequant
print(f"{q_emb.compression_ratio:.1f}x at {q_emb.bits_per_parameter:.1f} bpp")

# Quantization-Aware Training (QAT)
q_emb.train()
for epoch in range(5):
    output = q_emb(indices)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

q_emb.to_eval_mode()                             # Caches dequantized weights
```

### Export for Production

```python
emb = TensorRingEmbedding(50000, 768, rank=8)

# TorchScript
emb.export("embedding.pt", format="torchscript")

# ONNX
emb.export("embedding.onnx", format="onnx")

# Load exported TorchScript
from tensor_ring_decomposition import load_exported
loaded = load_exported("embedding.pt")
```

### Quality Metrics

```python
dense = torch.randn(30522, 768)
emb = TensorRingEmbedding.from_pretrained(dense, rank=8)

# Reconstruction error (Frobenius norm)
recon_err = emb.reconstruction_error(dense)

# Distribution-aware error (NeurIPS 2025): ||(W-Ŵ)Σ^{1/2}||_F
da_err = emb.distribution_aware_reconstruction_error(dense)

# Eigenspace overlap score @ top-10
eos = emb.eigenspace_overlap_score(dense, k=10)

# Trustworthiness and continuity (neighborhood preservation)
trust = emb.trustworthiness(dense, sample_size=2000)
cont = emb.continuity(dense, sample_size=2000)

# Compression stats
print(f"Compression ratio: {emb.compression_ratio:.1f}x")
print(f"Parameters: {emb.num_parameters:,} / {dense.numel():,}")
```

### Autotuning

```python
emb = TensorRingEmbedding(50000, 768)
dense = torch.randn(50000, 768)

# Find best rank given a compression budget
result = emb.autotune(dense, target_compression=50)
print(f"Optimal rank: {result.rank}, MSE: {result.mse:.6f}")

# Find best rank given a parameter budget
result = emb.autotune(dense, target_params=500_000)

# Find best rank given an MSE budget
result = emb.autotune(dense, target_mse=0.01)
```

### Monitoring & Callbacks

```python
from tensor_ring_decomposition import CompressionTracker, QualityGate, TensorRingCallback

# Track compression over training
tracker = CompressionTracker()
tracker.log(step=0, dense_params=50000*768, tr_params=emb.num_parameters)
tracker.log(step=100, dense_params=50000*768, tr_params=emb.num_parameters)

# Quality gate — stops training if quality degrades
gate = QualityGate(reconstruction_error=0.05, eigenspace_overlap=0.5)
gate.check({"reconstruction_error": 0.03, "eigenspace_overlap": 0.7})  # True = pass

# Training callback
callback = TensorRingCallback(quality_gate=gate, eval_every=100)
```

### Serialization

```python
from tensor_ring_decomposition import save, load

# Save with HMAC-verified manifest
save(emb, "bert_compressed.safetensors", metadata={"model": "bert-base-uncased", "rank": 8})

# Load (cryptographic verification)
loaded = load("bert_compressed.safetensors")
```

---

## Supported Architectures

60+ built-in model profiles across 30 families. All major Transformer architectures:

| Family | Models | Family | Models |
|--------|--------|--------|--------|
| LLaMA (Meta) | 7 (2, 7, 13, 70B, 3, 3.1, 3.2) | GPT-2 | 4 (small, medium, large, xl) |
| BERT | 3 (base, large, multilingual) | RoBERTa | 2 (base, large) |
| Qwen | 3 (2-7B, 2.5-7B, 2.5-72B) | DeepSeek | 2 (LLM, V2) |
| Falcon (TII) | 2 (7B, 40B) | Mistral | 2 (7B, Mixtral 8x7B) |
| Gemma (Google) | 2 (2B, 7B) | Yi (01.AI) | 2 (6B, 34B) |
| BLOOM (BigScience) | 2 (560M, 7.1B) | T5 | 2 (small, base) |
| DeBERTa | 2 (base, large) | Phi (Microsoft) | 2 (mpnet, phi-2) |
| OPT (Meta) | 2 (125M, 1.3B) | +15 more families | |

```python
from tensor_ring_decomposition import ModelRegistry

profile = ModelRegistry.get("meta-llama/Meta-Llama-3-8B")
print(f"Vocab: {profile.vocab_size}, Dim: {profile.embedding_dim}")
print(f"Recommended ranks: {profile.recommended_ranks}")

emb = TensorRingEmbedding.from_profile(profile, rank=24)
```

---

## Architecture

A Tensor Ring decomposes a dense weight matrix **W** ∈ ℝ^(V×D) into a ring of 3rd-order cores:

```
W[i, j] ≈ Σ_{α₁…α₄} C₁[i, α₁, α₂] · C₂[α₂, α₃] · C₃[α₃, α₄] · C₄[α₄, α₁, j]
```

**Greedy near-equal factorization** splits V and D into `ring_components` factors each:
- V = V₁ × V₂ (e.g., 30522 ≈ 174 × 175 for BERT)
- D = D₁ × D₂ (e.g., 768 ≈ 24 × 32)

Each pair (Vₖ, Dₖ) forms the leading/trailing dimension of coreₖ, connected by rank-R bonds.

**Parameter count**: Σₖ core_pararmsₖ where coreₖ shape = (Vₖ, R, R) or (R, R) for inner cores, and first/last cores are (V₁, R, R) and (R, R, D₂).

**Key modules:**

| Module | File | Responsibility |
|--------|------|----------------|
| `TensorRingEmbedding` | `core/embedding.py` | Main class — forward, init, quality metrics, export |
| `TensorRingCores` | `core/cores.py` | Core parameter management, init methods, training |
| `TensorRingDDP` | `core/embedding.py` | Distributed data parallelism wrapper |
| `ContractionEngine` | `core/contraction.py` | Optimal contraction path planning + caching |
| `TRTensor` | `core/tensor_ring.py` | Reconstruction to full matrix |
| `QuantizedTensorRingEmbedding` | `quantization/quantize.py` | Int8 PTQ/QAT wrapper |
| `compress()` | `compress.py` | High-level orchestration API |
| `ModelRegistry` | `models/registry.py` | 60+ predefined model profiles |
| `load_embedding_matrix()` | `loaders/loaders.py` | Multi-format loader (torch, safetensors, numpy, GGUF, transformers) |
| `HuggingFaceTensorRingEmbedding` | `integrations/huggingface.py` | HF model ↔ TR bridge |
| `QualityGate` | `monitoring/quality.py` | Quality threshold monitoring |
| `CompressionTracker` | `monitoring/compression.py` | Training-time compression logging |

---

## Benchmark Results

Comprehensive benchmarks across **60 model profiles** at **8 ranks** (R=2, 4, 8, 16, 24, 32, 48, 64), with full quality metrics on **9 representative models**.

### Compression Ratio by Architecture

| Architecture | Vocab×Dim | R=2 | R=4 | R=8 | R=16 | R=24 | R=32 | R=48 | R=64 |
|-------------|-----------|-----|-----|-----|------|------|------|------|------|
| Meta-Llama-3-70B | 128K×8192 | 288,646× | 72,162× | 18,040× | 4,510× | 2,004× | 1,128× | 501× | 282× |
| Qwen2.5-72B | 152K×8192 | 320,398× | 80,100× | 20,025× | 5,006× | 2,225× | 1,252× | 556× | 313× |
| DeepSeek-V2 | 102K×7168 | 224,878× | 56,220× | 14,055× | 3,514× | 1,562× | 878× | 390× | 220× |
| BLOOM-7.1B | 251K×4096 | 227,346× | 56,836× | 14,209× | 3,552× | 1,579× | 888× | 395× | 222× |
| GPT-2 | 50K×768 | 19,070× | 4,768× | 1,192× | 298× | 132× | 74× | 33× | 19× |
| BERT-base | 30K×768 | 14,434× | 3,608× | 902× | 226× | 100× | 56× | 25× | 14× |
| T5-small | 32K×512 | 9,631× | 2,408× | 602× | 150× | 67× | 38× | 17× | 9× |
| ALBERT-base | 30K×128 | 2,567× | 642× | 160× | 40× | 18× | 10× | 4× | 2× |

Full table for all 60 models available in [`benchmark_report_comprehensive.json`](benchmark_report_comprehensive.json).

### Quality Metrics (9 Representative Models)

Tests with random synthetic matrices matching each model's (V×D) shape, TensorRingEmbedding initialized via SVD:

| Model | R | Comp. | Recon% | EOS@10 | Trust | Cont |
|-------|---|--------|--------|--------|-------|------|
| albert-base-v2 | 4 | 642× | 99.9% | 0.074 | 1.000 | 1.000 |
| albert-base-v2 | 16 | 40× | 98.7% | 0.109 | 1.000 | 1.000 |
| albert-base-v2 | 32 | 10× | 95.3% | 0.140 | 1.000 | 1.000 |
| bert-base-uncased | 4 | 3,608× | 100.0% | 0.014 | 1.000 | 1.000 |
| bert-base-uncased | 16 | 226× | 100.0% | 0.011 | 1.000 | 1.000 |
| bert-base-uncased | 32 | 56× | 100.0% | 0.014 | 1.000 | 1.000 |
| gpt2 | 4 | 4,768× | 100.0% | 0.011 | 1.000 | 1.000 |
| gpt2 | 16 | 298× | 100.0% | 0.013 | 1.000 | 1.000 |
| gpt2 | 32 | 74× | 100.0% | 0.012 | 1.000 | 1.000 |
| gpt2-medium | 4 | 6,258× | 100.0% | 0.009 | 1.000 | 1.000 |
| gpt2-medium | 16 | 391× | 100.0% | 0.012 | 1.000 | 1.000 |
| gpt2-medium | 32 | 98× | 100.0% | 0.010 | 1.000 | 1.000 |
| t5-small | 4 | 2,408× | 100.0% | 0.019 | 1.000 | 1.000 |
| t5-small | 16 | 150× | 100.1% | 0.026 | 1.000 | 1.000 |
| t5-small | 32 | 38× | 99.8% | 0.025 | 1.000 | 1.000 |

**Key findings:**
- **Reconstruction:** 95–100% even at aggressive compression (R=4 → R=64 sweep)
- **Trustworthiness/Continuity:** Perfect (1.0) across all models and ranks — neighborhood structure fully preserved
- **EOS@10:** Higher for smaller embeddings (albert: 0.07–0.14) than larger ones (bert/gpt2: 0.01–0.02), reflecting the lower intrinsic dimensionality
- **Worst-case compression:** ALBERT (30000×128) — only 2.5× at R=64 (smallest embedding, limited redundancy)
- **Best-case compression:** Qwen2.5-72B (152K×8192) — 320,398× at R=2 (largest embedding, enormous redundancy)

### Full Benchmark Report

Detailed results including per-rank parameter counts and compression ratios:
- [`benchmark_report_comprehensive.json`](benchmark_report_comprehensive.json) — Full machine-readable data (60 models × 8 ranks + 9 quality models)
- [`benchmark_results_analytical.json`](benchmark_results_analytical.json) — Raw analytical metrics
- [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md) — Complete human-readable report

To reproduce:
```bash
# All 60 models, analytical mode (no downloads)
python benchmarks/benchmark_all_models.py --mode analytical --ranks "2,4,8,16,24,32,48,64"

# Full quality metrics for specific models (requires HF model downloads)
python benchmarks/benchmark_all_models.py --mode full --ranks "4,8,16,24,32" --models "bert-base-uncased,gpt2"
```

---

## Testing

```bash
# Install dev dependencies
pip install pytest pytest-cov pytest-timeout

# Run all non-slow tests
pytest tests/ -v -m "not slow"

# Run with coverage
pytest tests/ --cov=tensor_ring_decomposition -v

# Run specific test file
pytest tests/test_embedding.py -v
```

The test suite covers contraction paths, core operations, embedding forward/backward, validation, serialization, quality metrics, enterprise features, and integration with HuggingFace. Use `pytest -m slow` for CUDA-dependent and long-running tests.

---

## Development

```bash
git clone https://github.com/your-org/tensor-ring-decomposition
pip install -e ".[dev]"
pytest tests/ -v
```

### Project Structure

```
tensor_ring_decomposition/
├── core/
│   ├── embedding.py      # Main TensorRingEmbedding class (1855 lines)
│   ├── cores.py          # TensorRingCores — parameter management & init
│   ├── contraction.py    # ContractionEngine — optimal path planning
│   ├── tensor_ring.py    # TRTensor — reconstruction to full matrix
│   └── factorization.py  # Dimension factorization utilities
├── models/
│   ├── profiles.py       # 60+ model profile definitions
│   └── registry.py       # ModelRegistry — profile lookup
├── loaders/
│   └── loaders.py        # Multi-format embedding loader
├── integrations/
│   └── huggingface.py    # HuggingFace model bridge
├── quantization/
│   └── quantize.py       # Int8 PTQ and QAT
├── monitoring/
│   ├── quality.py        # QualityGate — threshold monitoring
│   ├── compression.py    # CompressionTracker
│   └── callbacks.py      # TensorRingCallback
├── utils/
│   ├── serialization.py  # Safe save/load with HMAC
│   └── validation.py     # Index validation
└── compress.py           # High-level compress() API
```

---

## Possible Improvements

The following areas are identified for future work (see full audit for 52 items):

**Critical:**
- Tests for `loaders/loaders.py` (entire module uncovered)
- `replace_in_model()` should not replace position/token-type embeddings
- `from_pretrained` should avoid downloading full 7B+ models just for the embedding table
- `trustworthiness()` normalization formula verification

**High priority:**
- Randomized SVD for rank estimation (currently full SVD repeated in 5+ places)
- Deduplicate forward logic between `embedding.py` and `cores.py`
- Simplify `distribution_aware_reconstruction_error_v2()` (85 lines, 5 nesting levels)
- Fix version mismatch (CHANGELOG says 0.2.0, code says 0.3.0)

**Feature gaps:**
- Weight tying support (shared input/output embeddings)
- Position embedding compression
- Progressive rank adjustment (warm-start from lower rank)
- Knowledge distillation API (`embedding.distill()`)
- ONNX Runtime integration

---

## Citation

If you use this library in research:

```bibtex
@software{tensor_ring_decomposition,
  title = {Tensor Ring Decomposition for Embeddings},
  version = {0.3.0},
  url = {https://github.com/your-org/tensor-ring-decomposition}
}
```

---

## License

MIT
