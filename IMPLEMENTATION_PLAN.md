# Tensor Ring Decomposition for Embeddings — Implementation Plan

> **Status**: Complete — ready for implementation  
> **Date**: 2026-06-08  
> **Review rounds**: 8 independent professional reviews synthesized  
> **Target**: Drop-in replacement for `nn.Embedding` using Tensor Ring Decomposition

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Architecture Overview](#3-architecture-overview)
4. [API Contract](#4-api-contract)
5. [Algorithm & Mathematics](#5-algorithm--mathematics)
6. [File-by-File Implementation Details](#6-file-by-file-implementation-details)
7. [Testing Strategy](#7-testing-strategy)
8. [Examples & Benchmarks](#8-examples--benchmarks)
9. [Open Decisions](#9-open-decisions)
10. [Appendix: Review Round Synthesis](#10-appendix-review-round-synthesis)

---

## 1. Executive Summary

**Goal**: Compress embedding matrices using Tensor Ring Decomposition (TRD), achieving 10–100× parameter reduction over dense `nn.Embedding` with minimal accuracy loss.

**Core insight**: Never materialize the full V×D matrix. The embedding exists only in compressed ring form, queried via optimized tensor contractions. At inference time, embedding cores are precontracted into a single `(R, D, R)` tensor cached once, eliminating per-batch overhead.

**Why Tensor Ring over Tensor Train**:
- Tensor Train has open boundaries (r₁ = r_{N+1} = 1), limiting expressiveness for matrix inputs.
- Tensor Ring has cyclic closure (trace), which is invariant to dimension permutation and provides strictly greater representational capacity at the same rank.
- For embedding matrices specifically, the cyclic structure naturally captures the symmetric relationship between vocabulary indices and embedding dimensions.

**Key design decisions (from 8 reviews)**:
1. **Gradient-safe cache**: `to_eval_mode()` caches detached; `train_mode()` clears cache. Gradients flow through full contraction path during training.
2. **Gauge fixing**: Explicit orthogonalization of cores via QR decomposition to prevent scale drift during gradient descent.
3. **Safetensors serialization**: No `torch.load` anywhere — only `safetensors.load_file` + HMAC-verified JSON manifest.
4. **Exact ONE compression config**: `rank`, `ranks`, `target_compression`, `target_params` — exactly one must be set; ValueError otherwise.
5. **bf16-safe**: fp16 training produces error ~10⁻³ from long bmm chains; bf16 is preferred.
6. **Memory-efficient ring closure**: Avoid B·D·R·R intermediate via loop over R dimension.
7. **DDP safety**: `dist.barrier()` before cache + hash verification across ranks.
8. **Thread-safe contraction path cache**: Deterministic, pinned paths; thread-safe singleton.

---

## 2. Problem Statement

### 2.1 The Embedding Memory Problem

For modern LLMs and recommendation systems, the embedding table is the single largest parameter block:

| Model | V | D | Dense Params | Memory (FP32) |
|-------|---|---|--------------|---------------|
| BERT-base | 30,522 | 768 | 23.4M | 93.8 MB |
| GPT-2 | 50,257 | 768 | 38.6M | 154.4 MB |
| GPT-3 | 50,257 | 12,288 | 617.6M | 2.4 GB |
| RecSys (large) | 10,000,000 | 256 | 2.56B | 10.2 GB |

### 2.2 Why Not Matrix Factorization?

Standard matrix factorization (SVD) gives V×D ≈ V×K × K×D, reducing from V·D to V·K + K·D parameters. But:
- Linear factorization misses nonlinear structure.
- K must be large (≥32) for reasonable quality → limited compression.
- No natural way to exploit the fact that most vocabulary tokens are rarely used.

### 2.3 Why Tensor Ring?

Tensor Ring Decomposition factorizes a matrix into a ring of factor cores. For V×D with balanced ring:

```
V = V₀ × V₁ × ... × V_{k-1}    (e.g., 50K = 9 × 9 × 9 × 68)
D = D₀ × D₁ × ... × D_{m-1}    (e.g., 768 = 3 × 4 × 8 × 8)
Ring cores: k vocab cores + m emb cores = (k+m) cores total
Each core: (factor_size, R, R) with rank R
```

**Compression**: For V=50K, D=768, R=8, 4+4 cores:
- Dense: 50K × 768 = 38.4M params
- TR: 4 × (50K/4 × 64) + 4 × (768/4 × 64) ≈ 800K params → **~48× compression**

---

## 3. Architecture Overview

### 3.1 Ring Structure Diagram

```
Vocab Side (per-batch)                    Embedding Side (cached once)
─────────────────────                    ──────────────────────────

Token indices (B,)
        │
        ▼
  ┌─────────────┐
  │  V₀ (V₀,R,R₁) │ ←── Gather at indices
  └──────┬──────┘
         │ bmm
         ▼
  ┌─────────────┐
  │  V₁ (R₁,R,R₂) │ ←── Gather at indices
  └──────┬──────┘
         │ bmm
         ▼
  ┌─────────────┐
  │  V₂ (R₂,R,R₃) │ ←── Gather at indices
  └──────┬──────┘
         │ bmm
         ▼
  ┌─────────────┐
  │  V₃ (R₃,R,R₄) │ ←── Gather at indices
  └──────┬──────┘
         │
         ▼
    (B, R₄, R₄)     ──────► ┌──────────────┐
    Vocab chain result         │              │
                               │  Ring closure │
                               │  einsum trace │
    (R₀, D, R₄)     ──────► │              │
    Cached emb contraction     └──────┬──────┘
                                      │
                                      ▼
                               (B, D)
                               Output embeddings
```

The embedding side cores are precontracted once at `to_eval_mode()`:
```
E₀(R₀, D₀, R₁) , E₁(R₁, D₁, R₂) , E₂(R₂, D₂, R₃) , E₃(R₃, D₃, R₄)
→ opt_einsum contraction → (R₀, D, R₄) stored as buffer
```

### 3.2 Module Hierarchy

```
TensorRingEmbedding (nn.Module)
├── vocab_cores: nn.ParameterList    # Vocab-side factor cores
├── emb_cores: nn.ParameterList      # Embedding-side factor cores
├── _emb_cache: Tensor | None        # Cached (R, D, R) for eval
├── _vocab_expr: ContractExpression  # Precomputed opt_einsum path
├── _emb_expr: ContractExpression    # Precomputed opt_einsum path
└── _closure_expr: ContractExpression # Precomputed ring closure path
```

### 3.3 Data Flow — Training

```python
def forward(self, indices):
    # indices: (B,) or (B, seq_len) → flattened to (B*L,)
    flat = indices.view(-1)  # (N,) where N = B*L
    
    # 1. Vocab chain: gather + bmm
    vocab_result = self._vocab_chain(flat)  # (N, R, R)
    
    # 2. Emb contraction: from cache or recomputed
    if self.training or not self._cache_valid:
        emb_contraction = self._compute_emb_contraction()  # (R, D, R) — differentiable
    else:
        emb_contraction = self._emb_cache  # (R, D, R) — detached
    
    # 3. Ring closure
    output = self._ring_closure(vocab_result, emb_contraction)  # (N, D)
    
    return output.view(*indices.shape, -1)  # (B, seq_len, D)
```

### 3.4 Data Flow — Inference

```python
def eval_forward(self, indices):
    flat = indices.view(-1)
    vocab_result = self._vocab_chain(flat)           # (N, R, R)
    output = self._ring_closure(vocab_result, self._emb_cache)  # (N, D)
    return output.view(*indices.shape, -1)
```

**Key difference**: `_emb_cache` is a precomputed buffer, not recomputed per batch.

---

## 4. API Contract

### 4.1 Constructor

```python
class TensorRingEmbedding(nn.Module):
    """
    Tensor Ring Decomposition for Embedding compression.
    
    Drop-in replacement for nn.Embedding with far fewer parameters.
    Never materializes the full V×D matrix.
    
    Exactly ONE of: rank, ranks, target_compression, target_params must be set.
    
    Example — explicit rank:
        emb = TensorRingEmbedding(50000, 768, rank=8)
    
    Example — target compression ratio:
        emb = TensorRingEmbedding.from_compression_ratio(50000, 768, ratio=50)
    
    Example — from pretrained:
        emb = TensorRingEmbedding.from_pretrained(original_embedding.weight, rank=8)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        rank: Optional[int] = None,
        ranks: Optional[List[int]] = None,
        ring_components: int = 4,
        target_compression: Optional[float] = None,
        target_params: Optional[int] = None,
        split_mode: Literal["balanced", "proportional", "manual"] = "balanced",
        init_method: Literal["uniform", "normal", "kaiming", "svd"] = "uniform",
        gauge_fix: Literal["none", "left", "right", "both"] = "left",
        gauge_fix_interval: int = 1000,
        padding_idx: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
```

**Validation rules** (enforced in `__init__`):

| Condition | Result |
|-----------|--------|
| More than one of rank/ranks/target_compression/target_params is set | `ValueError("Exactly one of rank, ranks, target_compression, target_params must be set")` |
| None of rank/ranks/target_compression/target_params is set | `ValueError("Must specify one of rank, ranks, target_compression, or target_params")` |
| `ranks` is not None and `len(ranks) != ring_components` | `ValueError(f"ranks length ({len(ranks)}) must equal ring_components ({ring_components})")` |
| `split_mode="manual"` but `ranks` is None | `ValueError("split_mode='manual' requires explicit ranks")` |
| `target_compression` is set and <= 1.0 | `ValueError("target_compression must be > 1.0 (ratio of dense/compressed params)")` |
| `init_method="svd"` but no `embedding_matrix` passed | `ValueError("svd init requires embedding_matrix parameter; use from_pretrained()")` |
| `vocab_size` or `embedding_dim` <= 0 | `ValueError("vocab_size and embedding_dim must be positive")` |

### 4.2 Class Methods

```python
@classmethod
def from_compression_ratio(
    cls,
    vocab_size: int,
    embedding_dim: int,
    ratio: float,
    ring_components: int = 4,
    **kwargs,
) -> "TensorRingEmbedding":
    """Create TR embedding targeting a specific compression ratio.
    
    Args:
        ratio: Target dense_params / compressed_params. E.g., 50 means 50× compression.
    """
    target_params = (vocab_size * embedding_dim) / ratio
    return cls(vocab_size, embedding_dim, target_params=target_params,
               ring_components=ring_components, **kwargs)

@classmethod
def from_target_params(
    cls,
    vocab_size: int,
    embedding_dim: int,
    params: int,
    ring_components: int = 4,
    **kwargs,
) -> "TensorRingEmbedding":
    """Create TR embedding targeting a specific parameter count."""
    return cls(vocab_size, embedding_dim, target_params=params,
               ring_components=ring_components, **kwargs)

@classmethod
def from_pretrained(
    cls,
    embedding_matrix: torch.Tensor,
    rank: int,
    ring_components: int = 4,
    **kwargs,
) -> "TensorRingEmbedding":
    """Create TR embedding initialized from pretrained dense embedding.
    
    Args:
        embedding_matrix: (V, D) tensor of pretrained embeddings
        rank: TR rank for all cores
    """
    V, D = embedding_matrix.shape
    emb = cls(V, D, rank=rank, ring_components=ring_components,
              init_method="svd", **kwargs)
    emb._init_from_pretrained(embedding_matrix)
    return emb

@classmethod
def from_huggingface(
    cls,
    model_name: str,
    rank: int,
    ring_components: int = 4,
    **kwargs,
) -> "TensorRingEmbedding":
    """Load HuggingFace model, extract input embeddings, decompose via TR."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_name)
    embedding = model.get_input_embeddings()
    return cls.from_pretrained(embedding.weight.data, rank, ring_components, **kwargs)
```

### 4.3 Instance Methods

```python
def forward(self, indices: torch.Tensor) -> torch.Tensor:
    """Compressed embedding lookup.
    
    Args:
        indices: Token IDs. Shape: (B,) or (B, seq_len)
    
    Returns:
        Compressed embeddings. Shape: (B, D) or (B, seq_len, D)
    
    Raises:
        IndexError: If any index >= vocab_size or < 0 (unless padding_idx set)
    """

def to_eval_mode(self) -> "TensorRingEmbedding":
    """Switch to inference mode.
    
    Precomputes and caches embedding cores contraction as (R, D, R) buffer.
    Must be called after DDP broadcast ensures all ranks have identical cores.
    
    Returns:
        self, for chaining
    """

def train_mode(self) -> "TensorRingEmbedding":
    """Switch to training mode.
    
    Clears cached contraction. Gradients flow through full contraction path.
    
    Returns:
        self, for chaining
    """

def config(self) -> dict:
    """Return immutable construction parameters for debugging/introspection.
    
    Returns:
        dict with keys: vocab_size, embedding_dim, rank, ranks, ring_components,
        split_mode, init_method, gauge_fix, padding_idx, compression_ratio,
        num_parameters, init_method_used
    """

def spectral_norms(self) -> Dict[str, float]:
    """Return spectral norm of each factor core (for monitoring gauge stability).
    
    Returns:
        dict mapping core name to spectral norm. E.g.:
        {"vocab_0": 1.002, "vocab_1": 0.998, "emb_0": 1.001, ...}
    """

def reconstruction_error(self, original_matrix: torch.Tensor) -> float:
    """Compute ||W - W_recon||_F / ||W||_F (relative Frobenius norm).
    
    WARNING: This reconstructs the full matrix. Use only for debugging.
    """

def reconstruct(self) -> torch.Tensor:
    """Reconstruct full V×D embedding matrix from ring factors.
    
    WARNING: Materializes full matrix. Never call during training or inference.
    """

@property
def compression_ratio(self) -> float:
    """Dense params / compressed params."""

@property
def num_parameters(self) -> int:
    """Total parameters in ring factor cores."""
```

### 4.4 Compatibility with `nn.Embedding`

| Feature | `nn.Embedding` | `TensorRingEmbedding` | Notes |
|---------|---------------|----------------------|-------|
| Constructor `(vocab_size, embedding_dim)` | ✅ | ✅ | Default args |
| `padding_idx` | ✅ | ✅ | Identical behavior |
| `max_norm`, `norm_type`, `scale_grad_by_freq` | ✅ | ❌ | Not applicable for TR |
| `freeze()` / `requires_grad` | ✅ | ✅ | Per-core freezing |
| `forward(indices)` | ✅ | ✅ | Same signature |
| `weight` property (full matrix) | ✅ | ❌ | TR cores, not a single weight |
| `sparse` output | ✅ | ❌ | Dense only |
| `.to(device)` | ✅ | ✅ | All cores moved |
| `.half()` / `.bfloat16()` | ✅ | ✅ | bf16 preferred for training |

---

## 5. Algorithm & Mathematics

### 5.1 Tensor Ring Decomposition

**Definition** (Zhao et al., 2016): A tensor ring decomposition of a d-dimensional tensor X of shape (n₁, n₂, ..., n_d) is a set of factor cores {Gₖ} where each Gₖ has shape (R, nₖ, R) and the original tensor is recovered via:

```
X(i₁, ..., i_d) = Tr(G₁(i₁) · G₂(i₂) · ... · G_d(i_d))
```

where Gₖ(iₖ) is the (R × R) slice of core Gₖ at index iₖ, and Tr denotes matrix trace.

**For 2D embedding matrix W of shape (V, D)**:

We reshape both V and D into factor chains:
```
V = V₀ × V₁ × ... × V_{k-1}
D = D₀ × D₁ × ... × D_{m-1}
```

Then W is viewed as a (k+m)-dimensional tensor:
```
W(i₀, ..., i_{k-1}, j₀, ..., j_{m-1}) = W[v, d]
where v = i₀·V₁·...·V_{k-1} + ... + i_{k-1} (mixed-radix encoding)
      d = j₀·D₁·...·D_{m-1} + ... + j_{m-1}
```

The ring has k+m cores:
- **Vocab cores** Vₖ: shape (Vₖ, R, R) — indexed by vocabulary decomposition
- **Embedding cores** Eₘ: shape (Dₘ, R, R) — indexed by embedding decomposition

**Ring contraction** (the full forward pass):
```
W[v, d] = Σ_{r₀,...,r_{k+m-1}} V₀[v₀, r₀, r₁] · V₁[v₁, r₁, r₂] · ... · V_{k-1}[v_{k-1}, r_{k-1}, r_k]
                                    · E₀[j₀, r_k, r_{k+1}] · E₁[j₁, r_{k+1}, r_{k+2}] · ... · E_{m-1}[j_{m-1}, r_{k+m-1}, r₀]
```

where r_{k+m} = r₀ (ring closure via trace).

### 5.2 Forward Pass Decomposition

The key insight: we never compute the full (V, D) matrix. Instead, we decompose into two chains:

**Step 1: Vocab chain** (computed per-batch):
```
Input: token indices (B,) where each element ∈ {0, ..., V-1}
For each token b:
    For each vocab core k:
        vₖ[b] = Vₖ[tokenₖ[b]]  →  (R, R) slice
    Chain product: V₀[v₀] · V₁[v₁] · ... · V_{k-1}[v_{k-1}]  →  (R, R) matrix per token
Result: (B, R, R) tensor
```

**Step 2: Embedding precontraction** (computed once, cached):
```
Precontract all embedding cores:
    E₀ · E₁ · ... · E_{m-1}  →  (R, D, R) tensor
    (This is done via opt_einsum with optimal path)
```

**Step 3: Ring closure** (computed per-batch, fixed cost):
```
For each token b:
    result[b] = Tr(vocab_result[b] · emb_contraction)
    = Σ_r,vocab_result[b, r, r'] · emb_contraction[r', d, r]
```

This is implemented as `einsum('bri,rdi->bdri')` followed by `einsum('bdri->bd')`.

### 5.3 Optimal Contraction Paths

**Vocab chain** (k cores):
```
Contraction: V₀[b,r₀,r₁] · V₁[b,r₁,r₂] · ... · V_{k-1}[b,r_{k-1},r_k] → (B, R, R)
opt_einsum eq: "bri,rj,sk->bsk" (if k=3, for example)
```

**Embedding precontraction** (m cores):
```
Contraction: E₀[r₀,d₀,r₁] · E₁[r₁,d₁,r₂] · ... · E_{m-1}[r_{m-1},d_{m-1},r₀] → (R, D, R)
opt_einsum eq: varies with number of cores
```

**Ring closure** (2 operands):
```
Contraction: (B, R, R) ⊗ (R, D, R) → (B, D)
opt_einsum eq: "bri,rdi->bd"
```

**Memory consideration**: The direct einsum `bri,rdi->bdri` creates intermediate (B, D, R, R) which can be large. For B=256, D=768, R=32: 256 × 768 × 32 × 32 = 201M elements ≈ 800MB in fp32.

**Solution**: Loop over R dimension:
```python
def ring_closure_mem_efficient(vocab_result, emb_contraction):
    """
    Memory-efficient ring closure via R-dimension loop.
    
    vocab_result: (B, R, R)
    emb_contraction: (R, D, R)
    Returns: (B, D)
    """
    R = vocab_result.shape[1]
    D = emb_contraction.shape[1]
    B = vocab_result.shape[0]
    output = torch.zeros(B, D, device=vocab_result.device, dtype=vocab_result.dtype)
    
    for r in range(R):
        # vocab_result[:, r, :] → (B, R)
        # emb_contraction[:, :, r] → (R, D)
        output += torch.mm(vocab_result[:, r, :], emb_contraction[:, :, r])
    
    return output
```

This avoids the O(B·D·R²) intermediate at cost of R matmuls of size (B×R) @ (R×D), which is R × O(B·R·D) total. For R=32, this is ~2× more compute but O(R) memory instead of O(R²).

**Alternative**: Let opt_einsum choose:
```python
path, _ = oe.contract_path('bri,rdi->bd', vocab_result, emb_contraction, optimize='greedy')
```

opt_einsum may find a path that avoids the large intermediate.

### 5.4 SVD-Based Initialization

**Algorithm** (recursive SVD from TensorLy, adapted for TR):

Given a pretrained embedding matrix W of shape (V, D) and target rank R:

1. **Compute target parameter count**:
   ```
   target_params = (V × D) / target_compression
   ```
   Solve for ring structure: k vocab cores + m emb cores = k+m total
   Each core has V/Rank/Rank or D/Rank/Rank params

2. **Reshape W** into tensor with dimensions matching ring structure:
   ```
   W → (V₀, V₁, ..., V_{k-1}, D₀, D₁, ..., D_{m-1})
   ```

3. **Recursive SVD** (mode by mode):
   ```
   Unfold along first mode → SVD → extract U₁ → residual
   Unfold residual along second mode → SVD → extract U₂ → residual
   ... continue for all modes
   Last mode: remaining tensor = last core (with ring closure)
   ```

4. **Reshape** each extracted Uₖ to core shape:
   ```
   Uₖ: (Vₖ × R) → reshape to (Vₖ, R, R_next)
   ```

5. **Ring closure**: Last core must close the ring:
   ```
   Last core shape: (D_{m-1}, R, R) — trace over both R dimensions
   ```

**Fallback chain** (for OOM prevention):
```
try: full SVD (requires V×D matrix in memory)
except OOM: streaming SVD (sketch-based, uses O(R × max(V,D)) memory)
except: Xavier init with warning
```

**Singularity value decay**:
- If σ_k/σ₁ > 0.1 for many k → TR with small R will have poor quality.
- Log singular value spectrum at init time for diagnostics.

### 5.5 Automatic Rank Selection

Given `target_compression` or `target_params`, solve for R:

```python
def _solve_rank(vocab_size, embedding_dim, ring_components, target_compression):
    dense_params = vocab_size * embedding_dim
    target_params = dense_params / target_compression
    
    # Parameter count formula for TR with balanced ranks:
    # vocab params = k × (V/k) × R² = V × R²
    # emb params = m × (D/m) × R² = D × R²
    # total = R² × (V + D)
    # Solve: R² × (V + D) = target_params
    # R = sqrt(target_params / (V + D))
    
    R = math.sqrt(target_params / (vocab_size + embedding_dim))
    R = max(2, int(round(R)))  # Minimum rank of 2
    
    return R
```

For heterogeneous ranks, use per-mode singular value analysis:
```python
def _solve_heterogeneous_ranks(matrix, ring_components, target_params):
    # Unfold matrix along V-mode and D-mode
    # Compute singular value spectra for each unfolding
    # Choose ranks where singular values drop below threshold
    # Ensure total params ≈ target_params
    ...
```

### 5.6 Gauge Fixing

Tensor Ring decomposition has **gauge freedom**: multiplying core Gₖ by α and Gₖ₊₁ by 1/α leaves the contraction unchanged. This causes:
- Scale drift across cores during training
- Wasted optimization budget (gradient steps change scale without changing output)

**Fix**: QR-based orthogonalization of cores.

**Left gauge fix** (fix first core):
```python
def fix_left_gauge(cores):
    """Orthogonalize cores from left to right."""
    for i in range(len(cores)):
        # Reshape core to (V_i × R, R_next)
        old_shape = cores[i].shape
        flat = cores[i].data.reshape(-1, old_shape[2])
        
        Q, R = torch.linalg.qr(flat)
        
        # Fix sign ambiguity
        sign = torch.sign(torch.diag(R))
        Q = Q * sign
        R = torch.diag(sign) @ R
        
        # Store orthogonalized core
        cores[i].data = Q.reshape(old_shape)
        
        # Propagate R to next core
        if i < len(cores) - 1:
            cores[i+1].data = torch.bmm(
                R.unsqueeze(0).expand(cores[i+1].shape[0], -1, -1),
                cores[i+1].data
            )
```

**Monitoring**: After gauge fix, check spectral norms:
```python
spectral_norms = {name: torch.linalg.svd(core.data.reshape(-1, core.shape[-1]))[1][0].item()
                  for name, core in named_cores}
# All norms should be ≈ 1.0; if > 1.5 or < 0.5, warn about instability
```

**Interval**: Apply gauge fix every `gauge_fix_interval` steps (default: 1000).

### 5.7 Numerical Precision

**Floating point error in vocab chain**:
- 8 cores, R=32: chain of 7 bmm operations.
- Each bmm: O(R³) flops, error O(ε·κ) where κ is condition number.
- Total error: O(7·ε·κ) ≈ 7 × 2⁻²³ × 100 ≈ 10⁻¹⁸ in fp32 (negligible).
- **fp16**: error ~10⁻⁴–10⁻³, can destabilize gradients. **Use bf16 or fp32 only**.

**Mixed precision training rules**:
```
ALLOWED: bf16 (sufficient dynamic range)
ALLOWED: fp32 (full precision)
NOT ALLOWED: fp16 (error accumulation too high for long bmm chains)
```

**Enforcement**:
```python
def forward(self, indices):
    if self.dtype == torch.float16:
        raise TypeError(
            "fp16 training not supported due to numerical instability in long bmm chains. "
            "Use bf16 or fp32 instead."
        )
```

---

## 6. File-by-File Implementation Details

### 6.1 `tensor_ring_decomposition/__init__.py`

```python
"""Tensor Ring Decomposition for Embeddings."""
from .core.embedding import TensorRingEmbedding
from .core.factorization import factorize_dimension, compute_ring_structure
from .core.cores import TensorRingCores
from .utils.serialization import save, load

__all__ = [
    "TensorRingEmbedding",
    "factorize_dimension",
    "compute_ring_structure",
    "TensorRingCores",
    "save",
    "load",
]

__version__ = "0.1.0"
```

### 6.2 `tensor_ring_decomposition/core/__init__.py`

```python
from .factorization import factorize_dimension, compute_ring_structure
from .cores import TensorRingCores
from .contraction import (
    compute_vocab_chain_path,
    compute_emb_precontraction_path,
    ring_closure,
    ContractionPathCache,
)
from .tensor_ring import TRTensor
from .embedding import TensorRingEmbedding
```

### 6.3 `tensor_ring_decomposition/core/factorization.py`

**Purpose**: Split integer dimensions into factor chains.

**Functions**:

```python
def factorize_dimension(dim: int, n_factors: int) -> List[int]:
    """Split dim into n_factors parts whose product = dim.
    
    Strategy: Greedy near-equal product.
    Start with each factor ≈ dim^(1/n_factors), adjust to ensure exact product.
    
    Examples:
        factorize_dimension(50000, 4) → [9, 9, 9, 68]
        factorize_dimension(768, 4) → [3, 4, 8, 8]
        factorize_dimension(100, 2) → [10, 10]
    
    Raises:
        ValueError: If dim < n_factors (cannot factorize)
    """
    if dim < n_factors:
        raise ValueError(f"Cannot factor {dim} into {n_factors} factors (each must be ≥ 1)")
    
    base = dim ** (1.0 / n_factors)
    factors = []
    remaining = dim
    
    for i in range(n_factors - 1):
        # Each factor should be close to remaining^(1/(factors_left))
        factors_left = n_factors - i
        size = max(1, int(round(remaining ** (1.0 / factors_left))))
        size = min(size, remaining)  # don't overshoot
        factors.append(size)
        remaining //= size
    
    factors.append(remaining)
    return factors


@dataclass
class RingStructure:
    """Complete specification of a tensor ring decomposition."""
    vocab_factor_sizes: List[int]     # [V₀, V₁, ..., V_{k-1}]
    emb_factor_sizes: List[int]       # [D₀, D₁, ..., D_{m-1}]
    ranks: List[int]                  # [R₀, R₁, ..., R_{k+m}] with R₀ = R_{k+m}
    rank: int                         # Common rank (if balanced)
    ring_components: int              # k + m total cores
    n_vocab_cores: int                # k
    n_emb_cores: int                  # m


def compute_ring_structure(
    vocab_size: int,
    embedding_dim: int,
    ring_components: int = 4,
    rank: int = 8,
    split_mode: str = "balanced",
    ranks: Optional[List[int]] = None,
) -> RingStructure:
    """Compute the complete ring structure.
    
    Args:
        vocab_size: V
        embedding_dim: D
        ring_components: Total cores k+m
        rank: Rank R (used if ranks is None)
        split_mode: "balanced", "proportional", or "manual"
        ranks: Explicit ranks (used with split_mode="manual")
    
    Returns:
        RingStructure with all factor sizes and ranks
    """
    if split_mode == "balanced":
        k = ring_components // 2
        m = ring_components - k
    elif split_mode == "proportional":
        total = vocab_size + embedding_dim
        k = max(1, int(round(ring_components * vocab_size / total)))
        m = ring_components - k
    elif split_mode == "manual":
        if ranks is None:
            raise ValueError("split_mode='manual' requires explicit ranks")
        # Parse ranks to determine k and m
        # ... implementation
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")
    
    vocab_factors = factorize_dimension(vocab_size, k)
    emb_factors = factorize_dimension(embedding_dim, m)
    
    if ranks is None:
        ranks = [rank] * (k + m + 1)  # +1 for ring closure rank
    
    return RingStructure(
        vocab_factor_sizes=vocab_factors,
        emb_factor_sizes=emb_factors,
        ranks=ranks,
        rank=rank,
        ring_components=ring_components,
        n_vocab_cores=k,
        n_emb_cores=m,
    )
```

### 6.4 `tensor_ring_decomposition/core/cores.py`

**Purpose**: Factor core initialization and management.

```python
class TensorRingCores(nn.Module):
    """Manages vocab and embedding factor cores for tensor ring."""
    
    def __init__(
        self,
        ring_structure: RingStructure,
        init_method: str = "uniform",
        gauge_fix: str = "left",
        gauge_fix_interval: int = 1000,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
    ):
        super().__init__()
        self.structure = ring_structure
        self.gauge_fix = gauge_fix
        self.gauge_fix_interval = gauge_fix_interval
        self.dtype = dtype
        
        k = ring_structure.n_vocab_cores
        m = ring_structure.n_emb_cores
        ranks = ring_structure.ranks
        
        # Create vocab cores: each (V_i, R_i, R_{i+1})
        self.vocab_cores = nn.ParameterList([
            nn.Parameter(torch.empty(
                ring_structure.vocab_factor_sizes[i],
                ranks[i],
                ranks[i + 1],
                dtype=dtype,
                device=device,
            ))
            for i in range(k)
        ])
        
        # Create emb cores: each (D_i, R_{k+i}, R_{k+i+1})
        self.emb_cores = nn.ParameterList([
            nn.Parameter(torch.empty(
                ring_structure.emb_factor_sizes[i],
                ranks[k + i],
                ranks[k + i + 1],
                dtype=dtype,
                device=device,
            ))
            for i in range(m)
        ])
        
        self._step = 0
    
    def initialize(self, init_method: str, embedding_matrix: torch.Tensor = None):
        """Initialize cores using specified method.
        
        Methods:
            uniform: Xavier uniform (default for training from scratch)
            normal: Xavier normal
            kaiming: Kaiming uniform
            svd: SVD decomposition from pretrained (requires embedding_matrix)
        """
        if init_method == "svd":
            if embedding_matrix is None:
                raise ValueError("SVD init requires embedding_matrix")
            self._init_svd(embedding_matrix)
        elif init_method == "uniform":
            self._init_xavier("uniform")
        elif init_method == "normal":
            self._init_xavier("normal")
        elif init_method == "kaiming":
            self._init_kaiming()
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
    
    def _init_svd(self, matrix: torch.Tensor):
        """SVD-based initialization with OOM fallback."""
        # ... see Section 5.4 for algorithm
    
    def _init_xavier(self, mode: str = "uniform"):
        """Xavier initialization for all cores."""
        for core in self.vocab_cores + self.emb_cores:
            nn.init.xavier_uniform_(core.data) if mode == "uniform" else nn.init.xavier_normal_(core.data)
    
    def _apply_gauge_fix(self):
        """Apply gauge fixing at configured interval."""
        if self.gauge_fix == "none":
            return
        
        self._step += 1
        if self._step % self.gauge_fix_interval != 0:
            return
        
        if self.gauge_fix in ("left", "both"):
            GaugeFixer.fix_left(self.vocab_cores)
            GaugeFixer.fix_left(self.emb_cores)
        
        if self.gauge_fix in ("right", "both"):
            GaugeFixer.fix_right(self.vocab_cores)
            GaugeFixer.fix_right(self.emb_cores)
    
    def spectral_norms(self) -> Dict[str, float]:
        """Compute spectral norm of each core."""
        norms = {}
        for i, core in enumerate(self.vocab_cores):
            flat = core.data.reshape(-1, core.shape[-1])
            norms[f"vocab_{i}"] = torch.linalg.svd(flat)[1][0].item()
        for i, core in enumerate(self.emb_cores):
            flat = core.data.reshape(-1, core.shape[-1])
            norms[f"emb_{i}"] = torch.linalg.svd(flat)[1][0].item()
        return norms
    
    def parameter_count(self) -> int:
        """Total parameters across all cores."""
        return sum(p.numel() for p in self.parameters())
    
    def dense_parameter_count(self) -> int:
        """Equivalent dense parameter count."""
        return self.structure.vocab_factor_sizes[0] * self.structure.emb_factor_sizes[0]
```

### 6.5 `tensor_ring_decomposition/core/contraction.py`

**Purpose**: Contraction path computation and execution using opt_einsum.

```python
import opt_einsum as oe
import threading
from typing import Tuple, List


class ContractionPathCache:
    """Thread-safe, deterministic cache for contraction paths.
    
    Once computed for given shapes, path is pinned and never recomputed.
    This ensures:
    1. No re-computation overhead
    2. Thread-safety for parallel workers
    3. Reproducibility across runs
    """
    
    _cache: Dict[Tuple[str, Tuple[Tuple[int, ...], ...]], Tuple[List[Tuple[int, int]], str]] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_path(
        cls,
        eq: str,
        shapes: List[Tuple[int, ...]],
        optimize: str = "greedy",
    ) -> Tuple[List[Tuple[int, int]], str]:
        """Get cached contraction path or compute and cache it.
        
        Args:
            eq: Einstein summation equation
            shapes: List of operand shapes
            optimize: Path optimization strategy ("greedy", "optimal", etc.)
        
        Returns:
            (path, path_info_str) tuple
        """
        key = (eq, tuple(tuple(s) for s in shapes))
        
        with cls._lock:
            if key not in cls._cache:
                path, path_info = oe.contract_path(
                    eq, *shapes, optimize=optimize
                )
                cls._cache[key] = (path, str(path_info))
        
        return cls._cache[key]
    
    @classmethod
    def clear(cls):
        """Clear all cached paths."""
        with cls._lock:
            cls._cache.clear()


def compute_vocab_chain_expression(
    vocab_core_shapes: List[Tuple[int, int, int]],
    rank: int,
) -> oe.contract_expression:
    """Precompute contraction path for vocab cores.
    
    Vocab core i has shape (V_i, R, R_{i+1}).
    After gathering at token indices: (B, R, R_{i+1}).
    Chain: (B, R₀, R₁) @ (B, R₁, R₂) @ ... @ (B, R_{k-1}, R_k) → (B, R₀, R_k)
    
    For ring closure: R_k = R₀ = R.
    
    Returns:
        ContractExpression that can be called with gathered cores.
    """
    k = len(vocab_core_shapes)
    
    if k == 1:
        eq = "bri->bri"  # Identity
        shapes = [vocab_core_shapes[0]]  # Will be gathered to (B, R, R)
    elif k == 2:
        eq = "bri,bsj->bsj"  # First core gathered, second gathered
        # Actually need to trace r:
        eq = "bri,brj->bsj"  # Not quite right either
        # Correct: contract over shared rank dimension
        eq = "bri,rj->bsj"  # Hmm, this is getting complex
    
    # Better approach: build the contraction string dynamically
    # For k vocab cores: V₀[b,r₀,r₁] · V₁[b,r₁,r₂] · ... · V_{k-1}[b,r_{k-1},r_k]
    # With ring: r_k = r₀
    # eq = "b,r,r->b,r,r" chain with k operands
    
    # Actually the simplest: compute it at runtime for each forward pass
    # Precompute only the path
    
    # Build equation string
    # Each vocab core: (B, R, R_next)
    # Chain: result += core[b, R, R_next]
    
    # For simplicity, use explicit contraction:
    # Start: (B, R₀, R₁)
    # After V₁: (B, R₀, R₂)
    # After V₂: (B, R₀, R₃)
    # ... After V_{k-1}: (B, R₀, R_k)
    
    # The equation is:
    # "bri,rj,sk,tl,...->b..." where each subsequent operand shares the rank index
    
    # For generic k:
    letters = ['b'] + [chr(ord('r') + i) for i in range(k + 1)]
    operands = []
    for i in range(k):
        # Core i: (B, R_i, R_{i+1})
        operands.append(f"b{letters[i+1]}{letters[i+2]}")
    eq = ','.join(operands) + '->b' + letters[0] + letters[k]
    
    # Wait, this isn't right for bmm chain. Let me reconsider.
    
    # The bmm chain is sequential:
    # result = V₀[b]  # (B, R, R)
    # for core in V₁..V_{k-1}:
    #     result = bmm(result, core[b])  # (B, R, R)
    # 
    # For opt_einsum, the full chain equation is:
    # "bri,rj,sk->bsk" (for k=3, contracting r₁ and r₂)
    
    # Actually, for the chain:
    # V₀[b, r₀, r₁] · V₁[b, r₁, r₂] · ... · V_{k-1}[b, r_{k-1}, r₀]
    # This is NOT what we want - the last core should output r₀ for ring closure
    
    # Let me use a cleaner formulation. For each pair of adjacent cores,
    # we contract over the shared rank index.
    
    # For k=3:
    # V₀[b, r₀, r₁] · V₁[b, r₁, r₂] · V₂[b, r₂, r₀]
    # This contracts r₁, r₂, and r₀ (trace) → (B,)
    # But we want (B, R, R) for ring closure!
    
    # The issue: we need to keep two rank indices open for ring closure.
    # Let's keep r₀ and r_{k} (which = r₀ for ring):
    
    # For k=3 with ring:
    # V₀[b, r₀, r₁] · V₁[b, r₁, r₂] · V₂[b, r₂, r₃]
    # where r₃ = r₀ (ring)
    # We don't trace over r₀ yet - that's done in ring closure
    
    # So the chain equation for k=3:
    # "br0r1,br1r2,br2r0->br0r0"  (trace would give (B,))
    # But we want (B, R, R) = (B, r₀, r₃) where r₃=r₀
    
    # Hmm, this means the chain result IS (B, R₀, R_k) and we want R_k = R₀ = R
    # Then ring closure traces over both R dimensions with emb contraction
    
    # For the CHAIN alone (no ring closure yet):
    # V₀[b, r₀, r₁] · V₁[b, r₁, r₂] · ... · V_{k-1}[b, r_{k-1}, r_k]
    # Result: (B, r₀, r_k)
    
    # The equation for this chain:
    operands = []
    rank_vars = []
    for i in range(k):
        r_in = f"r{i}"
        r_out = f"r{i+1}"
        operands.append(f"b{r_in}{r_out}")
        rank_vars.append((r_in, r_out))
    
    eq = ','.join(operands) + f'->br0r{k}'
    
    # Shapes: each operand is (B, R, R) (after gathering at indices)
    # But B is dynamic, so we use (B, R, R) shapes
    shapes = [(None, R, R) for R in [vocab_core_shapes[i][1] for i in range(k)]]
    # Actually shapes should be the core shapes without B:
    # vocab_core_shapes[i] = (V_i, R, R_next)
    # After gathering: (B, R, R_next) — B is batch size
    
    # For path computation, we need fixed shapes. Use (1, R, R) as placeholder for B.
    shapes = [(1, s[1], s[2]) for s in vocab_core_shapes]
    
    return oe.contract_expression(eq, *shapes)


def compute_emb_precontraction_expression(
    emb_core_shapes: List[Tuple[int, int, int]],
) -> oe.contract_expression:
    """Precompute contraction path for embedding cores.
    
    Emb core i has shape (D_i, R_i, R_{i+1}).
    Full contraction: E₀[d₀,r₀,r₁] · E₁[d₁,r₁,r₂] · ... · E_{m-1}[d_{m-1},r_{m-1},r₀]
    Result: (R₀, D, R₀) where D = D₀ × D₁ × ... × D_{m-1}
    
    Returns:
        ContractExpression returning (R, D, R) tensor.
    """
    m = len(emb_core_shapes)
    
    # Build equation
    # E₀[d₀, r₀, r₁] · E₁[d₁, r₁, r₂] · ... · E_{m-1}[d_{m-1}, r_{m-1}, r₀]
    operands = []
    for i in range(m):
        d_var = f"d{i}"
        r_in = f"r{i}"
        r_out = f"r{i+1}"
        operands.append(f"{d_var}{r_in}{r_out}")
    
    # Trace over r₀ (ring closure):
    eq = ','.join(operands) + f'->r0' + ''.join([f'd{i}' for i in range(m)])
    # Wait, this isn't right. The result should be (R₀, D₀×...×D_{m-1}, R₀)
    
    # Actually the contraction gives us a tensor with indices:
    # r₀, d₀, d₁, ..., d_{m-1}
    # We need to reshape to (R₀, D₀×...×D_{m-1}, R₀)
    
    # But wait, we also need to trace over r₀ for ring closure.
    # Let me reconsider.
    
    # The emb cores form a ring too (they're connected by r indices).
    # Full contraction:
    # Σ_{r₁,...,r_{m-1}} E₀[d₀,r₀,r₁] · E₁[d₁,r₁,r₂] · ... · E_{m-1}[d_{m-1},r_{m-1},r₀]
    # This gives: (r₀, d₀, d₁, ..., d_{m-1}) with r₀ appearing twice (trace)
    # 
    # For the embedding precontraction, we DON'T trace over r₀ yet.
    # We keep r₀ as an open index, and the result is (r₀, d₀×...×d_{m-1}, r₀)
    
    # So the correct equation:
    # E₀[d₀, r₀, r₁] · E₁[d₁, r₁, r₂] · ... · E_{m-1}[d_{m-1}, r_{m-1}, r_m]
    # where r_m = r₀ (ring closure)
    # Result: (r₀, d₀, d₁, ..., d_{m-1})
    # Then reshape to (r₀, D, r₀)
    
    # The equation:
    operands = []
    for i in range(m):
        d_var = f"d{i}"
        r_in = f"r{i}"
        r_out = f"r{i+1}"
        operands.append(f"{d_var}{r_in}{r_out}")
    
    # Last core: r_{m} = r₀ (ring closure)
    eq = ','.join(operands) + f'->r0' + ''.join([f'd{i}' for i in range(m)])
    # Wait, this gives r₀ as first index, then all d_i. That's (R₀, D₀, D₁, ..., D_{m-1})
    # which reshapes to (R₀, D, R₀) only if we also have the last r.
    
    # Actually I'm confusing myself. Let me think step by step.
    
    # We want emb precontraction to produce (R₀, D, R₀).
    # Each emb core has shape (D_i, R_i, R_{i+1}).
    # The contraction over the r indices gives us a tensor with:
    # - d₀, d₁, ..., d_{m-1} (free indices)
    # - r₀ (the ring closure index, appears in first and last core)
    
    # So the result has shape (R₀, D₀, D₁, ..., D_{m-1}) = (R₀, D₀, D₁, ..., D_{m-1})
    # But we want (R₀, D, R₀) where D = D₀ × ... × D_{m-1}
    
    # This means we need to ALSO include the last r₀ index in the output.
    # But it's traced (contracted) over! That's the problem.
    
    # Let me reconsider. For TR with ring closure:
    # Σ_{r₀} Σ_{r₁,...,r_{m-1}} E₀[d₀,r₀,r₁] · E₁[d₁,r₁,r₂] · ... · E_{m-1}[d_{m-1},r_{m-1},r₀]
    # = (d₀, d₁, ..., d_{m-1})  — a (D₀, D₁, ..., D_{m-1}) tensor
    # = reshape to (D,) — a vector!
    
    # That's NOT what we want. We want the precontraction to produce (R₀, D, R₀)
    # so that it can be contracted with the vocab chain result (B, R₀, R₀).
    
    # I think the issue is that for the embedding precontraction, we DON'T want ring closure.
    # We want to compute:
    # E₀[d₀, r₀, r₁] · E₁[d₁, r₁, r₂] · ... · E_{m-1}[d_{m-1}, r_{m-1}, r_m]
    # = (r₀, d₀, d₁, ..., d_{m-1}, r_m)
    # = (R₀, D₀, D₁, ..., D_{m-1}, R_m)
    # reshape to (R₀, D, R_m)
    
    # But in TR, R₀ = R_m = R (the ring closes). So (R₀, D, R_m) = (R, D, R).
    
    # So the emb precontraction should NOT trace over r₀.
    # It should keep r₀ and r_m as open indices.
    
    # The equation for emb precontraction:
    # E₀[d₀, r₀, r₁] · E₁[d₁, r₁, r₂] · ... · E_{m-1}[d_{m-1}, r_{m-1}, r_m]
    # Result: (r₀, d₀, d₁, ..., d_{m-1}, r_m)
    
    # This is just a sequential contraction without tracing.
    # For m=2: E₀[d₀, r₀, r₁] · E₁[d₁, r₁, r₂] → (r₀, d₀, d₁, r₂)
    # For m=3: E₀[d₀, r₀, r₁] · E₁[d₁, r₁, r₂] · E₂[d₂, r₂, r₃] → (r₀, d₀, d₁, d₂, r₃)
    
    # The equation:
    operands = []
    for i in range(m):
        d_var = f"d{i}"
        r_in = f"r{i}"
        r_out = f"r{i+1}"
        operands.append(f"{d_var}{r_in}{r_out}")
    
    eq = ','.join(operands) + f'->r0' + ''.join([f'd{i}' for i in range(m)]) + f'r{m}'
    
    # Shapes: (D_i, R_i, R_{i+1})
    shapes = [s for s in emb_core_shapes]
    
    return oe.contract_expression(eq, *shapes)


def ring_closure(
    vocab_result: torch.Tensor,
    emb_contraction: torch.Tensor,
    use_efficient: bool = True,
) -> torch.Tensor:
    """Combine vocab chain result with precontracted emb contraction.
    
    vocab_result: (B, R, R)
    emb_contraction: (R, D, R)
    Returns: (B, D)
    
    The ring closure traces over the pair of R dimensions:
    result[b, d] = Σ_{r,r'} vocab_result[b, r, r'] · emb_contraction[r', d, r]
    """
    if use_efficient:
        return _ring_closure_efficient(vocab_result, emb_contraction)
    else:
        return _ring_closure_einsum(vocab_result, emb_contraction)


def _ring_closure_efficient(
    vocab_result: torch.Tensor,
    emb_contraction: torch.Tensor,
) -> torch.Tensor:
    """Memory-efficient ring closure via R-dimension loop.
    
    Avoids O(B·D·R²) intermediate by looping over R.
    For R=32, this is ~2× more compute but O(R) memory.
    """
    R = vocab_result.shape[1]
    B = vocab_result.shape[0]
    D = emb_contraction.shape[1]
    device = vocab_result.device
    dtype = vocab_result.dtype
    
    output = torch.zeros(B, D, device=device, dtype=dtype)
    for r in range(R):
        # vocab_result[:, r, :] → (B, R)
        # emb_contraction[:, :, r] → (R, D)
        output += torch.mm(vocab_result[:, r, :], emb_contraction[:, :, r])
    
    return output


def _ring_closure_einsum(
    vocab_result: torch.Tensor,
    emb_contraction: torch.Tensor,
) -> torch.Tensor:
    """Einsum-based ring closure (may create large intermediate)."""
    eq = "bri,rdi->bd"
    shapes = [list(vocab_result.shape), list(emb_contraction.shape)]
    path, _ = ContractionPathCache.get_path(eq, shapes)
    return oe.contract(eq, vocab_result, emb_contraction, optimize=path)
```

### 6.6 `tensor_ring_decomposition/core/tensor_ring.py`

**Purpose**: Core TR mathematics — reconstruction, validation, utilities.

```python
class TRTensor:
    """Tensor Ring tensor representation."""
    
    def __init__(self, vocab_cores: List[torch.Tensor], emb_cores: List[torch.Tensor]):
        self.vocab_cores = vocab_cores
        self.emb_cores = emb_cores
    
    def to_tensor(self) -> torch.Tensor:
        """Reconstruct full V×D matrix from ring factors.
        
        WARNING: Materializes full matrix. Use only for debugging.
        """
        # Chain emb cores to get (R₀, D, R₀)
        emb_result = self.emb_cores[0]
        for core in self.emb_cores[1:]:
            # Contract over shared rank
            emb_result = torch.einsum('dri,dRj->rRij', emb_result, core)
            emb_result = emb_result.reshape(emb_result.shape[0], -1, emb_result.shape[-1])
        
        # Chain vocab cores to get (V, R₀, R₀)
        vocab_result = self.vocab_cores[0]
        for core in self.vocab_cores[1:]:
            vocab_result = torch.einsum('vri,rRj->vRij', vocab_result, core)
            vocab_result = vocab_result.reshape(vocab_result.shape[0], -1, vocab_result.shape[-1])
        
        # Ring closure: trace over R₀
        # vocab_result: (V, R₀, R₀), emb_result: (R₀, D, R₀)
        result = torch.einsum('vri,rdj->vdi', vocab_result, emb_result)
        return result.reshape(-1, result.shape[-1])  # (V, D)
    
    def parameter_count(self) -> int:
        """Total parameters."""
        return sum(c.numel() for c in self.vocab_cores + self.emb_cores)
```

### 6.7 `tensor_ring_decomposition/core/embedding.py`

**Purpose**: Main user-facing module.

```python
class TensorRingEmbedding(nn.Module):
    """Tensor Ring Decomposition for Embedding compression."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        rank: Optional[int] = None,
        ranks: Optional[List[int]] = None,
        ring_components: int = 4,
        target_compression: Optional[float] = None,
        target_params: Optional[int] = None,
        split_mode: Literal["balanced", "proportional", "manual"] = "balanced",
        init_method: Literal["uniform", "normal", "kaiming", "svd"] = "uniform",
        gauge_fix: Literal["none", "left", "right", "both"] = "left",
        gauge_fix_interval: int = 1000,
        padding_idx: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__()
        
        # Validate compression config
        self._validate_compression_config(rank, ranks, target_compression, target_params)
        
        # Store config
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.ring_components = ring_components
        self.split_mode = split_mode
        self.init_method = init_method
        self.gauge_fix = gauge_fix
        self.gauge_fix_interval = gauge_fix_interval
        self.padding_idx = padding_idx
        self.dtype = dtype
        
        # Compute rank if target_compression or target_params specified
        if target_compression is not None or target_params is not None:
            rank = self._solve_rank(vocab_size, embedding_dim, ring_components,
                                   target_compression, target_params)
        
        # Compute ring structure
        self.structure = compute_ring_structure(
            vocab_size, embedding_dim, ring_components, rank,
            split_mode, ranks
        )
        
        # Create cores
        self.cores = TensorRingCores(
            self.structure, init_method, gauge_fix, gauge_fix_interval,
            dtype, device
        )
        
        # Initialize
        self.cores.initialize(init_method)
        
        # Contraction paths (precomputed)
        self._vocab_expr = compute_vocab_chain_expression(
            self.structure.vocab_factor_sizes,
            rank
        )
        self._emb_expr = compute_emb_precontraction_expression(
            self.structure.emb_factor_sizes
        )
        
        # Eval cache
        self._emb_cache: Optional[torch.Tensor] = None
        self._cache_valid: bool = False
        
        # Log
        logger.info(
            f"TensorRingEmbedding initialized: V={vocab_size}, D={embedding_dim}, "
            f"rank={rank}, components={ring_components}, "
            f"compression={self.compression_ratio:.1f}x, "
            f"params={self.num_parameters:,}"
        )
    
    def _validate_compression_config(self, rank, ranks, target_compression, target_params):
        """Validate exactly one compression config is set."""
        configs = [rank is not None, ranks is not None,
                   target_compression is not None, target_params is not None]
        if sum(configs) != 1:
            raise ValueError(
                f"Exactly one of rank, ranks, target_compression, target_params must be set. "
                f"Got {sum(configs)} set."
            )
        if target_compression is not None and target_compression <= 1.0:
            raise ValueError(f"target_compression must be > 1.0, got {target_compression}")
    
    def _solve_rank(self, vocab_size, embedding_dim, ring_components,
                    target_compression, target_params):
        """Solve for rank given target compression or target params."""
        if target_compression is not None:
            dense_params = vocab_size * embedding_dim
            target_params = dense_params / target_compression
        
        # For balanced TR: total_params = R² × (V + D)
        R = math.sqrt(target_params / (vocab_size + embedding_dim))
        return max(2, int(round(R)))
    
    def _vocab_chain(self, flat_indices: torch.Tensor) -> torch.Tensor:
        """Compute vocab chain: gather cores at indices → chain bmm.
        
        Args:
            flat_indices: (N,) token indices
        
        Returns:
            (N, R, R) chain result
        """
        # Gather each vocab core at indices
        gathered = [core[flat_indices] for core in self.cores.vocab_cores]
        
        # Chain bmm
        result = gathered[0]
        for core_gathered in gathered[1:]:
            result = torch.bmm(result, core_gathered)
        
        return result  # (N, R, R)
    
    def _compute_emb_contraction(self) -> torch.Tensor:
        """Compute embedding cores precontraction.
        
        Returns:
            (R, D, R) tensor
        """
        return self._emb_expr(*[core for core in self.cores.emb_cores])
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Compressed embedding lookup.
        
        Args:
            indices: (B,) or (B, seq_len) token IDs
        
        Returns:
            (B, D) or (B, seq_len, D) compressed embeddings
        """
        # Validate dtype
        if self.dtype == torch.float16:
            raise TypeError(
                "fp16 not supported due to numerical instability. Use bf16 or fp32."
            )
        
        original_shape = indices.shape
        flat = indices.view(-1)
        
        # Vocab chain
        vocab_result = self._vocab_chain(flat)  # (N, R, R)
        
        # Emb contraction (from cache or recomputed)
        if self.training or not self._cache_valid:
            emb_contraction = self._compute_emb_contraction()  # differentiable
        else:
            emb_contraction = self._emb_cache  # detached
        
        # Ring closure
        output = ring_closure(vocab_result, emb_contraction)  # (N, D)
        
        return output.view(*original_shape, self.embedding_dim)
    
    def to_eval_mode(self) -> "TensorRingEmbedding":
        """Switch to inference mode with cached emb contraction."""
        if dist.is_initialized():
            dist.barrier()  # Ensure all ranks synchronized
        
        self.eval()
        with torch.no_grad():
            self._emb_cache = self._compute_emb_contraction()
            self._cache_valid = True
        
        return self
    
    def train_mode(self) -> "TensorRingEmbedding":
        """Switch to training mode, clearing cache."""
        self.train()
        self._cache_valid = False
        self._emb_cache = None
        return self
    
    def config(self) -> dict:
        """Return immutable construction parameters."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "rank": self.structure.rank,
            "ranks": self.structure.ranks,
            "ring_components": self.ring_components,
            "split_mode": self.split_mode,
            "init_method": self.init_method,
            "gauge_fix": self.gauge_fix,
            "padding_idx": self.padding_idx,
            "compression_ratio": self.compression_ratio,
            "num_parameters": self.num_parameters,
            "vocab_factor_sizes": self.structure.vocab_factor_sizes,
            "emb_factor_sizes": self.structure.emb_factor_sizes,
        }
    
    def spectral_norms(self) -> Dict[str, float]:
        """Return spectral norm of each core."""
        return self.cores.spectral_norms()
    
    def reconstruction_error(self, original_matrix: torch.Tensor) -> float:
        """Compute relative Frobenius norm of reconstruction error."""
        with torch.no_grad():
            reconstructed = self.reconstruct()
            error = torch.norm(original_matrix - reconstructed)
            baseline = torch.norm(original_matrix)
            return (error / baseline).item()
    
    def reconstruct(self) -> torch.Tensor:
        """Reconstruct full V×D matrix. WARNING: materializes full matrix."""
        tr_tensor = TRTensor(self.cores.vocab_cores, self.cores.emb_cores)
        return tr_tensor.to_tensor()
    
    @property
    def compression_ratio(self) -> float:
        """Dense params / compressed params."""
        dense = self.vocab_size * self.embedding_dim
        compressed = self.num_parameters
        return dense / compressed
    
    @property
    def num_parameters(self) -> int:
        """Total parameters in ring factor cores."""
        return self.cores.parameter_count()
```

### 6.8 `tensor_ring_decomposition/utils/gauge.py`

**Purpose**: Gauge fixing utilities.

```python
class GaugeFixer:
    """Orthogonalize TR cores to eliminate gauge freedom."""
    
    @staticmethod
    def fix_left(cores: nn.ParameterList):
        """QR-based left gauge fix: orthogonalize from left to right."""
        for i in range(len(cores)):
            old_shape = cores[i].shape  # (factor_size, R, R_next)
            flat = cores[i].data.reshape(-1, old_shape[2])
            
            Q, R_mat = torch.linalg.qr(flat)
            
            # Fix sign ambiguity
            sign = torch.sign(torch.diag(R_mat))
            Q = Q * sign
            R_mat = torch.diag(sign) @ R_mat
            
            cores[i].data = Q.reshape(old_shape)
            
            if i < len(cores) - 1:
                # Propagate R to next core
                next_core = cores[i + 1]
                R_expanded = R_mat.unsqueeze(0).expand(next_core.shape[0], -1, -1)
                cores[i + 1].data = torch.bmm(R_expanded, next_core.data)
    
    @staticmethod
    def fix_right(cores: nn.ParameterList):
        """RQ-based right gauge fix: orthogonalize from right to left."""
        for i in range(len(cores) - 1, -1, -1):
            old_shape = cores[i].shape
            flat = cores[i].data.reshape(old_shape[0], -1)
            
            # RQ decomposition (via reversed QR)
            Q, R_mat = torch.linalg.qr(flat.T)
            L = Q.T  # Lower triangular from RQ
            R_factor = R_mat.T
            
            cores[i].data = L.reshape(old_shape)
            
            if i > 0:
                # Propagate L to previous core
                prev_core = cores[i - 1]
                L_expanded = L.unsqueeze(1).expand(-1, prev_core.shape[2], -1)
                cores[i - 1].data = torch.bmm(prev_core.data, L_expanded)
    
    @staticmethod
    def spectral_norms(cores: nn.ParameterList) -> List[float]:
        """Compute spectral norm of each core."""
        norms = []
        for core in cores:
            flat = core.data.reshape(-1, core.shape[-1])
            norm = torch.linalg.svdvals(flat)[0].item()
            norms.append(norm)
        return norms
```

### 6.9 `tensor_ring_decomposition/utils/serialization.py`

**Purpose**: Save/load with safetensors + HMAC verification.

```python
CHECKPOINT_SCHEMA_VERSION = "1.0"

def save(
    embedding: TensorRingEmbedding,
    path: str,
    secret_key: Optional[bytes] = None,
    extra_metadata: Optional[dict] = None,
):
    """Save TR embedding with safetensors + HMAC-verified manifest.
    
    Creates two files:
    - {path}.safetensors: Weights (safetensors format, no code execution)
    - {path}.json: Manifest with metadata and hash
    
    Args:
        embedding: TensorRingEmbedding to save
        path: Base path (without extension)
        secret_key: Optional HMAC key for hash verification
        extra_metadata: Optional additional metadata to include
    """
    import safetensors.torch as sf
    
    # Collect weights
    weights = {name: param.data for name, param in embedding.named_parameters()}
    
    # Save weights as safetensors
    weights_path = Path(path).with_suffix(".safetensors")
    sf.save_file(weights, weights_path)
    
    # Compute hash
    weights_bytes = weights_path.read_bytes()
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
        "weight_shapes": {name: list(param.shape) for name, param in embedding.named_parameters()},
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
) -> TensorRingEmbedding:
    """Load TR embedding from safetensors + manifest.
    
    Verifies hash before loading. Uses safetensors (not torch.load) —
    no code execution possible.
    
    Args:
        path: Base path (without extension)
        secret_key: HMAC key (must match key used in save())
        device: Target device for weights
    
    Returns:
        Loaded TensorRingEmbedding
    
    Raises:
        SecurityError: If hash mismatch detected
        FileNotFoundError: If files don't exist
    """
    import safetensors.torch as sf
    
    # Load manifest
    manifest_path = Path(path).with_suffix(".json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    manifest = json.loads(manifest_path.read_text())
    
    # Verify hash
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
            f"Hash mismatch! Expected {expected[:16]}..., got {manifest['core_hash'][:16]}... "
            f"This checkpoint may have been tampered with."
        )
    
    # Load weights (safetensors — no code execution)
    weights = sf.load_file(weights_path, device=device)
    
    # Reconstruct embedding
    config = manifest["tr_config"]
    embedding = TensorRingEmbedding(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        rank=config["rank"],
        ring_components=config["ring_components"],
        split_mode=config["split_mode"],
        init_method="uniform",  # Will be overwritten by loaded weights
        gauge_fix=config.get("gauge_fix", "left"),
        padding_idx=config.get("padding_idx"),
    )
    embedding.load_state_dict(weights)
    
    return embedding
```

### 6.10 `tensor_ring_decomposition/utils/validation.py`

**Purpose**: Input validation.

```python
def validate_indices(indices: torch.Tensor, vocab_size: int, padding_idx: Optional[int] = None):
    """Validate token indices are within bounds."""
    if indices.min() < 0:
        if padding_idx is None or indices.min() < -1:
            raise IndexError(
                f"Indices contain negative values (min={indices.min().item()}). "
                f"Set padding_idx if negative indices are intentional."
            )
    if indices.max() >= vocab_size:
        raise IndexError(
            f"Index {indices.max().item()} out of range for vocab_size={vocab_size}"
        )

def validate_compatibility(embedding: TensorRingEmbedding, downstream_module: nn.Module):
    """Check that TR embedding output dimension matches downstream module input."""
    # Find first linear layer in downstream module
    for name, module in downstream_module.named_modules():
        if isinstance(module, nn.Linear):
            if module.in_features != embedding.embedding_dim:
                raise ValueError(
                    f"Downstream module '{name}' expects input dim {module.in_features}, "
                    f"but TR embedding outputs dim {embedding.embedding_dim}"
                )
            break
```

### 6.11 `tensor_ring_decomposition/monitoring/compression.py`

```python
class CompressionTracker:
    """Track compression statistics during training."""
    
    def __init__(self, embedding: TensorRingEmbedding, log_interval: int = 1000):
        self.embedding = embedding
        self.log_interval = log_interval
        self.dense_params = embedding.vocab_size * embedding.embedding_dim
    
    def log_metrics(self, step: int):
        """Log compression metrics every log_interval steps."""
        if step % self.log_interval != 0:
            return
        
        metrics = {
            "tr/compression_ratio": self.embedding.compression_ratio,
            "tr/num_parameters": self.embedding.num_parameters,
            "tr/params_saved": self.dense_params - self.embedding.num_parameters,
            "tr/spectral_norms": self.embedding.spectral_norms(),
        }
        
        # Log to configured logger (wandb, tensorboard, etc.)
        for key, value in metrics.items():
            logger.log(key, value, step=step)
    
    def memory_bytes(self) -> int:
        """Estimate memory footprint in bytes."""
        return self.embedding.num_parameters * 4  # Assuming fp32
```

### 6.12 `tensor_ring_decomposition/monitoring/quality.py`

```python
class QualityGate:
    """Automated quality monitoring with rollback trigger."""
    
    def __init__(self, baseline_metrics: Dict[str, float], threshold: float = 0.02):
        """
        Args:
            baseline_metrics: Baseline quality metrics (e.g., from dense model)
            threshold: Maximum allowed relative drop before rollback (default: 2%)
        """
        self.baseline = baseline_metrics
        self.threshold = threshold
        self.triggered = False
    
    def check(self, current_metrics: Dict[str, float]) -> bool:
        """Returns True if quality is acceptable.
        
        Checks each metric against baseline. If any drops > threshold,
        returns False and logs the failure.
        """
        for key, baseline_value in self.baseline.items():
            if key not in current_metrics:
                continue
            current_value = current_metrics[key]
            drop = (baseline_value - current_value) / abs(baseline_value)
            
            if drop > self.threshold:
                logger.error(
                    f"Quality gate FAILED: {key} dropped {drop:.1%} "
                    f"(baseline={baseline_value:.4f}, current={current_value:.4f})"
                )
                self.triggered = True
                return False
        
        return True
    
    def should_rollback(self) -> bool:
        """Whether rollback should be triggered."""
        return self.triggered
```

### 6.13 `tensor_ring_decomposition/monitoring/callbacks.py`

```python
class TensorRingCallback:
    """Training loop callback for monitoring TR embeddings.
    
    Compatible with PyTorch Lightning and HuggingFace Trainer.
    """
    
    def __init__(
        self,
        embedding: TensorRingEmbedding,
        quality_gate: Optional[QualityGate] = None,
        gauge_fix: bool = True,
        log_interval: int = 1000,
    ):
        self.embedding = embedding
        self.quality_gate = quality_gate
        self.gauge_fix = gauge_fix
        self.tracker = CompressionTracker(embedding, log_interval)
    
    def on_train_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """Called after each training batch."""
        self.tracker.log_metrics(batch_idx)
        
        if self.gauge_fix:
            self.embedding.cores._apply_gauge_fix()
    
    def on_validation_end(self, metrics: Dict[str, float], **kwargs):
        """Called after validation. Check quality gate."""
        if self.quality_gate is not None:
            if not self.quality_gate.check(metrics):
                logger.warning("Quality gate triggered — consider rollback")
```

### 6.14 `tensor_ring_decomposition/integrations/huggingface.py`

```python
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
        # Validate compatibility
        validate_compatibility(tr_embedding, model)
        
        # Find and replace embedding layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                # Create TR version with same config
                tr_emb = TensorRingEmbedding(
                    module.num_embeddings,
                    module.embedding_dim,
                    rank=tr_embedding.structure.rank,
                )
                # Copy initialized weights
                tr_emb.cores.initialize("svd", module.weight.data)
                
                # Replace
                parent_name = '.'.join(name.split('.')[:-1])
                parent = model if not parent_name else dict(model.named_modules())[parent_name]
                setattr(parent, name.split('.')[-1], tr_emb)
        
        return model
```

### 6.15 `tensor_ring_decomposition/quantization/quantize.py`

Post-MVP module for int8 quantization.

```python
class QuantizedTensorRingEmbedding:
    """Post-training int8 quantization for TR embeddings.
    
    Per-core quantization with int8 gather and AMX-accelerated matmul.
    """
    
    def __init__(self, embedding: TensorRingEmbedding):
        self.original = embedding
        self.scales = {}
        self.zero_points = {}
    
    def quantize(self, per_core_scales: bool = True):
        """Quantize each core to int8 with per-core scale."""
        for name, core in embedding.named_parameters():
            if per_core_scales:
                scale = core.data.abs().max() / 127.0
                zero_point = 0
            else:
                # Per-tensor scale
                scale = core.data.abs().max() / 127.0
                zero_point = 0
            
            self.scales[name] = scale
            self.zero_points[name] = zero_point
    
    def forward_int8(self, indices: torch.Tensor) -> torch.Tensor:
        """Forward pass with int8 gather and matmul."""
        # Gather at int8 → dequant → matmul
        # Or: gather at int8 → propagate scales → torch.ops.aten._int_mm
        pass
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

| Test | File | What It Tests |
|------|------|--------------|
| `test_factorize_dimension` | `test_core.py` | `factorize_dimension` produces correct product |
| `test_ring_structure` | `test_core.py` | `compute_ring_structure` with all split_modes |
| `test_cores_init_uniform` | `test_core.py` | Xavier init produces valid cores |
| `test_cores_init_svd` | `test_core.py` | SVD init reconstructs to < 1e-6 relative error |
| `test_contraction_path` | `test_contraction.py` | opt_einsum path matches naive einsum |
| `test_ring_closure` | `test_contraction.py` | Ring closure produces correct output |
| `test_ring_closure_memory_efficient` | `test_contraction.py` | Memory-efficient variant matches einsum |
| `test_gauge_fix_left` | `test_core.py` | QR orthogonalization produces Q with unit columns |
| `test_gauge_fix_right` | `test_core.py` | RQ decomposition correctness |
| `test_embedding_forward` | `test_embedding.py` | Forward pass produces correct shape |
| `test_embedding_gradient_flow` | `test_embedding.py` | Gradients flow to all cores |
| `test_eval_mode_cache` | `test_embedding.py` | Cache is populated and used correctly |
| `test_train_mode_cache_cleared` | `test_embedding.py` | Cache is cleared on train_mode() |
| `test_padding_idx` | `test_embedding.py` | Padding index behavior matches nn.Embedding |
| `test_config` | `test_embedding.py` | config() returns correct values |
| `test_compression_ratio` | `test_embedding.py` | Compression ratio calculation is correct |
| `test_serialization_roundtrip` | `test_serialization.py` | save() + load() produces identical output |
| `test_serialization_hash_verification` | `test_serialization.py` | Modified checkpoint raises SecurityError |
| `test_validation_out_of_bounds` | `test_validation.py` | Indices >= vocab_size raises IndexError |
| `test_validation_negative_indices` | `test_validation.py` | Negative indices without padding_idx raises |
| `test_quality_gate_pass` | `test_quality.py` | Quality gate passes when metrics within threshold |
| `test_quality_gate_fail` | `test_quality.py` | Quality gate fails when metrics drop > threshold |

### 7.2 Integration Tests

| Test | File | What It Tests |
|------|------|--------------|
| `test_bert_compression` | `test_integration.py` | Compress BERT embeddings, verify MLM accuracy |
| `test_gpt2_compression` | `test_integration.py` | Compress GPT-2 embeddings, verify perplexity |
| `test_hf_integration` | `test_integration.py` | from_huggingface + replace_in_model works |
| `test_ddp_consistency` | `test_integration.py` | DDP cache synchronization across ranks |
| `test_gradient_checkpointing` | `test_integration.py` | Works with torch.utils.checkpoint |

### 7.3 Performance Tests

| Test | File | What It Tests |
|------|------|--------------|
| `test_forward_latency` | `test_integration.py` | Forward pass < 2× dense lookup time |
| `test_memory_footprint` | `test_integration.py` | Memory < compression_ratio × dense memory |
| `test_eval_vs_training_speedup` | `test_integration.py` | eval_mode is ≥ 1.5× faster than training |

### 7.4 GPU Tests

| Test | File | What It Tests |
|------|------|--------------|
| `test_gpu_forward` | `test_integration.py` | Forward pass on GPU |
| `test_bf16_training` | `test_integration.py` | bf16 training produces valid gradients |
| `test_fp16_rejection` | `test_integration.py` | fp16 raises TypeError |

---

## 8. Examples & Benchmarks

### 8.1 LLM Compression Example

```python
# examples/llm_compression/bert_embedding.py
from transformers import BertModel
from tensor_ring_decomposition import TensorRingEmbedding

# Load pretrained BERT
model = BertModel.from_pretrained("bert-base-uncased")
original_emb = model.get_input_embeddings()

# Create TR version
tr_emb = TensorRingEmbedding.from_pretrained(
    original_emb.weight.data,
    rank=8,
    ring_components=4,
    gauge_fix="left",
)

# Compare
print(f"Dense params: {original_emb.weight.numel():,}")
print(f"TR params: {tr_emb.num_parameters:,}")
print(f"Compression: {tr_emb.compression_ratio:.1f}x")
print(f"Reconstruction error: {tr_emb.reconstruction_error(original_emb.weight.data):.6f}")

# Replace in model
from tensor_ring_decomposition.integrations.huggingface import HuggingFaceTensorRingEmbedding
HuggingFaceTensorRingEmbedding.replace_in_model(model, tr_emb)
```

### 8.2 Benchmark Script

```python
# examples/run_benchmark.py
import torch
import time
from tensor_ring_decomposition import TensorRingEmbedding

def benchmark(vocab_size, dim, rank, device="cuda", num_iters=1000):
    """Benchmark TR embedding vs dense."""
    # Create TR embedding
    tr_emb = TensorRingEmbedding(vocab_size, dim, rank=rank).to(device)
    
    # Create dense embedding
    dense_emb = torch.nn.Embedding(vocab_size, dim).to(device)
    
    # Warmup
    indices = torch.randint(0, vocab_size, (32, 128)).to(device)
    for _ in range(10):
        tr_emb(indices)
        dense_emb(indices)
    
    # Benchmark training mode
    tr_emb.train_mode()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        tr_emb(indices)
    torch.cuda.synchronize()
    tr_train_time = time.time() - start
    
    # Benchmark eval mode
    tr_emb.to_eval_mode()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        tr_emb(indices)
    torch.cuda.synchronize()
    tr_eval_time = time.time() - start
    
    # Benchmark dense
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        dense_emb(indices)
    torch.cuda.synchronize()
    dense_time = time.time() - start
    
    print(f"V={vocab_size}, D={dim}, R={rank}")
    print(f"  Dense:      {dense_time:.3f}s")
    print(f"  TR train:   {tr_train_time:.3f}s ({tr_train_time/dense_time:.2f}x)")
    print(f"  TR eval:    {tr_eval_time:.3f}s ({tr_eval_time/dense_time:.2f}x)")
    print(f"  Compression: {tr_emb.compression_ratio:.1f}x")

if __name__ == "__main__":
    for rank in [4, 8, 16, 32]:
        benchmark(50000, 768, rank)
```

---

## 9. Open Decisions

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| 1 | **Gradient checkpointing for deep chains** | On/Off/Auto | Auto: enable when ring_components > 6 |
| 2 | **CUDA kernel fusion for ring closure** | Implement now/Post-MVP | Post-MVP: einsum loop is fast enough for R≤64 |
| 3 | **Memory-mapped vocab cores for V>1M** | torch.load mmap / numpy.memmap | torch.load mmap=True |
| 4 | **Sequence packing support** | Pad-contract-unpack / Custom | Pad-contract-unpack initially |
| 5 | **Quantization in initial release** | Include/Exclude | Exclude (Post-MVP module exists) |
| 6 | **TorchRec integration priority** | Phase 2/Phase 3 | Phase 3 (after HuggingFace) |

---

## 10. Appendix: Review Round Synthesis

### Round 1 Reviews

| Perspective | Key Findings |
|-------------|--------------|
| **ML Engineer** | Gradient flow through cache, gauge freedom causing scale drift, SVD init risks, ALS vs GD mismatch, catastrophic forgetting during fine-tuning |
| **Systems Engineer** | Memory efficiency overstated for large B, small tensor GPU underutilization, opt_einsum caching race condition, eval_mode cache double-edged, batch size non-linear scaling, fragmentation risk, gradient checkpointing needed, vocab core indexing O(num_cores) |
| **Researcher** | V×D→TR mapping under-specified, ring closure trace correctness, SVD init needs scrutiny, TT vs TR correct choice, rank selection under-theorized, balanced ranks limit expressiveness, orthonormal initialization missing |
| **MLOps** | Serialization underspecified, QualityMetrics needs production definition, HF integration coupling risk, eval_mode unsafe for DDP, SVD init single-point dependency, rollback story incomplete |

### Round 2 Reviews

| Perspective | Key Findings |
|-------------|--------------|
| **Numerical Analyst** | fp16 error ~10⁻³ from long bmm chains, autograd through opt_einsum works, ring closure O(B·D·R·R) overflow risk, SVD init stability concerns, bf16 preferred over fp16 |
| **API Designer** | Overloaded compression config confusing, n_cores ambiguous, split_mode needs Literal validation, rank=None behavior opaque, padding_idx parity missing, init_method="svd" misleading for training, device parameter vestigial, builder pattern needed |
| **Data Pipeline** | DataLoader compatible, prefetching feasible, DDP per-rank cache safe if identical init, checkpoint frequency should be epoch+step, V>1M needs memory-mapping, variable seq_len needs pad-contract-unpack |
| **Security** | torch.load unsafe (must use safetensors), opt_einsum path deterministic, SVD intermediates persist in memory, model inversion possible from compressed cores, SHA256 insufficient without HMAC, DDP rank poisoning via cache inconsistency |
| **Quantization Expert** | Per-core int8 ideal, R=32 FP16 TensorCore may beat INT8 on A100, gather works with qint8, quantization error ~0.3-1.2% MSE, per-token for vocab cores, per-batch for emb cores |

### Fixes Applied

| Original Issue | Fix Applied |
|---------------|-------------|
| `eval_mode()` breaks gradient flow | Explicit `to_eval_mode()` / `train_mode()` with cache validity flag |
| No gauge fixing | `GaugeFixer` with QR orthogonalization, configurable interval |
| `torch.load` unsafe | `safetensors.load_file` only + HMAC-SHA256 hash |
| `n_cores` ambiguous | Renamed to `ring_components` |
| 3-way compression config | Exactly ONE of rank/ranks/target_compression/target_params; ValueError otherwise |
| `init_method="svd"` misleading | Default changed to `"uniform"`; SVD via `from_pretrained()` |
| fp16 training unsafe | Explicit rejection with clear error message |
| Ring closure B·D·R·R overflow | Memory-efficient loop over R dimension |
| DDP cache inconsistency | `dist.barrier()` before cache + hash verification |
| No padding_idx | Added `padding_idx` parameter with nn.Embedding parity |
| No config introspection | Added `config()` method |
| No builder pattern | Added `from_compression_ratio()`, `from_target_params()`, `from_pretrained()` class methods |

---

## Implementation Priority

| Phase | Files | Key Focus |
|-------|-------|-----------|
| **Phase 1** | `factorization.py`, `cores.py`, `contraction.py`, `tensor_ring.py`, `gauge.py` | Core algorithms |
| **Phase 2** | `embedding.py`, `validation.py` | User-facing module |
| **Phase 3** | `serialization.py`, `path_cache.py` | Safety and performance |
| **Phase 4** | `compression.py`, `quality.py`, `callbacks.py` | Monitoring |
| **Phase 5** | `huggingface.py` | Framework integration |
| **Phase 6** | All tests | Verification |
| **Phase 7** | Examples and benchmarks | Documentation |
| **Phase 8** | `quantization/` | Post-MVP |
