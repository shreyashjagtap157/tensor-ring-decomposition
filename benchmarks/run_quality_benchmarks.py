"""Run quality benchmarks on representative models using synthetic matrices."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tensor_ring_decomposition import TensorRingEmbedding

models_to_test = [
    ("albert-base-v2", 30000, 128),
    ("bert-base-uncased", 30522, 768),
    ("gpt2", 50257, 768),
    ("t5-small", 32128, 512),
    ("roberta-base", 50265, 768),
    ("xlnet-base-cased", 32000, 768),
    ("distilbert-base-uncased", 30522, 768),
    ("bert-large-uncased", 30522, 1024),
    ("gpt2-medium", 50257, 1024),
    ("t5-base", 32128, 768),
]
ranks_to_test = [4, 8, 16, 24, 32]
device = "cpu"

results = {}
for name, V, D in models_to_test:
    print(f"Benchmarking {name} ({V}x{D})...")
    matrix = torch.randn(V, D, device=device)
    dense_params = V * D
    model_results = []

    for rank in ranks_to_test:
        t0 = time.monotonic()
        emb = TensorRingEmbedding.from_pretrained(matrix, rank=rank, device=device)
        comp_time = time.monotonic() - t0

        recon_err = emb.reconstruction_error(matrix)
        eos_10 = emb.eigenspace_overlap_score(matrix, k=10)
        trust = emb.trustworthiness(matrix, sample_size=2000)
        cont = emb.continuity(matrix, sample_size=2000)
        da_err = emb.distribution_aware_reconstruction_error(matrix)

        model_results.append({
            "rank": rank,
            "compression_ratio": round(emb.compression_ratio, 1),
            "tr_params": emb.num_parameters,
            "pct_of_dense": round(emb.num_parameters / dense_params * 100, 3),
            "reconstruction_error": round(recon_err, 6),
            "reconstruction_error_pct": round(recon_err * 100, 3),
            "distribution_aware_error": round(da_err, 6),
            "eigenspace_overlap_k10": round(eos_10, 4),
            "trustworthiness": round(trust, 4),
            "continuity": round(cont, 4),
            "comp_time_s": round(comp_time, 1),
            "spectral_norms": {k: round(v, 4) for k, v in emb.spectral_norms().items()},
        })
        print(f"  R={rank}: comp={emb.compression_ratio:.0f}x "
              f"recon={recon_err*100:.1f}% EOS@10={eos_10:.4f} "
              f"Trust={trust:.4f} Cont={cont:.4f}")

    results[name] = {
        "vocab_size": V,
        "embedding_dim": D,
        "dense_params": dense_params,
        "results": model_results,
    }

output_path = "benchmark_quality_results.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {output_path}")

print("\n" + "=" * 90)
print("FINAL QUALITY METRICS SUMMARY")
print("=" * 90)
hdr = f"{'Model':25s} {'R':>3} {'Comp':>8} {'Recon%':>8} {'DA_Err':>8} {'EOS@10':>7} {'Trust':>7} {'Cont':>7}"
print(hdr)
print("-" * len(hdr))
for name, data in results.items():
    for r in data["results"]:
        print(f"{name[:23]:25s} {r['rank']:>3} "
              f"{r['compression_ratio']:>7.1f}x "
              f"{r['reconstruction_error_pct']:>7.2f}% "
              f"{r['distribution_aware_error']:>8.4f} "
              f"{r['eigenspace_overlap_k10']:>7.4f} "
              f"{r['trustworthiness']:>7.4f} "
              f"{r['continuity']:>7.4f}")
