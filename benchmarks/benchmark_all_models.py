"""Comprehensive benchmark: test ALL registered models with ALL metrics.

Tests every model profile in the registry across multiple ranks.
Generates a complete JSON report with every available metric.
"""

import json
import math
import sys
import time
import torch
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensor_ring_decomposition import (
    TensorRingEmbedding, ModelRegistry, ModelProfile, compute_ring_structure,
    list_models,
)


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def compute_analytical_profile(profile: ModelProfile, ranks: List[int]) -> List[Dict]:
    """Compute analytical metrics for a profile at given ranks (no matrix needed)."""
    rows = []
    for r in ranks:
        dense = profile.dense_params
        struct = compute_ring_structure(
            profile.vocab_size, profile.embedding_dim,
            profile.ring_components, r,
        )
        total_params = 0
        for i in range(struct.n_vocab_cores):
            total_params += struct.vocab_factor_sizes[i] * r * r
        for i in range(struct.n_emb_cores):
            total_params += struct.emb_factor_sizes[i] * r * r
        comp_ratio = dense / total_params if total_params > 0 else float('inf')

        core_params = []
        for i, s in enumerate(struct.vocab_factor_sizes):
            core_params.append({
                "name": f"vocab_{i}", "shape": [s, r, r], "params": s * r * r,
            })
        for i, s in enumerate(struct.emb_factor_sizes):
            core_params.append({
                "name": f"emb_{i}", "shape": [s, r, r], "params": s * r * r,
            })

        rows.append({
            "rank": r,
            "dense_params": dense,
            "tr_params": total_params,
            "compression_ratio": round(comp_ratio, 1),
            "pct_of_dense": round(total_params / dense * 100, 3),
            "params_saved": dense - total_params,
            "params_saved_fmt": fmt_params(dense - total_params),
            "ring_components": profile.ring_components,
            "n_vocab_cores": struct.n_vocab_cores,
            "n_emb_cores": struct.n_emb_cores,
            "vocab_factors": struct.vocab_factor_sizes,
            "emb_factors": struct.emb_factor_sizes,
            "ranks_per_boundary": struct.ranks,
            "core_details": core_params,
        })
    return rows


def compute_full_metrics(
    profile: ModelProfile, rank: int, matrix: torch.Tensor, device: torch.device,
) -> Dict:
    """Compute ALL available quality metrics for a TR embedding at given rank."""
    V, D = matrix.shape
    dense_params = V * D

    t0 = time.monotonic()
    emb = TensorRingEmbedding.from_pretrained(
        matrix, rank=rank, ring_components=profile.ring_components,
        init_method="svd", device=device,
    )
    comp_time = time.monotonic() - t0
    tr_params = emb.num_parameters
    comp_ratio = emb.compression_ratio

    t1 = time.monotonic()
    recon_err = emb.reconstruction_error(matrix)
    t_recon = time.monotonic() - t1

    t2 = time.monotonic()
    eos_10 = emb.eigenspace_overlap_score(matrix, k=10)
    eos_50 = emb.eigenspace_overlap_score(matrix, k=min(50, D))
    t_eos = time.monotonic() - t2

    t3 = time.monotonic()
    trust = emb.trustworthiness(matrix, n_neighbors=15, metric="euclidean", sample_size=2000)
    t_trust = time.monotonic() - t3

    t4 = time.monotonic()
    cont = emb.continuity(matrix, n_neighbors=15, metric="euclidean", sample_size=2000)
    t_cont = time.monotonic() - t4

    da_err = emb.distribution_aware_reconstruction_error(matrix)
    sn = emb.spectral_norms()

    return {
        "rank": rank,
        "compression_ratio": round(comp_ratio, 1),
        "tr_params": tr_params,
        "pct_of_dense": round(tr_params / dense_params * 100, 3),
        "reconstruction_error": round(recon_err, 6),
        "reconstruction_error_pct": round(recon_err * 100, 3),
        "distribution_aware_error": round(da_err, 6),
        "eigenspace_overlap_k10": round(eos_10, 4),
        "eigenspace_overlap_k50": round(eos_50, 4),
        "trustworthiness": round(trust, 4),
        "continuity": round(cont, 4),
        "spectral_norms": {k: round(v, 4) for k, v in sn.items()},
        "comp_time_s": round(comp_time, 1),
        "recon_time_s": round(t_recon, 2),
        "eos_time_s": round(t_eos, 2),
        "trust_time_s": round(t_trust, 2),
        "cont_time_s": round(t_cont, 2),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive all-models benchmark")
    parser.add_argument("--mode", choices=["analytical", "full", "all"],
                        default="analytical",
                        help="analytical=profiles only, full=also download+eval, all=both")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: all registered)")
    parser.add_argument("--ranks", type=str, default="2,4,8,16,24,32,48,64",
                        help="Comma-separated ranks")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    all_ranks = [int(r) for r in args.ranks.split(",")]

    profiles: List[ModelProfile] = ModelRegistry.list_all()
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
        profiles = [p for p in profiles if p.name in model_names]

    profiles = sorted(profiles, key=lambda p: p.name)

    print(f"Tensor Ring Decomposition — Comprehensive Benchmark")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Models: {len(profiles)}")
    print(f"Ranks: {all_ranks}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    report = {
        "metadata": {
            "version": "0.3.0",
            "mode": args.mode,
            "ranks": all_ranks,
            "device": str(device),
        },
        "models": {},
    }

    # Phase 1: Analytical benchmarks for ALL models
    print("Phase 1: Analytical benchmarks (all profiles)")
    print("-" * 80)
    for p in profiles:
        t0 = time.monotonic()
        analytical = compute_analytical_profile(p, all_ranks)
        elapsed = time.monotonic() - t0
        report["models"][p.name] = {
            "family": p.family.value,
            "vocab_size": p.vocab_size,
            "embedding_dim": p.embedding_dim,
            "dense_params": p.dense_params,
            "max_seq_len": p.max_seq_len,
            "ring_components": p.ring_components,
            "default_rank": p.default_rank,
            "recommended_ranks": p.recommended_ranks,
            "analytical": analytical,
            "full": {},
        }
        best = max(analytical, key=lambda x: x['compression_ratio'])
        best_comp = max(analytical, key=lambda x: x['compression_ratio'])
        print(f"  {p.name:35s} V={p.vocab_size:<6} D={p.embedding_dim:<4} "
              f"Best: R{best_comp['rank']:>2}=>{best_comp['compression_ratio']}x")

    # Phase 2: Full metrics for representative models (requires HF download)
    if args.mode in ("full", "all"):
        print(f"\nPhase 2: Full quality metrics (downloading models)")
        print("-" * 80)

        for p in profiles:
            print(f"\n  Loading {p.name}...")
            try:
                from tensor_ring_decomposition.loaders.loaders import load_embedding_matrix
                t0 = time.monotonic()
                matrix = load_embedding_matrix(p.name, device=device)
                load_time = time.monotonic() - t0
                print(f"    Loaded {list(matrix.shape)} in {load_time:.1f}s")
            except Exception as e:
                print(f"    SKIP: {e}")
                continue

            V, D = matrix.shape
            test_ranks = [r for r in all_ranks if r <= min(64, V, D)]
            if not test_ranks:
                test_ranks = [p.default_rank]

            for rank in test_ranks:
                try:
                    t0 = time.monotonic()
                    metrics = compute_full_metrics(p, rank, matrix, device)
                    elapsed = time.monotonic() - t0
                    report["models"][p.name]["full"][str(rank)] = metrics
                    print(f"    R={rank:>2}: comp={metrics['compression_ratio']}x "
                          f"recon={metrics['reconstruction_error_pct']:.2f}% "
                          f"EOS@10={metrics['eigenspace_overlap_k10']:.4f} "
                          f"Trust={metrics['trustworthiness']:.4f} "
                          f"({elapsed:.0f}s)")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"    R={rank}: ERROR: {e}")
                    report["models"][p.name]["full"][str(rank)] = {"error": str(e)}

    # Save results
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':35s} {'V':>6} {'D':>4} {'Dense':>10} {'R4':>8} {'R8':>8} {'R16':>8} {'R32':>8} {'R64':>8}")
    print("-" * 97)
    for p in profiles:
        row = report["models"][p.name]
        comps = {a["rank"]: a["compression_ratio"] for a in row["analytical"]}
        print(f"{p.name:35s} {p.vocab_size:>6} {p.embedding_dim:>4} "
              f"{fmt_params(p.dense_params):>10} "
              f"{comps.get(4, '-'):>8} {comps.get(8, '-'):>8} "
              f"{comps.get(16, '-'):>8} {comps.get(32, '-'):>8} "
              f"{comps.get(64, '-'):>8}")

    # Summary table for full metrics
    if args.mode in ("full", "all"):
        has_full = {n: m["full"] for n, m in report["models"].items() if m["full"]}
        if has_full:
            print(f"\n{'='*120}")
            print("FULL QUALITY METRICS (representative)")
            print(f"{'='*120}")
            hdr = f"{'Model':30s} {'R':>3} {'Comp':>8} {'Recon%':>8} {'DA_Err':>8} {'EOS@10':>7} {'EOS@50':>7} {'Trust':>7} {'Cont':>7}"
            print(hdr)
            print("-" * len(hdr))
            for mname, mdata in sorted(has_full.items()):
                for r_str, metrics in sorted(mdata.items()):
                    if "error" in metrics:
                        continue
                    print(f"{mname[:28]:30s} {r_str:>3} "
                          f"{metrics['compression_ratio']:>7.1f}x "
                          f"{metrics['reconstruction_error_pct']:>7.3f}% "
                          f"{metrics['distribution_aware_error']:>8.4f} "
                          f"{metrics['eigenspace_overlap_k10']:>7.4f} "
                          f"{metrics['eigenspace_overlap_k50']:>7.4f} "
                          f"{metrics['trustworthiness']:>7.4f} "
                          f"{metrics['continuity']:>7.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
