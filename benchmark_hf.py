"""CLI benchmark: compress any HuggingFace model's embedding layer."""

import argparse
import json
import sys
import time
import torch

sys.path.insert(0, ".")

from tensor_ring_decomposition.compress import compress
from tensor_ring_decomposition.core.embedding import TensorRingEmbedding
from tensor_ring_decomposition.loaders.loaders import load_embedding_matrix


def main():
    parser = argparse.ArgumentParser(description="Compress HF model embeddings via TR decomposition")
    parser.add_argument("model", type=str, help="HuggingFace model name (e.g., gpt2, bert-base-uncased)")
    parser.add_argument("--ranks", type=str, default="4,8,16,24,32,48",
                        help="Comma-separated ranks to test (default: 4,8,16,24,32,48)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results (default: print to stdout)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)
    ranks = [int(r) for r in args.ranks.split(",")]

    print(f"Model: {args.model}")
    print(f"Ranks: {ranks}")
    print(f"Device: {device}\n")

    # 1. Load embedding matrix
    print("Loading embedding matrix...")
    t0 = time.monotonic()
    matrix = load_embedding_matrix(args.model, device=device)
    V, D = matrix.shape
    dense_params = V * D
    print(f"  Shape: {V}x{D}, Dense params: {dense_params:,}")
    print(f"  Load time: {time.monotonic() - t0:.1f}s")

    # 2. SVD baseline
    print("Computing SVD baseline...")
    t0 = time.monotonic()
    U, S, Vh = torch.linalg.svd(matrix.to(torch.float32), full_matrices=False)
    total_var = (S ** 2).sum()
    print(f"  SVD time: {time.monotonic() - t0:.1f}s")
    print(f"  Top-5 singular values: {[round(s.item(), 1) for s in S[:5]]}")

    results = []
    for rank in ranks:
        print(f"\n  --- Rank={rank} ---")
        t_comp = time.monotonic()
        emb = compress(matrix, rank=rank, ring_components=4, init_method="svd", device=device)
        comp_time = time.monotonic() - t_comp

        comp_ratio = emb.compression_ratio
        num_params = emb.num_parameters
        recon_err = emb.reconstruction_error(matrix)
        eos_10 = emb.eigenspace_overlap_score(matrix, k=10)
        eos_50 = emb.eigenspace_overlap_score(matrix, k=min(50, D))
        trust = emb.trustworthiness(matrix, n_neighbors=15, metric="euclidean", sample_size=2000)
        cont = emb.continuity(matrix, n_neighbors=15, metric="euclidean", sample_size=2000)

        svd_err = 1.0 - (S[:rank] ** 2).sum() / total_var

        row = {
            "rank": rank,
            "compression_ratio": round(comp_ratio, 1),
            "num_parameters": num_params,
            "pct_of_dense": round(num_params / dense_params * 100, 2),
            "reconstruction_error_pct": round(recon_err * 100, 2),
            "svd_estimated_error_pct": round(svd_err.item() * 100, 1),
            "eigenspace_overlap_k10": round(eos_10, 4),
            "eigenspace_overlap_k50": round(eos_50, 4),
            "trustworthiness": round(trust, 4),
            "continuity": round(cont, 4),
            "compression_time_s": round(comp_time, 1),
        }
        results.append(row)
        print(f"    Comp: {comp_ratio:.1f}x | Params: {num_params:,} ({row['pct_of_dense']:.2f}%)")
        print(f"    Recon error: {recon_err*100:.1f}% | EOS@10: {eos_10:.4f}")
        print(f"    Trust: {trust:.4f} | Cont: {cont:.4f}")

    output = {
        "model": args.model,
        "vocab_size": V,
        "embedding_dim": D,
        "dense_parameters": dense_params,
        "results": results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
