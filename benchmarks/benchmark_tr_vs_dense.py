"""Benchmark Tensor Ring embedding vs dense nn.Embedding.

Measures:
- Compression ratio vs rank
- Forward pass latency (train and eval modes)
- Memory usage (parameters)
- Reconstruction error vs rank
- Scaling with vocab size and embedding dim
"""

import argparse
import math
import platform
import time
import torch
import torch.nn as nn

from tensor_ring_decomposition import TensorRingEmbedding

def fmt(n: float) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:.2f}"

def benchmark_latency(emb, indices, n_runs=100, warmup=10):
    """Measure forward pass latency in ms."""
    for _ in range(warmup):
        emb(indices)
    if emb.training:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_runs):
        emb(indices)
    if emb.training:
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) * 1000 / n_runs
    return elapsed

def benchmark_shape(V, D, ranks, device):
    """Benchmark a single (V, D) shape across ranks."""
    rows = []
    indices = torch.randint(0, min(V, 256), (4, 128), device=device)

    for rank in ranks:
        try:
            emb = TensorRingEmbedding(V, D, rank=rank, device=device)
        except Exception as e:
            rows.append({"V": V, "D": D, "rank": rank, "error": str(e)})
            continue

        dense_emb = nn.Embedding(V, D, device=device)
        dense_params = V * D
        tr_params = emb.num_parameters
        comp_ratio = emb.compression_ratio

        emb.train_mode()
        train_latency = benchmark_latency(emb, indices)

        with torch.no_grad():
            emb.to_eval_mode()
            eval_latency = benchmark_latency(emb, indices)

        with torch.no_grad():
            dense_latency = benchmark_latency(dense_emb, indices)

        recon_error = None
        if V * D <= 5_000_000:
            try:
                dense_weight = dense_emb.weight.data.clone()
                recon_error = emb.reconstruction_error(dense_weight)
            except Exception:
                recon_error = None

        rows.append({
            "V": V, "D": D, "rank": rank,
            "dense_params": dense_params, "tr_params": tr_params,
            "compression": f"{comp_ratio:.1f}x",
            "train_ms": f"{train_latency:.3f}",
            "eval_ms": f"{eval_latency:.3f}",
            "dense_ms": f"{dense_latency:.3f}",
            "speedup_train": f"{dense_latency/max(train_latency,1e-9):.1f}x",
            "speedup_eval": f"{dense_latency/max(eval_latency,1e-9):.1f}x",
            "recon_error": f"{recon_error:.4f}" if recon_error is not None else "N/A",
            "dense_params_fmt": fmt(dense_params),
            "tr_params_fmt": fmt(tr_params),
        })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="CSV output path")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print()

    configs = [
        (50000, 64, [2, 4, 6, 8], "small vocab, small dim"),
        (50000, 128, [2, 4, 6, 8, 12], "small vocab, medium dim"),
        (50000, 256, [2, 4, 6, 8, 12], "small vocab, large dim"),
        (100000, 64, [2, 4, 6, 8], "medium vocab, small dim"),
        (100000, 128, [2, 4, 6, 8, 12], "medium vocab, medium dim"),
        (500000, 64, [2, 4, 6, 8], "large vocab, small dim"),
        (500000, 128, [2, 4, 6, 8, 12], "large vocab, medium dim"),
    ]

    all_rows = []

    for V, D, ranks, desc in configs:
        print(f"── {desc}: V={fmt(V)}, D={D} ──")
        rows = benchmark_shape(V, D, ranks, device)
        all_rows.extend(rows)

        header = f"{'Rank':>5} | {'Params':>9} | {'Comp':>6} | {'Train(ms)':>9} | {'Eval(ms)':>8} | {'Dense(ms)':>9} | {'Speedup T':>9} | {'Speedup E':>9} | {'MSE':>7}"
        print(header)
        print("-" * len(header))
        for r in rows:
            if "error" in r:
                print(f"  {r['rank']:>3}  ERROR: {r['error']}")
            else:
                print(f"  {r['rank']:>3}  {r['tr_params_fmt']:>9}  {r['compression']:>6}  {r['train_ms']:>9}  {r['eval_ms']:>8}  {r['dense_ms']:>9}  {r['speedup_train']:>9}  {r['speedup_eval']:>9}  {r['recon_error']:>7}")
        print()

    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
        print(f"Wrote {len(all_rows)} results to {args.output}")

if __name__ == "__main__":
    main()
