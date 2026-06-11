"""Benchmark script comparing TR embedding vs dense.

Usage:
    python examples/run_benchmark.py                          # CPU
    python examples/run_benchmark.py --device cuda            # GPU
    python examples/run_benchmark.py --device cuda --iters 1000
"""

import argparse
import time
import torch
from tensor_ring_decomposition import TensorRingEmbedding


def benchmark(vocab_size, dim, rank, device, num_iters):
    tr_emb = TensorRingEmbedding(vocab_size, dim, rank=rank).to(device)
    dense_emb = torch.nn.Embedding(vocab_size, dim).to(device)

    indices = torch.randint(0, vocab_size, (32, 128)).to(device)

    # Warmup
    for _ in range(10):
        tr_emb(indices)
        dense_emb(indices)

    if device != "cpu":
        torch.cuda.synchronize()

    # Training mode
    tr_emb.train_mode()
    start = time.time()
    for _ in range(num_iters):
        tr_emb(indices)
    if device != "cpu":
        torch.cuda.synchronize()
    tr_train_time = time.time() - start

    # Eval mode
    tr_emb.to_eval_mode()
    start = time.time()
    for _ in range(num_iters):
        tr_emb(indices)
    if device != "cpu":
        torch.cuda.synchronize()
    tr_eval_time = time.time() - start

    # Dense
    start = time.time()
    for _ in range(num_iters):
        dense_emb(indices)
    if device != "cpu":
        torch.cuda.synchronize()
    dense_time = time.time() - start

    print(f"V={vocab_size}, D={dim}, R={rank}")
    print(f"  Dense:      {dense_time:.3f}s")
    print(f"  TR train:   {tr_train_time:.3f}s ({tr_train_time/dense_time:.2f}x)")
    print(f"  TR eval:    {tr_eval_time:.3f}s ({tr_eval_time/dense_time:.2f}x)")
    print(f"  Compression: {tr_emb.compression_ratio:.1f}x")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=1000 if torch.cuda.is_available() else 100)
    args = parser.parse_args()

    for rank in [4, 8, 16, 32]:
        benchmark(50000, 768, rank, args.device, args.iters)
