"""Comprehensive model testing script for Tensor Ring Embedding.

Tests compression across all registered model profiles.
Usage:
    python examples/test_all_models.py --list        # List all models
    python examples/test_all_models.py --quick       # Quick test (small models only)
    python examples/test_all_models.py --bert        # Test BERT only
    python examples/test_all_models.py --all         # Test all models (requires download)
    python examples/test_all_models.py --profile bert-base-uncased --rank 8
"""

import argparse
import importlib
import time
from typing import List, Optional

import torch

from tensor_ring_decomposition import ModelRegistry, ModelProfile, TensorRingEmbedding
from tensor_ring_decomposition.core.embedding import AutotuneResult


def check_transformers() -> bool:
    try:
        import transformers
        return True
    except ImportError:
        return False


def test_profile(
    profile: ModelProfile,
    rank: int,
    init_method: str = "svd",
    device: Optional[torch.device] = None,
    timeout_s: float = 120.0,
) -> dict:
    """Test a single model profile with Tensor Ring compression.

    Args:
        profile: ModelProfile to test.
        rank: TR rank.
        init_method: "svd" (full training) or "tr_svd" (fast).
        device: torch device.
        timeout_s: Maximum time in seconds.

    Returns:
        Dict of test results.
    """
    result = {
        "name": profile.name,
        "vocab_size": profile.vocab_size,
        "embedding_dim": profile.embedding_dim,
        "rank": rank,
        "init_method": init_method,
        "compression_ratio": 0.0,
        "reconstruction_error": 0.0,
        "init_time_s": 0.0,
        "forward_time_ms": 0.0,
        "status": "unknown",
        "error": "",
    }

    try:
        # Generate synthetic matrix if HF not available or testing generic
        matrix = torch.randn(profile.vocab_size, profile.embedding_dim, device=device)

        t0 = time.time()
        emb = TensorRingEmbedding.from_pretrained(
            matrix, rank=rank, ring_components=profile.ring_components,
            init_method=init_method, device=device,
        )
        init_time = time.time() - t0

        result["compression_ratio"] = emb.compression_ratio
        result["init_time_s"] = round(init_time, 2)

        # Forward pass timing
        batch_indices = torch.randint(0, min(profile.vocab_size, 10000), (4, 128), device=device)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(10):
                emb(batch_indices)
        forward_time = (time.time() - t0) / 10 * 1000

        result["forward_time_ms"] = round(forward_time, 2)
        result["reconstruction_error"] = round(emb.reconstruction_error(matrix), 4)
        result["status"] = "passed"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        if "out of memory" in str(e).lower():
            result["status"] = "oom"

    return result


def autotune_profile(
    profile: ModelProfile,
    target_compression: float = 10.0,
    device: Optional[torch.device] = None,
) -> AutotuneResult:
    """Autotune rank for a model profile."""
    matrix = torch.randn(profile.vocab_size, profile.embedding_dim, device=device)
    return TensorRingEmbedding.autotune(
        matrix, ring_components=profile.ring_components,
        target_compression=target_compression, verbose=False,
    )


def print_table(rows: List[dict], title: str = ""):
    """Print results as a formatted table."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"  {title}")
        print(f"{'=' * 80}")

    header = f"{'Model':30s} {'V':>7s} {'D':>5s} {'R':>3s} {'Comp':>6s} {'Error':>7s} {'Init(s)':>8s} {'Fwd(ms)':>8s} {'Status':>10s}"
    print(header)
    print("-" * 80)

    for r in rows:
        name = r["name"][:28]
        comp = f"{r['compression_ratio']:.0f}x" if r["compression_ratio"] else "-"
        err = f"{r['reconstruction_error']:.4f}" if r["reconstruction_error"] else "-"
        init = f"{r['init_time_s']:.1f}" if r["init_time_s"] else "-"
        fwd = f"{r['forward_time_ms']:.1f}" if r["forward_time_ms"] else "-"
        print(
            f"{name:30s} {r['vocab_size']:>7d} {r['embedding_dim']:>5d} "
            f"{r['rank']:>3d} {comp:>6s} {err:>7s} {init:>8s} {fwd:>8s} {r['status']:>10s}"
        )


def main():
    parser = argparse.ArgumentParser(description="Test all model profiles with TR compression")
    parser.add_argument("--list", action="store_true", help="List all registered models")
    parser.add_argument("--quick", action="store_true", help="Test small models only (<50K vocab)")
    parser.add_argument("--all", action="store_true", help="Test all registered profiles")
    parser.add_argument("--bert", action="store_true", help="Test BERT base only")
    parser.add_argument("--profile", type=str, default=None, help="Test a specific profile by name")
    parser.add_argument("--rank", type=int, default=8, help="TR rank (default: 8)")
    parser.add_argument("--init", type=str, default="svd", choices=["svd", "tr_svd", "uniform"],
                        help="Init method (default: svd)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.list:
        print(ModelRegistry.summary())
        return

    profiles_to_test: List[ModelProfile] = []

    if args.profile:
        prof = ModelRegistry.get(args.profile)
        if prof is None:
            print(f"Profile '{args.profile}' not found.")
            print(f"Available: {', '.join(p.name for p in ModelRegistry.list_all()[:10])}...")
            return
        profiles_to_test = [prof]
    elif args.bert:
        prof = ModelRegistry.get("bert-base-uncased")
        if prof:
            profiles_to_test = [prof]
    elif args.quick:
        all_profiles = ModelRegistry.list_all()
        profiles_to_test = [p for p in all_profiles if p.vocab_size < 50000 and p.embedding_dim <= 1024]
        if not profiles_to_test:
            profiles_to_test = all_profiles[:5]
    elif args.all:
        profiles_to_test = ModelRegistry.list_all()
    else:
        print(ModelRegistry.summary())
        print("\nUse --bert, --quick, --all, or --profile <name> to test.")
        return

    print(f"Testing {len(profiles_to_test)} profiles with rank={args.rank}, init={args.init}...")
    results = []
    passed = 0
    for i, prof in enumerate(profiles_to_test):
        print(f"  [{i+1}/{len(profiles_to_test)}] {prof.name}... ", end="", flush=True)
        r = test_profile(prof, rank=args.rank, init_method=args.init, device=device)
        results.append(r)
        status = "OK" if r["status"] == "passed" else f"FAIL({r['status']})"
        print(f"{status} (comp={r['compression_ratio']:.0f}x, err={r['reconstruction_error']:.4f})")
        if r["status"] == "passed":
            passed += 1

    print_table(results, f"Results: {passed}/{len(profiles_to_test)} passed (R={args.rank}, init={args.init})")


if __name__ == "__main__":
    main()
