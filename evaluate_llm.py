"""Full LLM evaluation: compress embedding, measure perplexity before/after."""

import argparse
import json
import math
import sys
import time
import torch

sys.path.insert(0, ".")

from tensor_ring_decomposition.compress import compress
from tensor_ring_decomposition.core.embedding import TensorRingEmbedding
from tensor_ring_decomposition.loaders.loaders import load_embedding_matrix
from tensor_ring_decomposition.integrations.huggingface import HuggingFaceTensorRingEmbedding


def load_model(model_name: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, texts, device, max_length=128, stride=64):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encodings = tokenizer(text, return_tensors="pt", truncation=True,
                                  max_length=max_length).to(device)
            input_ids = encodings.input_ids[0]
            seq_len = input_ids.size(0)

            if seq_len <= 1:
                continue

            nll_sum = 0.0
            n_tokens = 0

            for begin in range(0, seq_len - 1, stride):
                end = min(begin + max_length, seq_len)
                chunk = input_ids[begin:end].unsqueeze(0)
                labels = chunk.clone()
                outputs = model(input_ids=chunk, labels=labels)
                loss = outputs.loss
                nll_sum += loss.item() * (end - begin - 1)
                n_tokens += (end - begin - 1)

            if n_tokens > 0:
                total_loss += nll_sum
                total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM with TR-compressed embedding")
    parser.add_argument("model", type=str, help="HF model name (e.g., gpt2, facebook/opt-125m)")
    parser.add_argument("--rank", type=int, default=8, help="TR rank")
    parser.add_argument("--texts", type=str, nargs="+", default=[
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is a field of artificial intelligence.",
        "The theory of relativity was developed by Albert Einstein.",
        "Machine learning models can learn complex patterns from data.",
        "The capital of France is Paris and it is known for the Eiffel Tower.",
    ])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Model: {args.model}")
    print(f"Rank: {args.rank}")
    print(f"Device: {device}\n")

    # 1. Load full model + compute baseline perplexity
    print("Loading model and tokenizer...")
    t0 = time.monotonic()
    model, tokenizer = load_model(args.model, device)
    load_time = time.monotonic() - t0
    print(f"  Load time: {load_time:.1f}s")

    orig_emb = model.get_input_embeddings()
    V, D = orig_emb.weight.shape
    dense_params = V * D
    print(f"  Embedding: {V}x{D}, {dense_params:,} params")

    print("\nComputing baseline perplexity...")
    t0 = time.monotonic()
    base_ppl = compute_perplexity(model, tokenizer, args.texts, device)
    base_time = time.monotonic() - t0
    print(f"  Baseline PPL: {base_ppl:.4f} ({base_time:.1f}s)")

    # 2. Create TR embedding
    print(f"\nCompressing embedding (rank={args.rank})...")
    t0 = time.monotonic()
    tr_emb = TensorRingEmbedding.from_pretrained(
        orig_emb.weight.data, rank=args.rank, ring_components=4,
    )
    comp_time = time.monotonic() - t0
    comp_ratio = tr_emb.compression_ratio
    tr_params = tr_emb.num_parameters
    recon_err = tr_emb.reconstruction_error(orig_emb.weight.data)
    print(f"  Compression: {comp_ratio:.1f}x ({tr_params:,} vs {dense_params:,} dense)")
    print(f"  Recon error: {recon_err*100:.2f}%")
    print(f"  Compress time: {comp_time:.1f}s")

    # 3. Replace in model
    print("\nReplacing embedding in model...")
    model = HuggingFaceTensorRingEmbedding.replace_in_model(model, tr_emb)
    print(f"  New embedding type: {type(model.get_input_embeddings()).__name__}")

    # 4. Measure compressed perplexity
    print("\nComputing compressed perplexity...")
    t0 = time.monotonic()
    comp_ppl = compute_perplexity(model, tokenizer, args.texts, device)
    comp_time_eval = time.monotonic() - t0

    # 5. Report
    ppl_change = ((comp_ppl - base_ppl) / base_ppl) * 100

    print(f"\n{'='*55}")
    print(f"  RESULTS: {args.model} @ rank={args.rank}")
    print(f"{'='*55}")
    print(f"  {'Metric':<30} {'Before':>10} {'After':>10} {'Change':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Parameters':<30} {dense_params:>10,} {tr_params:>10,} {comp_ratio:>8.1f}x")
    print(f"  {'Perplexity':<30} {base_ppl:>10.4f} {comp_ppl:>10.4f} {ppl_change:>+9.2f}%")
    print(f"  {'Recon Error':<30} {'':>10} {recon_err*100:>9.2f}%")
    print(f"  {'Compression Ratio':<30} {'':>10} {comp_ratio:>8.1f}x")

    result = {
        "model": args.model,
        "rank": args.rank,
        "vocab_size": V,
        "embedding_dim": D,
        "dense_parameters": dense_params,
        "tr_parameters": tr_params,
        "compression_ratio": comp_ratio,
        "reconstruction_error_pct": round(recon_err * 100, 2),
        "baseline_perplexity": round(base_ppl, 4),
        "compressed_perplexity": round(comp_ppl, 4),
        "perplexity_change_pct": round(ppl_change, 2),
        "compression_time_s": round(comp_time, 1),
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
