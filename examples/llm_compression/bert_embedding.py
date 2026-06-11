"""BERT Embedding Compression Example.

Demonstrates:
  1. Creating a TR embedding from scratch
  2. Loading a HuggingFace BERT model and replacing its embedding layer
"""

import torch
from tensor_ring_decomposition import TensorRingEmbedding


def from_scratch():
    """Create a TR embedding from scratch for the BERT vocab/embedding dims."""
    vocab_size = 30522
    embedding_dim = 768
    rank = 8

    print("=== From scratch ===")
    tr_emb = TensorRingEmbedding(
        vocab_size, embedding_dim,
        rank=rank, ring_components=4, gauge_fix="left",
    )

    dense_params = vocab_size * embedding_dim
    print(f"Dense params:    {dense_params:>12,}")
    print(f"TR params:       {tr_emb.num_parameters:>12,}")
    print(f"Compression:     {tr_emb.compression_ratio:>12.1f}x")

    indices = torch.randint(0, vocab_size, (4, 16))
    output = tr_emb(indices)
    print(f"Input shape:     {indices.shape}")
    print(f"Output shape:    {output.shape}")

    random_matrix = torch.randn(vocab_size, embedding_dim)
    error = tr_emb.reconstruction_error(random_matrix)
    print(f"Recon error (vs random): {error:.6f}")

    tr_emb.to_eval_mode()
    with torch.no_grad():
        output_eval = tr_emb(indices)
    print(f"Eval output shape: {output_eval.shape}")

    cfg = tr_emb.config()
    print(f"Vocab factor sizes: {cfg['vocab_factor_sizes']}")
    print(f"Emb factor sizes:   {cfg['emb_factor_sizes']}")
    return tr_emb


def replace_in_bert():
    """Load a pretrained BERT model and replace its embedding with TR."""
    print("\n=== Replace in BERT model ===")
    try:
        from transformers import BertModel
        from tensor_ring_decomposition.integrations.huggingface import (
            HuggingFaceTensorRingEmbedding,
        )
    except ImportError:
        print("transformers not installed — skipping")
        return

    model = BertModel.from_pretrained("bert-base-uncased")
    original_emb = model.get_input_embeddings()
    print(f"Original BERT embedding: {original_emb.weight.shape}")

    # Create TR embedding initialized from the pretrained weights
    tr_emb = TensorRingEmbedding.from_pretrained(
        original_emb.weight.data, rank=8,
    )
    print(f"TR compression: {tr_emb.compression_ratio:.1f}x")

    recon_error = tr_emb.reconstruction_error(original_emb.weight.data)
    print(f"Reconstruction error: {recon_error:.6f}")

    # Replace in model
    model = HuggingFaceTensorRingEmbedding.replace_in_model(model, tr_emb)
    replaced = model.get_input_embeddings()
    print(f"Replaced embedding type: {type(replaced).__name__}")

    # Forward pass through the modified model
    dummy = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        out = model(input_ids=dummy)
    print(f"Model output shape: {out.last_hidden_state.shape}")


if __name__ == "__main__":
    from_scratch()
    replace_in_bert()
