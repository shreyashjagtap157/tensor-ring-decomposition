Model Registry
==============

The library includes 63+ built-in model profiles covering popular architectures.
Each profile specifies the vocabulary size, embedding dimension, and recommended rank.

.. code-block:: python

   from tensor_ring_decomposition import list_models, ModelRegistry

   # List all models
   print(list_models())

   # Filter by family
   print(list_models(family="llama"))

   # Look up a profile
   profile = ModelRegistry.get("bert-base-uncased")
   print(profile.default_rank)

   # Find rank for a target compression
   rank = profile.rank_for_compression(50)

Architecture families
---------------------

* BERT (base, large, tiny, mini, small, medium)
* RoBERTa (base, large)
* DeBERTa (base, large, xlarge, v2, v3)
* ALBERT (base, large, xlarge, xxlarge)
* GPT-2 (small, medium, large, xl)
* LLaMA (1, 2, 3, 3.1, 3.2)
* Mistral (v0.1, v0.2, v0.3, Nemo)
* Falcon (7B, 40B, 180B)
* Gemma (2B, 7B)
* Phi (1, 1.5, 2, 3)
* BLOOM (560M, 1.1B, 1.7B, 3B, 7.1B, 176B)
* T5 (small, base, large, 3B, 11B)
* BART (base, large)
* LayoutLM (base, large)
* Electra (small, base, large)
* XLM-RoBERTa (base, large)
* Longformer (base, large)
* BigBird (base, large)
* Nystromformer (base, large)
* GPT-J, GPT-NeoX, Pythia, StarCoder, DBRX
* Command R, Jamba, MPT, Yi, DeepSeek, StableLM
* Zephyr, Solar, XVERSE, Qwen 2.5
* Granite (3B, 8B, 20B, 34B)
