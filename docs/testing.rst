Testing
=======

.. code-block:: bash

   pip install pytest pytest-cov
   pytest tests/ -v

The test suite includes 300+ tests covering:

* Forward pass for diverse embedding shapes
* All init methods (uniform, normal, kaiming, SVD, TR-SVD, distribution-aware)
* Numerical stability across all configurations
* Train/eval convention compliance
* Input validation (OOB, negative indices, empty tensors)
* Spectral regularization
* Gauge fixing (none, left, right, both, interval)
* Compression ratios (2x to 1000x)
* Padding index edge cases
* All init methods with and without matrices
* Distribution-aware metrics
* Eigenspace overlap score
* Quantization roundtrip
* Full serialization with config and manifest verification
* Model registry completeness
* DDP safety
* Autotune (compression, params, MSE)
* Compress function (tensor, profile, autotune)
* Reset parameters and cache management
* Weight property (nn.Embedding compatibility)
* High-rank stability (16-48)
* Memory efficiency (sub-linear scaling)
* FromProfile builder
* Edge inputs (scalar, empty, single token)
* Reproducibility (deterministic init and forward)
* Monitoring (CompressionTracker, QualityGate, TensorRingCallback)
* Mixed precision (bf16 forward)
* Ring closure equivalence
* Factorization edge cases
* ValidateCompatibility
* HuggingFace integration
* GPU support

Run a specific test file:

.. code-block:: bash

   pytest tests/test_embedding.py -v
   pytest tests/test_enterprise.py -v
