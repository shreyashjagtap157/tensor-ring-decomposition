# Tensor Ring Decomposition for Embeddings

Tensor Ring Decomposition (TRD) is a Python/PyTorch library for representing large embedding tables with tensor-ring factors. The goal is to reduce embedding parameter and storage costs while making the quality, latency and memory trade-offs measurable.

> **Repository status — 22 October 2026:** the code, tests, examples and benchmark material are present, but release readiness is not established. The most recent inspected Tests workflow ended in `startup_failure`, and recent security workflow runs failed. See [Actions](https://github.com/shreyashjagtap157/tensor-ring-decomposition/actions) before relying on a candidate revision.

## Capabilities

The source and project documentation cover:

- `TensorRingEmbedding`, intended as a compressed alternative to `torch.nn.Embedding`.
- Rank-, target-parameter- and target-compression-based configuration.
- ALS/SVD and other initialization paths.
- Pretrained embedding conversion and optional Hugging Face integration.
- Quantization-related paths, export utilities and optional GGUF tooling.
- Examples, tests, documentation and benchmark harnesses.

**Compression and quality claims are configuration-dependent.** A theoretical parameter ratio is not a guarantee of equivalent model quality, lower wall-clock latency, lower total training cost or production suitability. Reproduce the benchmark configuration and report quality metrics alongside parameter counts.

## Requirements

The package metadata specifies Python 3.9+, PyTorch 2.0+, safetensors 0.4+ and NumPy 1.24+. Optional integrations may need additional dependencies.

## Development setup

From a clean checkout:

```bash
python -m venv .venv
# Activate the environment for your shell.
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m pytest
```

Optional extras are declared in `pyproject.toml`; install only the integrations needed for your work. The PyPI publication state was not verified in this repository audit, so use the repository checkout when you need to test the current source revision.

## Benchmarks and evidence

Benchmark material is in `BENCHMARK_REPORT.md` and `benchmarks/`. The checked-in report describes 60 model profiles, while full quality metrics are reported for a smaller subset. Re-run experiments in a clean environment before relying on those numbers.

The package metadata currently reports version `0.4.0`, while the benchmark report describes version `0.3.0`. Keep version, benchmark artifacts and release notes aligned as part of release preparation.

## Current engineering priorities

1. Diagnose the test workflow startup failure and make the test job execute reliably.
2. Investigate and fix the failing security workflow runs.
3. Reproduce compression and quality benchmarks from committed scripts/configuration.
4. Test serialization, pretrained conversion, quantization and export paths independently.
5. Tag and publish a release only after clean installation, tests and security checks pass.

## Repository layout

- `tensor_ring_decomposition/` — Python package.
- `tests/` — automated tests.
- `benchmarks/` — benchmark tooling.
- `examples/` — examples.
- `docs/` — documentation.
- `pyproject.toml` — package metadata and optional dependencies.
- `BENCHMARK_REPORT.md` — existing benchmark report.
- `CHANGELOG.md` — release history.

## License

MIT. See [LICENSE](LICENSE).
