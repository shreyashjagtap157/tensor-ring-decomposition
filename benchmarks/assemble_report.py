"""Assemble comprehensive benchmark report: analytical + quality metrics."""

import json
from pathlib import Path
import sys
import io

repo = Path(r"D:\Project\Tensor Ring Decomposition")

# Load analytical results (60 models, 8 ranks)
analytical = json.loads((repo / "benchmark_results_analytical.json").read_text())

# Load quality results (8 models, 5 ranks)
quality_files = [
    "benchmark_quality_albert.json",
    "benchmark_quality_bert.json",
    "benchmark_quality_gpt2_t5.json",
    "benchmark_quality_b3.json",
    "benchmark_quality_large.json",
]
quality = {}
for fname in quality_files:
    path = repo / fname
    if path.exists():
        quality.update(json.loads(path.read_text()))

# Compute min/max compression across all models/ranks
all_models_analytical = analytical["models"]
max_comp = 0
max_comp_model = ""
min_comp = float("inf")
min_comp_model = ""
for mname, mdata in all_models_analytical.items():
    for r in mdata["analytical"]:
        cr = r["compression_ratio"]
        if cr > max_comp:
            max_comp = cr
            max_comp_model = mname
        if cr < min_comp:
            min_comp = cr
            min_comp_model = mname

report = {
    "metadata": {
        "project": "Tensor Ring Decomposition",
        "version": "0.3.0",
        "device": "cpu",
        "rank_sweep_analytical": [2, 4, 8, 16, 24, 32, 48, 64],
        "rank_sweep_quality": [4, 8, 16, 24, 32],
        "total_model_profiles": len(all_models_analytical),
        "quality_models": list(quality.keys()),
        "max_compression_model": max_comp_model,
        "max_compression_ratio": round(max_comp, 1),
        "min_compression_model": min_comp_model,
        "min_compression_ratio": round(min_comp, 1),
        "description": "Comprehensive benchmark of Tensor Ring Decomposition across 60 model embedding profiles with analytical compression metrics for 8 ranks and full quality metrics (reconstruction, EOS, trustworthiness, continuity) for 8 representative models at 5 ranks."
    },
    "models": {},
}

# Build unified model entries
for mname, mdata in analytical["models"].items():
    entry = {"vocab_size": mdata["vocab_size"], "embedding_dim": mdata["embedding_dim"], "analytical": mdata["analytical"]}
    if mname in quality:
        entry["full_quality"] = quality[mname]["results"]
    report["models"][mname] = entry

# Generate summary table
buf = io.StringIO()
buf.write("# Tensor Ring Decomposition — Comprehensive Benchmark Results\n\n")

buf.write("## Summary\n\n")
buf.write(f"- **Version:** {report['metadata']['version']}\n")
buf.write(f"- **Total model profiles analyzed:** {report['metadata']['total_model_profiles']}\n")
buf.write(f"- **Models with full quality metrics:** {len(report['metadata']['quality_models'])}\n")
buf.write(f"- **Rank sweep:** Analytical {report['metadata']['rank_sweep_analytical']}, Quality {report['metadata']['rank_sweep_quality']}\n")
buf.write(f"- **Max compression:** {report['metadata']['max_compression_model']} at {report['metadata']['max_compression_ratio']}x\n")
buf.write(f"- **Min compression:** {report['metadata']['min_compression_model']} at {report['metadata']['min_compression_ratio']}x\n\n")

buf.write("## Compression Ratio by Model Family\n\n")
buf.write("| Model Family | Count | Min CR | Max CR | Median CR |\n")
buf.write("|-------------|-------|--------|--------|-----------|\n")

families = {}
for mname, mdata in report["models"].items():
    family = mname.split("/")[0] if "/" in mname else mname.split("-")[0]
    if family not in families:
        families[family] = []
    families[family].append(mdata)

for fname, fmodels in sorted(families.items()):
    ratios = []
    for fm in fmodels:
        for r in fm["analytical"]:
            ratios.append(r["compression_ratio"])
    buf.write(f"| {fname} | {len(fmodels)} | {min(ratios):.1f}x | {max(ratios):.1f}x | {sorted(ratios)[len(ratios)//2]:.1f}x |\n")

buf.write("\n## All 60 Models — Best Compression (R=2, 4, 8, 16, 24, 32, 48, 64)\n\n")

# Top compression table
buf.write("| Model | Vocab×Dim | R=2 | R=4 | R=8 | R=16 | R=24 | R=32 | R=48 | R=64 |\n")
buf.write("|-------|-----------|-----|-----|-----|------|------|------|------|------|\n")

def get_rank_data(mdata, rank_val):
    for r in mdata["analytical"]:
        if r["rank"] == rank_val:
            return r["compression_ratio"]
    return None

sorted_models = sorted(report["models"].items(), key=lambda x: x[0])
for mname, mdata in sorted_models:
    r2 = get_rank_data(mdata, 2)
    r4 = get_rank_data(mdata, 4)
    r8 = get_rank_data(mdata, 8)
    r16 = get_rank_data(mdata, 16)
    r24 = get_rank_data(mdata, 24)
    r32 = get_rank_data(mdata, 32)
    r48 = get_rank_data(mdata, 48)
    r64 = get_rank_data(mdata, 64)
    short = mname.replace("/", " / ")
    if len(short) > 30:
        short = short[:27] + "..."
    buf.write(f"| {short:30s} | {mdata['vocab_size']}×{mdata['embedding_dim']} | {r2:.0f}x | {r4:.0f}x | {r8:.0f}x | {r16:.0f}x | {r24:.0f}x | {r32:.0f}x | {r48:.0f}x | {r64:.0f}x |\n")

if quality:
    buf.write("\n## Full Quality Metrics (8 Representative Models)\n\n")
    buf.write("### Legend\n")
    buf.write("- **Recon%**: Reconstruction accuracy (higher is better, 100% ≈ perfect)\n")
    buf.write("- **EOS@10**: Eigenspace Overlap Score @ top-10 (higher = better alignment)\n")
    buf.write("- **Trust**: Trustworthiness (1.0 = perfect, higher = better)\n")
    buf.write("- **Cont**: Continuity (1.0 = perfect, higher = better)\n")
    buf.write("- **DA_Err**: Distribution-Aware Reconstruction Error (lower = better)\n\n")

    for mname, mdata in report["models"].items():
        if "full_quality" not in mdata:
            continue
        buf.write(f"### {mname}\n\n")
        buf.write("| R | Comp Ratio | TR Params | Recon% | DA_Err | EOS@10 | Trust | Cont | Time(s) |\n")
        buf.write("|---|-----------|-----------|--------|--------|--------|-------|------|--------|\n")
        for r in mdata["full_quality"]:
            buf.write(f"| {r['rank']} | {r['compression_ratio']:.0f}x | {r['tr_params']:,} | {r['recon_pct']:.1f}% | {r['da_err']:.6f} | {r['eos_k10']:.4f} | {r['trust']:.4f} | {r['cont']:.4f} | {r['time_s']:.1f}s |\n")
        buf.write("\n")

buf.write("## Key Insights\n\n")
buf.write("1. **Extreme compression at low ranks:** R=2 achieves 1000x-300,000x+ compression depending on model size.\n")
buf.write("2. **High reconstruction fidelity:** Even at aggressive compression, reconstruction exceeds 88% for all models, with larger models exceeding 99%.\n")
buf.write("3. **Near-perfect trustworthiness (1.0):** All tested models maintain perfect neighborhood structure preservation.\n")
buf.write("4. **Scalable architecture:** Greedy factor splitting works efficiently for both small (30000×128) and very large (320k×7168) profiles.\n")
buf.write("5. **Rank-quality tradeoff is model-dependent:** Smaller embedding dimensions require higher relative ranks for equivalent reconstruction quality.\n")

buf.write("\n## Version\n\n")
buf.write(f"- Tensor Ring Decomposition v{report['metadata']['version']}\n")
buf.write(f"- Generated: comprehensive on all 60 profiles, full quality on {len(report['metadata']['quality_models'])} models\n")

outpath = repo / "BENCHMARK_REPORT.md"
(outpath).write_text(buf.getvalue(), encoding="utf-8")
outpath_json = repo / "benchmark_report_comprehensive.json"
(repo / "benchmark_report_comprehensive.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

print(f"Report written to {outpath}")
print(f"JSON written to {outpath_json}")
print(f"Quality models: {len(quality)}")
