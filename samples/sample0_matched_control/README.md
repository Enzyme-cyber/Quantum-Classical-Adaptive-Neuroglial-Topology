# S0 — Matched quantum-formal and classical controls

[Repository quick start](../../README.md) · [Original technical notes](METHOD_NOTES.md)

## 1. Purpose and scope

Compare a coherent, quantum-formal fast layer with a matched classical joint-state transition model while keeping the slow controller and task protocol aligned. Both implementations run on the CPU with NumPy; this module does not submit QPU jobs.

## 2. Setup and working directory

Install dependencies from the repository root as described in the [root README](../../README.md). Commands below run **inside this sample directory**:

```bash
cd samples/sample0_matched_control
```

Use a fresh output directory for each experiment. Do not overwrite the included reference results. The core scientific dependencies are sufficient.

## 3. Resource or numerical check

```bash
python verify_sample0.py
```

## 4. Small software smoke test

```bash
python run_sample0.py --seeds 2 --max-training-trials 8 --grid-size 3 \
  --conditions q_original c_original q_context c_context \
  --output-dir sample0_smoke
```

## 5. Primary protocol

```bash
python run_sample0.py --seeds 20 --shots 3000 --output-dir sample0_primary
```

The command uses the bundled 20-seed matched-control protocol.

## Configuration and controls

- Bundled baseline: `v71_original/` (frozen source and configuration; not reformatted).
- Default comparison: 20 paired seeds across 10 conditions.
- `--seed-start` sets the first seed; `--seeds` controls repetitions.
- `--shots 0` requires `--seeds 1` and produces an exact-probability diagnostic, not a multi-seed finite-shot experiment.

## Outputs

- `seed_summary.csv` and aggregate summaries: paired performance comparisons.
- Per-seed/per-condition logs, probes, edge histories and compressed raw counts.
- `figures/Fig0_matched_control.svg` and PNG; `figure_data/` stores plotting inputs.

### Reanalyse a fresh run

```bash
python analyze_sample0.py --input-dir sample0_primary
python verify_result_files.py --input-dir sample0_primary
python audit_phase.py --output-dir phase_audit_new
```

The analysis and verification tools write into their target directories. Copy bundled `results_20seeds/` before reanalysis if you need to preserve its original bytes. `verify_sample0.py` writes `verification.json` in this sample directory. Use `--resume` only with an identical run specification; do not reuse a nonempty output directory for a different experiment.

## Interpretation and reproducibility limits

The `q_original`/`c_original` comparison and the `q_context`/`c_context` extension answer different questions and should be reported separately. The optional context mixer changes the model. The `no_phase` control removes designated RZ/ZZ operations; it does not remove every source of coherence. A model-level difference is not evidence of biological quantum computation or general quantum advantage.

Older explanatory material is preserved in [METHOD_NOTES.md](METHOD_NOTES.md). That archive may contain historical paths, run commands or diagnostic values; use this README and the actual JSON configuration for this package's execution instructions.
