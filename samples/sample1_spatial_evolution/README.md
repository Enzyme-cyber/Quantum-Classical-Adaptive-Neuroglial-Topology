# S1 — Spatially evolving neuroglial topology

[Repository quick start](../../README.md) · [Original technical notes](METHOD_NOTES.md)

## 1. Purpose and scope

Demonstrate history-dependent recruitment and competition of local neuroglial motifs, including the addition-to-multiplication task. The controller chooses among specified candidate primitives; it does not invent arbitrary operators outside that library.

## 2. Setup and working directory

Install dependencies from the repository root as described in the [root README](../../README.md). Commands below run **inside this sample directory**:

```bash
cd samples/sample1_spatial_evolution
```

Use a fresh output directory for each experiment. Do not overwrite the included reference results. Resource/surrogate checks use the core dependencies; `local` and `qpu` additionally require `pyqpanda3`. A surrogate result is not a quantum result.

## 3. Resource or numerical check

```bash
python V7_1_SPATIAL_EVOLVING_NETWORK.py \
  --core V7_1_neuroglial_gate_qpu.py \
  --project V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json \
  --evolution-config V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json --mode resource
```

## 4. Small software smoke test

```bash
python V7_1_SPATIAL_EVOLVING_NETWORK.py \
  --core V7_1_neuroglial_gate_qpu.py \
  --project V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json \
  --evolution-config V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json --mode surrogate --max-training-trials 8 --skip-probes --output-dir sample1_smoke
```

## 5. Local quantum smoke test

```bash
python V7_1_SPATIAL_EVOLVING_NETWORK.py \
  --core V7_1_neuroglial_gate_qpu.py \
  --project V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json \
  --evolution-config V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json --mode local --max-training-trials 8 --skip-probes --output-dir sample1_local_smoke
```

This is a diagnostic, not the full primary experiment.

## 6. Primary protocol

```bash
python V7_1_SPATIAL_EVOLVING_NETWORK.py \
  --core V7_1_neuroglial_gate_qpu.py \
  --project V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json \
  --evolution-config V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json --mode local --output-dir sample1_primary
```

This selects the complete primary task. The runner has no multi-seed CLI: a multi-run analysis needs an explicitly documented repetition and randomness-control workflow. One run does not reproduce a multi-run manuscript analysis. Local state-vector simulation can be expensive.

## Configuration and controls

- Primary project: `V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json`.
- Evolution settings: `V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json`.
- Resource audit: **16 qubits (8 neuronal + 8 glial), 4 layers, 21 candidate edges**.
- Training has 24 repeated-task trials followed by 20 consolidation trials.
- This runner has **no `--n-seeds`, `--seed-offset` or `--seed` option**. Separate output directories alone do not establish controlled independent simulator seeds.

## Outputs

- `candidate_edges.csv`, `emergence_events.csv`, `edge_evolution.csv`.
- `node_activity_and_slow_trace.csv`, `trial_summary.csv`, `V7_1_manifest.json`.
- Pre/post feedback and arithmetic-validation probe CSVs.
- Built-in diagnostic PNG plots; a separate publication-SVG workflow is not bundled.

### Frozen-plasticity control

Repeat the primary command with `--freeze-plasticity` and a new output directory, such as `sample1_frozen`. Keep all other settings matched.

## Physical-QPU execution

First follow the [QPU setup and safety notes](../../README.md#physical-qpu-execution). Select `--mode qpu` explicitly and choose the intended task configuration. Use a new output directory. A reduced QPU configuration is a different protocol, not automatically a primary replication. Resource and surrogate checks do not submit device jobs. Device execution was not tested during this documentation update.

## Interpretation and reproducibility limits

`surrogate` is a software diagnostic, not a quantum simulation. Emergence times quoted in historical notes can depend on mode and configuration and are not guaranteed reproduction targets. Gate recruitment follows local eligibility, slow traces, affinity, competition and hysteresis; do not describe it as unconstrained de novo operator discovery.

Older explanatory material is preserved in [METHOD_NOTES.md](METHOD_NOTES.md). That archive may contain historical paths, run commands or diagnostic values; use this README and the actual JSON configuration for this package's execution instructions.
