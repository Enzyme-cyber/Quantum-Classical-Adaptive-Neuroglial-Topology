# S2B — Astrocytic context and state-matched probes

[Repository quick start](../../README.md) · [Original technical notes](METHOD_NOTES.md)

## 1. Purpose and scope

Contrast histories with opposite rules and probe how glial activity and topology contribute to their subsequent response differences after resetting neuronal slow traces.

## 2. Setup and working directory

Install dependencies from the repository root as described in the [root README](../../README.md). Commands below run **inside this sample directory**:

```bash
cd samples/sample2b_astrocytic_context
```

Use a fresh output directory for each experiment. Do not overwrite the included reference results. Resource/surrogate checks use the core dependencies; `local` and `qpu` additionally require `pyqpanda3`. A surrogate result is not a quantum result.

## 3. Resource or numerical check

```bash
python V7_1_SAMPLE2B_ASTROCYTIC_CONTEXT_NECESSITY.py --resource
```

## 4. Small software smoke test

```bash
python V7_1_SAMPLE2B_ASTROCYTIC_CONTEXT_NECESSITY.py --train-mode surrogate --probe-mode surrogate --n-seeds 1 --output-dir sample2b_smoke
```

## 5. Local quantum smoke test

```bash
python V7_1_SAMPLE2B_ASTROCYTIC_CONTEXT_NECESSITY.py --train-mode surrogate --probe-mode local --n-seeds 1 --output-dir sample2b_local_probe
```

This is a diagnostic, not the full primary experiment.

## 6. Primary protocol

```bash
python V7_1_SAMPLE2B_ASTROCYTIC_CONTEXT_NECESSITY.py --train-mode local --probe-mode local --n-seeds 1 --output-dir sample2b_primary
```

This selects the primary task configuration with one task repetition. Set `--n-seeds` to the planned/reported repetition count for a multi-seed analysis; one repetition does not reproduce a multi-seed manuscript result. Local state-vector simulation can be expensive.

## Configuration and controls

- Task: `V7_1_SAMPLE2B_task_PRIMARY.json`.
- Resource audit: **20 qubits**, 120 history trials per branch; direct neuronal–neuronal base and dynamic edges absent.
- Five probe conditions: `intact_glia`, `activity_equalized`, `topology_equalized_union`, `glia_equalized`, `glia_ablated`.
- Uses the bundled SAMPLE2B-specific engine and core, not the unmodified S2 engine.
- Unlike the other runners, use `--resource`, `--train-mode` and `--probe-mode`; there is no `--mode resource`.
- Increasing `--n-seeds` repeats the task; document both execution modes and `--seed-offset`.

## Outputs

- Group and seed summaries; history-training and paired-probe records.
- `sample2b_state_matching_diagnostics.csv` and manifests record the matching intervention.
- Built-in diagnostic figures in PNG and SVG.

### Check the probe contrasts

```bash
python CHECK_SAMPLE2B_RESULTS.py sample2b_primary
```

The default negative-control tolerance is 0.08. Treat the checker as a model diagnostic; its PASS/FAIL status is not a substitute for an inferential analysis. Do not tune the tolerance after seeing results.

## Physical-QPU execution

First follow the [QPU setup and safety notes](../../README.md#physical-qpu-execution). Choose `--train-mode qpu` and/or `--probe-mode qpu` deliberately; report which stage actually used hardware. Use a new output directory. A reduced QPU configuration is a different protocol, not automatically a primary replication. Resource and surrogate checks do not submit device jobs. Device execution was not tested during this documentation update.

## Interpretation and reproducibility limits

The local-probe smoke command above uses **surrogate training with quantum-simulated probes only**. It is not all-local training and certainly not full QPU learning. Activity matching and topology matching are distinct interventions. These tests establish conditional behavior in the model, not biological necessity; they are not a substitute for a future fully topology-matched biological experiment.

Older explanatory material is preserved in [METHOD_NOTES.md](METHOD_NOTES.md). That archive may contain historical paths, run commands or diagnostic values; use this README and the actual JSON configuration for this package's execution instructions.
