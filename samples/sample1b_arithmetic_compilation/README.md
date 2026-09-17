# S1B — Arithmetic compilation and causal controls

[Repository quick start](../../README.md) · [Original technical notes](METHOD_NOTES.md)

## 1. Purpose and scope

Test whether repeated procedural addition can support a more direct product response, using held-out operands, causal edge ablations and topology-reset controls.

## 2. Setup and working directory

Install dependencies from the repository root as described in the [root README](../../README.md). Commands below run **inside this sample directory**:

```bash
cd samples/sample1b_arithmetic_compilation
```

Use a fresh output directory for each experiment. Do not overwrite the included reference results. Resource/surrogate checks use the core dependencies; `local` and `qpu` additionally require `pyqpanda3`. A surrogate result is not a quantum result.

## 3. Resource or numerical check

```bash
python V7_1_SAMPLE1B_ARITHMETIC_COMPILATION.py --mode resource
```

## 4. Small software smoke test

```bash
python V7_1_SAMPLE1B_ARITHMETIC_COMPILATION.py --mode surrogate --task-config V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_task_QPU_SMOKE.json --agents full --n-seeds 1 --output-dir sample1b_smoke
```

## 5. Local quantum smoke test

```bash
python V7_1_SAMPLE1B_ARITHMETIC_COMPILATION.py --mode local --task-config V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_task_QPU_SMOKE.json --agents full --n-seeds 1 --output-dir sample1b_local_smoke
```

This is a diagnostic, not the full primary experiment.

## 6. Primary protocol

```bash
python V7_1_SAMPLE1B_ARITHMETIC_COMPILATION.py --mode local --task-config V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_task_PRIMARY.json --agents full,frozen_plasticity,no_tripartite_growth --n-seeds 1 --output-dir sample1b_primary
```

This selects the primary task configuration with one task repetition. Set `--n-seeds` to the planned/reported repetition count for a multi-seed analysis; one repetition does not reproduce a multi-seed manuscript result. Local state-vector simulation can be expensive.

## Configuration and controls

- `V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_task_PRIMARY.json`: 3 epochs, 5,000 shots; training/calibration operands 1–3, held-out operand 4.
- `V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_task_QPU.json`: shortened 2-epoch, 3,000-shot protocol.
- `V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_task_QPU_SMOKE.json`: 1-epoch diagnostic.
- **The bundled project uses 14 qubits (7 neuronal + 7 glial), 1 layer and 6 candidate edges. It is not a 28-qubit configuration.** Reconcile this with the manuscript before claiming exact reproduction.
- Agents: `full`, `frozen_plasticity`, `no_tripartite_growth`.
- `--n-seeds` and `--seed-offset` control task repetitions; the task seed starts at 20260824. These flags are not proof that the SDK shot sampler is seeded.

## Outputs

- `arithmetic_agent_summary.csv` at the run root.
- Per-agent arithmetic validation, training, edge/event, direct-product and held-out probe CSVs.
- Product-edge ablation, topology-reset and call-count comparisons; `arithmetic_manifest.json`.
- For multiple seeds, per-agent outputs are nested under `seed_<seed>/`; no standalone publication-figure generator is bundled.

## Physical-QPU execution

First follow the [QPU setup and safety notes](../../README.md#physical-qpu-execution). Select `--mode qpu` explicitly and choose the intended task configuration. Use a new output directory. A reduced QPU configuration is a different protocol, not automatically a primary replication. Resource and surrogate checks do not submit device jobs. Device execution was not tested during this documentation update.

## Interpretation and reproducibility limits

Affine decoder calibration must remain separate from the held-out test; report raw and decoded responses where appropriate. A shortened QPU task is not interchangeable with PRIMARY. Resource validation and QPU-ready code do not demonstrate successful device-level arithmetic compilation.

Older explanatory material is preserved in [METHOD_NOTES.md](METHOD_NOTES.md). That archive may contain historical paths, run commands or diagnostic values; use this README and the actual JSON configuration for this package's execution instructions.
