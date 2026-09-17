# S2 — Hidden-rule learning and reversal

[Repository quick start](../../README.md) · [Original technical notes](METHOD_NOTES.md)

## 1. Purpose and scope

Evaluate history-dependent binary-choice learning, reversal and mechanistic controls when the hidden rule is not supplied as an explicit context label.

## 2. Setup and working directory

Install dependencies from the repository root as described in the [root README](../../README.md). Commands below run **inside this sample directory**:

```bash
cd samples/sample2_hidden_rule
```

Use a fresh output directory for each experiment. Do not overwrite the included reference results. Resource/surrogate checks use the core dependencies; `local` and `qpu` additionally require `pyqpanda3`. A surrogate result is not a quantum result.

## 3. Resource or numerical check

```bash
python V7_1_SAMPLE2_HIDDEN_RULE_V2.py --mode resource
```

## 4. Small software smoke test

```bash
python V7_1_SAMPLE2_HIDDEN_RULE_V2.py --mode surrogate --task-config V7_1_SAMPLE2_HIDDEN_RULE_task_LOCAL_FAST.json --reward-suite probabilistic --agents full_spatial_competition --n-seeds 1 --output-dir sample2_smoke
```

## 5. Local quantum smoke test

```bash
python V7_1_SAMPLE2_HIDDEN_RULE_V2.py --mode local --task-config V7_1_SAMPLE2_HIDDEN_RULE_task_LOCAL_FAST.json --reward-suite probabilistic --agents full_spatial_competition --n-seeds 1 --output-dir sample2_local_smoke
```

This is a diagnostic, not the full primary experiment.

## 6. Primary protocol

```bash
python V7_1_SAMPLE2_HIDDEN_RULE_V2.py --mode local --task-config V7_1_SAMPLE2_HIDDEN_RULE_task_PRIMARY.json --reward-suite probabilistic --n-seeds 1 --output-dir sample2_primary
```

This selects the primary task configuration with one task repetition. Set `--n-seeds` to the planned/reported repetition count for a multi-seed analysis; one repetition does not reproduce a multi-seed manuscript result. Local state-vector simulation can be expensive.

## Configuration and controls

- PRIMARY: `V7_1_SAMPLE2_HIDDEN_RULE_task_PRIMARY.json`; 220 trials, reversal at trial index 100.
- `V7_1_SAMPLE2_HIDDEN_RULE_task_LOCAL_FAST.json`: 96 trials, reversal at 38; reduced agent set.
- `V7_1_SAMPLE2_HIDDEN_RULE_task_QPU_MINIMAL.json`: 110 trials, reversal at 44; full agent only.
- Resource audit: **20 qubits, 2 layers, 26 candidate edges**; hidden-context input disabled.
- **The CLI defaults to deterministic rewards. Use `--reward-suite probabilistic` explicitly for the probabilistic primary protocol (0.90/0.10).**
- PRIMARY includes 10 agents; omit `--agents` to use the configured suite. `--agents full_spatial_competition` is only a single-agent diagnostic.
- Task seeds follow `task.seed + seed_offset + 1009 × repetition`; this is not an SDK sampler-seeding guarantee.

## Outputs

- Trial log, agent/seed and group summaries, environment schedule and manifest.
- Checkpoint probes, same-stimulus flip summary and paired-control effects.
- Topology time series, candidate edges, emergence and slow-trace logs.
- Built-in rolling-accuracy, topology-selectivity and same-stimulus-flip PNG plots.

## Physical-QPU execution

First follow the [QPU setup and safety notes](../../README.md#physical-qpu-execution). Select `--mode qpu` explicitly and choose the intended task configuration. Use a new output directory. A reduced QPU configuration is a different protocol, not automatically a primary replication. Resource and surrogate checks do not submit device jobs. Device execution was not tested during this documentation update.

## Interpretation and reproducibility limits

Report reward regime, reversal index, seed count and agent suite with each result. Included `REFERENCE_surrogate_*` outputs are software diagnostics, not local CPUQVM or physical-QPU evidence. Reduced FAST/MINIMAL protocols cannot be labelled as PRIMARY replications.

Older explanatory material is preserved in [METHOD_NOTES.md](METHOD_NOTES.md). That archive may contain historical paths, run commands or diagnostic values; use this README and the actual JSON configuration for this package's execution instructions.
