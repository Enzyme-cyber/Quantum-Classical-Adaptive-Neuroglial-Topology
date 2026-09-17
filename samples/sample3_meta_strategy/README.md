# S3 — Effort-driven strategy selection

[Repository quick start](../../README.md) · [Original technical notes](METHOD_NOTES.md)

## 1. Purpose and scope

Test selection among supplied factorization procedures using effort feedback and history-dependent topology, including quotient re-entry and mechanistic ablations.

## 2. Setup and working directory

Install dependencies from the repository root as described in the [root README](../../README.md). Commands below run **inside this sample directory**:

```bash
cd samples/sample3_meta_strategy
```

Use a fresh output directory for each experiment. Do not overwrite the included reference results. Resource/surrogate checks use the core dependencies; `local` and `qpu` additionally require `pyqpanda3`. A surrogate result is not a quantum result.

## 3. Resource or numerical check

```bash
python V7_1_SAMPLE3_FACTOR_STRATEGY_V3.py --mode resource
```

## 4. Small software smoke test

```bash
python V7_1_SAMPLE3_FACTOR_STRATEGY_V3.py --mode surrogate --task-config V7_1_SAMPLE3_FACTOR_STRATEGY_V3_task_LOCAL_FAST.json --agents full_competition --n-seeds 1 --output-dir sample3_smoke
```

## 5. Local quantum smoke test

```bash
python V7_1_SAMPLE3_FACTOR_STRATEGY_V3.py --mode local --task-config V7_1_SAMPLE3_FACTOR_STRATEGY_V3_task_LOCAL_FAST.json --agents full_competition --n-seeds 1 --output-dir sample3_local_smoke
```

This is a diagnostic, not the full primary experiment.

## 6. Primary protocol

```bash
python V7_1_SAMPLE3_FACTOR_STRATEGY_V3.py --mode local --task-config V7_1_SAMPLE3_FACTOR_STRATEGY_V3_task_PRIMARY.json --n-seeds 1 --output-dir sample3_primary
```

This selects the primary task configuration with one task repetition. Set `--n-seeds` to the planned/reported repetition count for a multi-seed analysis; one repetition does not reproduce a multi-seed manuscript result. Local state-vector simulation can be expensive.

## Configuration and controls

- PRIMARY: `V7_1_SAMPLE3_FACTOR_STRATEGY_V3_task_PRIMARY.json`; 95 training trials, 13 agents.
- LOCAL_FAST: `V7_1_SAMPLE3_FACTOR_STRATEGY_V3_task_LOCAL_FAST.json`; reduced training and 4 agents.
- QPU_MINIMAL: `V7_1_SAMPLE3_FACTOR_STRATEGY_V3_task_QPU_MINIMAL.json`; 20 training trials, 2 agents.
- Resource audit: **20 qubits, 2 layers, 16 candidate edges**.
- Nine hold-outs: 4, 9, 25, 49, 420, 630, 945, 1024 and 997.
- Omit `--agents` for the configured suite; use `--agents full_competition` only for a diagnostic.
- Task seeds follow `task.seed + seed_offset + repetition`; this does not establish simulator sampler seeding.

## Outputs

- Agent/group summaries, hold-out probes and paired-control effects.
- Emergence events, topology time series and token-flow traces.
- Run manifest and built-in diagnostic PNG plots.

## Physical-QPU execution

First follow the [QPU setup and safety notes](../../README.md#physical-qpu-execution). Select `--mode qpu` explicitly and choose the intended task configuration. Use a new output directory. A reduced QPU configuration is a different protocol, not automatically a primary replication. Resource and surrogate checks do not submit device jobs. Device execution was not tested during this documentation update.

## Interpretation and reproducibility limits

Effort is measured through defined primitive-call costs, not a hardware wall-clock speedup. The host supplies arithmetic primitives and decoding. This is neither Shor's algorithm nor evidence of quantum computational advantage. Surrogate reference performance must not be presented as physical-QPU performance.

Older explanatory material is preserved in [METHOD_NOTES.md](METHOD_NOTES.md). That archive may contain historical paths, run commands or diagnostic values; use this README and the actual JSON configuration for this package's execution instructions.
