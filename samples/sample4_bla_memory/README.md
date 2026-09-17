# S4 — BLA fear and extinction memory

[Repository quick start](../../README.md) · [Original technical notes](METHOD_NOTES.md)

## 1. Purpose and scope

Explore competing fear/extinction memories in a neuroglial model constrained by the included Bukalo-derived protocol and summary information.

## 2. Setup and working directory

Install dependencies from the repository root as described in the [root README](../../README.md). Commands below run **inside this sample directory**:

```bash
cd samples/sample4_bla_memory
```

Use a fresh output directory for each experiment. Do not overwrite the included reference results. Resource/surrogate checks use the core dependencies; `local` and `qpu` additionally require `pyqpanda3`. A surrogate result is not a quantum result.

## 3. Resource or numerical check

```bash
python V7_1_SAMPLE4_BLA_FEAR_EXTINCTION_V1_3.py --mode resource
```

## 4. Small software smoke test

```bash
python V7_1_SAMPLE4_BLA_FEAR_EXTINCTION_V1_3.py --mode surrogate --task-config V7_1_SAMPLE4_BLA_task_LOCAL_FAST.json --agents full_biological --n-seeds 1 --output-dir sample4_smoke
```

## 5. Local quantum smoke test

```bash
python V7_1_SAMPLE4_BLA_FEAR_EXTINCTION_V1_3.py --mode local --task-config V7_1_SAMPLE4_BLA_task_LOCAL_FAST.json --agents full_biological --n-seeds 1 --output-dir sample4_local_smoke
```

This is a diagnostic, not the full primary experiment.

## 6. Primary protocol

```bash
python V7_1_SAMPLE4_BLA_FEAR_EXTINCTION_V1_3.py --mode local --task-config V7_1_SAMPLE4_BLA_task_PRIMARY.json --n-seeds 1 --output-dir sample4_primary
```

This selects the primary task configuration with one task repetition. Set `--n-seeds` to the planned/reported repetition count for a multi-seed analysis; one repetition does not reproduce a multi-seed manuscript result. Local state-vector simulation can be expensive.

## Configuration and controls

- PRIMARY: `V7_1_SAMPLE4_BLA_task_PRIMARY.json`; 3 conditioning trials in context A, two extinction days of 25 trials each in context B, 5 retrieval trials and 5 renewal trials; 13 agents.
- LOCAL_FAST: `V7_1_SAMPLE4_BLA_task_LOCAL_FAST.json`; schedule 3 + 6 + 6 + 2 + 2, 4 agents.
- QPU_SMOKE: `V7_1_SAMPLE4_BLA_task_QPU_SMOKE.json`; schedule 3 + 5 + 5 + 2 + 2, full agent only.
- QPU_BIOLOGICAL: `V7_1_SAMPLE4_BLA_task_QPU_BIOLOGICAL.json`; primary-length schedule, full agent only.
- Resource audit: **20 qubits, 4 layers, 21 candidate edges**; six required motifs present, two CS-only bypass motifs absent.
- PRIMARY's configured controls include frozen plasticity, glial/tripartite ablations, CalEx-like and DREADD-like interventions, context and retained-memory controls.
- `--n-seeds` and `--seed-offset` control task repetitions; use all configured agents for the full suite.

## Outputs

- `sample4_agent_group_summary.csv`, trial and seed logs, run manifest.
- Stage, transition and perturbation validation CSVs.
- Emergence events and topology time series.
- This runner writes tabular outputs; a standalone publication-figure generator is not bundled.

### Mechanism-focused DREADD comparison

Use the primary command with:

```text
--agents full_biological,hm3dq_like,hm3dq_fast_only,hm4di_like,hm4di_fast_only
```

Choose a fresh output directory and keep all other settings fixed.

### Biological source status

`bukalo2026_protocol_parameters.json` and `bukalo2026_source_provenance.json` describe constraints derived from the cited study. The original source workbook is **not bundled**; see `SOURCE_FILES_NOT_INCLUDED.txt`. Derived constraint tables are not a complete raw-data reanalysis. See the [BioSample plan](../../docs/BIOSAMPLE_REPRODUCIBILITY_PLAN.md).

## Physical-QPU execution

First follow the [QPU setup and safety notes](../../README.md#physical-qpu-execution). Select `--mode qpu` explicitly and choose the intended task configuration. Use a new output directory. A reduced QPU configuration is a different protocol, not automatically a primary replication. Resource and surrogate checks do not submit device jobs. Device execution was not tested during this documentation update.

## Interpretation and reproducibility limits

Stored candidate topology and moment-to-moment expression are different quantities. Simulated retained or competing memory motifs are model hypotheses, not experimentally observed anatomical edges. A biological-like intervention is not an exact reproduction of all effects of the corresponding biological manipulation.

Older explanatory material is preserved in [METHOD_NOTES.md](METHOD_NOTES.md). That archive may contain historical paths, run commands or diagnostic values; use this README and the actual JSON configuration for this package's execution instructions.
