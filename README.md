# Neuroglial Quantum-Formal Simulation Framework

Runnable sample code for matched fast-layer controls and history-dependent neuroglial topology. Start with the commands below; detailed configuration and interpretation notes are linked for each sample.

## Quick start: installation

Run from the **extracted repository root** (the directory containing this README). Python 3.10+ is recommended; the documentation checks used Python 3.12.14.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements-core.txt
```

For local quantum simulation or physical-QPU submission, also install:

```bash
python -m pip install -r requirements-qpu.txt
```

The quantum backend uses `pyqpanda3`. Dependencies are not version-locked in this snapshot. Record your actual Python/package versions and run configuration; installation alone does not establish SDK/backend compatibility.

## Core run commands

Each block starts **from the repository root** and returns there. The multiline commands use Bash line continuation; in PowerShell/CMD paste each Python command as a single line. Install `requirements-qpu.txt, and`Use a new output directory on reruns. 

Each sample ca Commands below select the full primary task configuration; S1B/S2/S2B/S3/S4 deliberately start with **one task repetition**, not a complete multi-seed reproduction. Increase `--n-seeds` to the planned/reported count (for example, 20) and keep the configured control suite when generating inferential results. These blocks do not submit physical-QPU jobs.  

### S0 — Matched quantum-formal and classical controls

[Sample README](samples/sample0_matched_control/README.md)

```bash
cd samples/sample0_matched_control
python verify_sample0.py

python run_sample0.py --seeds 20 --shots 3000 --output-dir sample0_primary
cd ../..
```



### S1 — Spatially evolving neuroglial topology

[Sample README](samples/sample1_spatial_evolution/README.md)

```bash
cd samples/sample1_spatial_evolution
python V7_1_SPATIAL_EVOLVING_NETWORK.py \
  --core V7_1_neuroglial_gate_qpu.py \
  --project V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json \
  --evolution-config V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json --mode resource

python V7_1_SPATIAL_EVOLVING_NETWORK.py \
  --core V7_1_neuroglial_gate_qpu.py \
  --project V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json \
  --evolution-config V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json --mode local --output-dir sample1_primary
cd ../..
```



### S1B — Arithmetic compilation and causal controls

[Sample README](samples/sample1b_arithmetic_compilation/README.md)

```bash
cd samples/sample1b_arithmetic_compilation
python V7_1_SAMPLE1B_ARITHMETIC_COMPILATION.py --mode resource

python V7_1_SAMPLE1B_ARITHMETIC_COMPILATION.py --mode local --task-config V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_task_PRIMARY.json --agents full,frozen_plasticity,no_tripartite_growth --n-seeds 1 --output-dir sample1b_primary
cd ../..
```



### S2 — Hidden-rule learning and reversal

[Sample README](samples/sample2_hidden_rule/README.md)

```bash
cd samples/sample2_hidden_rule
python V7_1_SAMPLE2_HIDDEN_RULE_V2.py --mode resource

python V7_1_SAMPLE2_HIDDEN_RULE_V2.py --mode local --task-config V7_1_SAMPLE2_HIDDEN_RULE_task_PRIMARY.json --reward-suite probabilistic --n-seeds 1 --output-dir sample2_primary
cd ../..
```



### S2B — Astrocytic context and state-matched probes

[Sample README](samples/sample2b_astrocytic_context/README.md)

```bash
cd samples/sample2b_astrocytic_context
python V7_1_SAMPLE2B_ASTROCYTIC_CONTEXT_NECESSITY.py --resource

python V7_1_SAMPLE2B_ASTROCYTIC_CONTEXT_NECESSITY.py --train-mode local --probe-mode local --n-seeds 1 --output-dir sample2b_primary
cd ../..
```



### S3 — Effort-driven strategy selection

[Sample README](samples/sample3_meta_strategy/README.md)

```bash
cd samples/sample3_meta_strategy
python V7_1_SAMPLE3_FACTOR_STRATEGY_V3.py --mode resource

python V7_1_SAMPLE3_FACTOR_STRATEGY_V3.py --mode local --task-config V7_1_SAMPLE3_FACTOR_STRATEGY_V3_task_PRIMARY.json --n-seeds 1 --output-dir sample3_primary
cd ../..
```



### S4 — BLA fear and extinction memory

[Sample README](samples/sample4_bla_memory/README.md)

```bash
cd samples/sample4_bla_memory
python V7_1_SAMPLE4_BLA_FEAR_EXTINCTION_V1_3.py --mode resource

python V7_1_SAMPLE4_BLA_FEAR_EXTINCTION_V1_3.py --mode local --task-config V7_1_SAMPLE4_BLA_task_PRIMARY.json --n-seeds 1 --output-dir sample4_primary
cd ../..
```

## Execution modes: what each run establishes


| Mode                       | What executes                                             | Appropriate use                                                            |
| -------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------- |
| S0 verification / resource | Numerical identities or resource/configuration inspection | Preflight checks; not task-performance evidence                            |
| `surrogate`                | Classical software surrogate                              | Wiring, output and control-flow diagnostics; not quantum results           |
| `local`                    | Quantum circuit simulation through CPUQVM                 | Local quantum-formal evaluation; not physical-QPU evidence                 |
| `qpu`                      | Submission to an authorized device backend                | Hardware results only when a completed job and its provenance are retained |
| S0 matched control         | NumPy coherent and matched classical kernels              | CPU-based matched comparison; does not use the above mode switch           |


S2B separates `--train-mode` and `--probe-mode`. Surrogate training plus local/QPU probes must be labelled as a mixed-mode experiment. Repetition seeds in the task controller are not a guarantee that CPUQVM or device shot sampling is independently seeded.

## Real-QPU Execution

In the manuscript runs, QPanda-dispatched results were obtained from physical QPU execution (WK_C180_1). The actual backend can be selected through the configuration or environment without changing the numerical framework.

Real-QPU backend API can be obtained through webportal of Origin Quantum:

[https://console.originqc.com.cn/zh/services](https://console.originqc.com.cn/zh/services)

Windows PowerShell:

```powershell
$env:QPANDA_QCLOUD_API_KEY="YOUR_API_KEY"
$env:QPANDA_QCLOUD_BACKEND="auto"
```

Linux/macOS:

```bash
export QPANDA_QCLOUD_API_KEY="YOUR_API_KEY"
export QPANDA_QCLOUD_BACKEND="auto"
```

Optional connectivity check:

```bash
python src/originq_cloud_setup.py --run-bell
```

To run QPU, simply replace the mode-local to mode-qpu:

```bash
--mode qpu
```

