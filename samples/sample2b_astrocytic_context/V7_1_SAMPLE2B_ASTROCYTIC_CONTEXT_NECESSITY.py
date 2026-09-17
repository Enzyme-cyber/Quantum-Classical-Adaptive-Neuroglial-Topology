#!/usr/bin/env python3
"""
V7.1 SAMPLE 2B — ASTROCYTIC CONTEXT NECESSITY TEST
===================================================

Purpose
-------
Sample 2 shows that reward history can select between opposite hidden rules.
Sample 2B asks the stricter biological/computational question:

    If the CURRENT neuronal state/input and direct neuronal topology are matched,
    can a history-dependent astroglial state still disambiguate two past histories?

This is a falsifiable *model prediction*, not a claim that wet-lab experiments
have already demonstrated N_A == N_B with G_A != G_B.

Primary paired-history construction
-----------------------------------
Two branches start from the same base network and receive the same stimulus
schedule. They differ only in reward contingency:

    history_default : S0->LEFT,  S1->RIGHT
    history_reversal: S0->RIGHT, S1->LEFT

No hidden-context bit is passed to the circuit. Direct plastic N->N candidates
are removed. Rewarded sensory-action replay may modify only astroglial G-G and
tripartite N-G-N candidate structure and slow astroglial traces.

At the final probe:
  1. all neuronal slow traces are hard-matched to zero;
  2. the direct neuronal graph is identical in the two branches;
  3. the same current stimulus is presented;
  4. history-specific astroglial state/topology is either preserved or
     selectively equalized/ablated.

The decisive prediction is therefore not "glia improves accuracy". It is:

    matched current neuronal state + same current input
    + different astroglial residual state
    -> different output distribution.

Controls decompose what carries the residual information:
  intact_glia               : history-specific G activity + G-mediated topology
  activity_equalized        : history-specific topology, equalized G activity
  topology_equalized_union  : equalized topology, history-specific G activity
  glia_equalized            : equalized topology + equalized G activity
  glia_ablated              : dynamic G-G/tripartite routes off; G activity zero

Interpretation
--------------
If intact_glia disambiguates histories while glia_equalized/glia_ablated do not,
the model supports a testable "astrocytic residual-context" prediction.
If only activity_equalized works, memory is mainly structural/topological.
If only topology_equalized_union works, memory is mainly the carried glial state.
If glia_equalized still disambiguates histories, the matching/control logic has
failed and the experiment should be treated as invalid.

Surrogate mode is a deterministic structural smoke test, NOT a quantum simulator.
Local/QPU probe modes execute the final matched circuits through the V7.1 backend.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import zlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

VERSION = "V7.1-SAMPLE2B-ASTROCYTIC-CONTEXT-NECESSITY-1.0"
DEFAULT_CORE = "V7_1_neuroglial_gate_qpu_SAMPLE2B.py"
DEFAULT_ENGINE = "V7_1_SPATIAL_EVOLVING_NETWORK_SAMPLE2B.py"
DEFAULT_SAMPLE2 = "V7_1_SAMPLE2_HIDDEN_RULE_V2.py"
DEFAULT_PROJECT = "V7_1_SAMPLE2_HIDDEN_RULE_project.json"
DEFAULT_EVOLUTION = "V7_1_SAMPLE2_HIDDEN_RULE_evolution.json"
DEFAULT_TASK = "V7_1_SAMPLE2B_task_PRIMARY.json"
EPS = 1e-12


@dataclass
class Task2B:
    name: str = "Sample 2B astrocytic context necessity"
    history_trials: int = 120
    stimulus_probability: float = 0.96
    reward_replay_probability: float = 0.98
    baseline_exploration: float = 0.06
    error_exploration: float = 0.36
    choice_gain: float = 14.0
    reward_correct: float = 0.90
    reward_incorrect: float = 0.10
    reward_suite: str = "deterministic"
    seed: int = 20260913
    probe_repeats: int = 8
    glial_trace_probability_gain: float = 0.95
    glial_probability_cap: float = 0.95
    probe_conditions: tuple[str, ...] = (
        "intact_glia",
        "activity_equalized",
        "topology_equalized_union",
        "glia_equalized",
        "glia_ablated",
    )
    node_map: dict[str, Any] | None = None


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_task(path: Path) -> Task2B:
    d = json.loads(path.read_text(encoding="utf-8"))
    return Task2B(
        name=str(d.get("name", "Sample 2B astrocytic context necessity")),
        history_trials=int(d.get("history_trials", 120)),
        stimulus_probability=float(d.get("stimulus_probability", 0.96)),
        reward_replay_probability=float(d.get("reward_replay_probability", 0.98)),
        baseline_exploration=float(d.get("baseline_exploration", 0.06)),
        error_exploration=float(d.get("error_exploration", 0.36)),
        choice_gain=float(d.get("choice_gain", 14.0)),
        reward_correct=float(d.get("reward_correct", 0.90)),
        reward_incorrect=float(d.get("reward_incorrect", 0.10)),
        reward_suite=str(d.get("reward_suite", "deterministic")),
        seed=int(d.get("seed", 20260913)),
        probe_repeats=int(d.get("probe_repeats", 8)),
        glial_trace_probability_gain=float(d.get("glial_trace_probability_gain", 0.95)),
        glial_probability_cap=float(d.get("glial_probability_cap", 0.95)),
        probe_conditions=tuple(str(x) for x in d.get("probe_conditions", [
            "intact_glia", "activity_equalized", "topology_equalized_union",
            "glia_equalized", "glia_ablated"
        ])),
        node_map=dict(d.get("node_map", {})),
    )


def validate_task(task: Task2B):
    if task.history_trials < 20:
        raise ValueError("history_trials must be >=20")
    if task.probe_repeats < 1:
        raise ValueError("probe_repeats must be >=1")
    if task.reward_suite not in {"deterministic", "probabilistic"}:
        raise ValueError("reward_suite must be deterministic or probabilistic")
    for x in (task.stimulus_probability, task.reward_replay_probability,
              task.reward_correct, task.reward_incorrect):
        if not (0.0 <= x <= 1.0):
            raise ValueError("probabilities must be 0..1")
    if not (0.0 <= task.glial_trace_probability_gain <= 3.0):
        raise ValueError("glial_trace_probability_gain should be 0..3")
    if not (0.0 < task.glial_probability_cap <= 1.0):
        raise ValueError("glial_probability_cap must be (0,1]")
    allowed = {
        "intact_glia", "activity_equalized", "topology_equalized_union",
        "glia_equalized", "glia_ablated"
    }
    bad = set(task.probe_conditions) - allowed
    if bad:
        raise ValueError(f"Unknown probe_conditions: {sorted(bad)}")
    for sec in ("sensory", "association_replay", "decision"):
        if sec not in (task.node_map or {}):
            raise ValueError(f"node_map missing {sec}")


def as_sample2_task(s2, task: Task2B):
    """Reuse Sample2 event/candidate-role logic without a reversal schedule."""
    return s2.TaskSettings(
        name=task.name,
        trials=task.history_trials,
        reversal_trial=max(8, task.history_trials // 2),
        stimulus_probability=task.stimulus_probability,
        reward_replay_probability=task.reward_replay_probability,
        baseline_exploration=task.baseline_exploration,
        error_exploration=task.error_exploration,
        choice_gain=task.choice_gain,
        reward_correct=task.reward_correct,
        reward_incorrect=task.reward_incorrect,
        seed=task.seed,
        rolling_window=12,
        evaluation_window=min(30, max(8, task.history_trials // 4)),
        recovery_accuracy=0.70,
        recovery_consecutive_windows=2,
        agents=("no_nn_consolidation",),
        node_map=task.node_map,
    )


def balanced_stimuli(n: int, rng: np.random.Generator) -> np.ndarray:
    out = []
    while len(out) < n:
        b = np.array([0, 0, 1, 1], dtype=int)
        rng.shuffle(b)
        out.extend(b.tolist())
    return np.asarray(out[:n], dtype=int)


def history_correct_action(rule: str, stimulus: int) -> int:
    if rule == "default":
        return int(stimulus)
    if rule == "reversal":
        return 1 - int(stimulus)
    raise ValueError(rule)


def glial_probabilities_from_traces(traces: dict[str, float], n_glia: int, task: Task2B) -> np.ndarray:
    vals = np.asarray([float(traces.get(f"G{i}", 0.0)) for i in range(n_glia)], dtype=float)
    return np.clip(task.glial_trace_probability_gain * vals, 0.0, task.glial_probability_cap)


def hard_reset_neuronal_traces(traces: dict[str, float], n_neurons: int) -> dict[str, float]:
    out = dict(traces)
    for i in range(n_neurons):
        out[f"N{i}"] = 0.0
    return out


def candidate_state_copy(engine, states):
    return {k: engine.CandidateState(float(v.strength), bool(v.active)) for k, v in states.items()}


def union_equalized_states(engine, evo, a, b):
    """Same permissive G-mediated topology for both histories."""
    out = {}
    for c in evo.candidates:
        sa, sb = a[c.id], b[c.id]
        out[c.id] = engine.CandidateState(
            strength=float(max(sa.strength, sb.strength)),
            active=bool(sa.active or sb.active),
        )
    return out


def ablated_states(engine, evo):
    return {c.id: engine.CandidateState(0.0, False) for c in evo.candidates}


def train_history_branch(
    rule: str,
    seed: int,
    train_mode: str,
    core,
    engine,
    s2,
    base_cfg,
    gains,
    hw,
    base_evo,
    task2: Task2B,
    task_s2,
    stimuli,
    choice_u,
    reward_u,
    outdir: Path,
):
    """Train one fixed hidden-rule history with no plastic direct N->N edges."""
    evo = copy.deepcopy(base_evo)
    # Critical necessity constraint: history cannot be stored in dynamic direct N->N edges.
    evo.candidates = [c for c in evo.candidates if c.type != "neuron_to_neuron"]
    engine.validate_evolution_config(base_cfg, evo)
    traces = engine.initialize_traces(base_cfg, evo)
    states = engine.initialize_states(evo)
    shots = int(evo.shots_override or hw.shots or 2500)
    if train_mode == "local" and evo.shots_override is None:
        shots = max(1000, min(12000, int(getattr(hw, "shots", 1000)) * 4))

    previous = None
    reward_memory = []
    trial_rows = []
    node_rows = []
    edge_rows = []

    for t in range(task2.history_trials):
        stim = int(stimuli[t])
        correct = history_correct_action(rule, stim)
        replay_record = previous if (previous is not None and int(previous.get("reward", 0)) == 1) else None
        events = s2.build_trial_events(engine, core, gains, replay_record, stim, task_s2)
        cfg = engine.materialize_dynamic_config(core, base_cfg, evo.candidates, states, events)
        _, marg = engine.run_trial_backend(
            train_mode, core, cfg, gains, shots, hw, outdir,
            f"sample2b_train_seed{seed}_{rule}_{t:03d}", readout_mode="population"
        )
        pL = float(marg[s2.decision_neuron(task_s2, 0)])
        pR = float(marg[s2.decision_neuron(task_s2, 1)])
        delta = pR - pL
        eps = task2.error_exploration if (previous is not None and int(previous.get("reward", 0)) == 0) else task2.baseline_exploration
        pright = s2.choice_probability_right(delta, eps, task2.choice_gain)
        action = 1 if float(choice_u[t]) < pright else 0
        is_correct = int(action == correct)
        if task2.reward_suite == "deterministic":
            reward = is_correct
            rprob = float(is_correct)
        else:
            rprob = task2.reward_correct if is_correct else task2.reward_incorrect
            reward = int(float(reward_u[t]) < rprob)

        engine.update_traces(traces, marg, cfg, evo.trace)
        engine.update_candidate_states(evo.candidates, states, traces, evo.competition)

        trial_rows.append({
            "seed": seed, "history_rule": rule, "trial": t, "stimulus": stim,
            "correct_action": correct, "action": action, "correct": is_correct,
            "reward": reward, "P_LEFT": pL, "P_RIGHT": pR,
            "delta_RIGHT_minus_LEFT": delta, "choice_probability_RIGHT": pright,
            "active_dynamic_edges": int(sum(st.active for st in states.values())),
        })
        for i in range(base_cfg.n_neurons):
            node_rows.append({
                "seed": seed, "history_rule": rule, "trial": t, "node": f"N{i}",
                "P_active": float(marg[i]), "slow_trace": float(traces[f"N{i}"]),
            })
        for g in range(base_cfg.n_glia):
            node_rows.append({
                "seed": seed, "history_rule": rule, "trial": t, "node": f"G{g}",
                "P_active": float(marg[base_cfg.n_neurons + g]), "slow_trace": float(traces[f"G{g}"]),
            })
        for c in evo.candidates:
            st = states[c.id]
            edge_rows.append({
                "seed": seed, "history_rule": rule, "trial": t,
                "candidate_id": c.id, "edge_type": c.type,
                "strength": float(st.strength), "active": bool(st.active),
                "driver": float(engine.driver_value(c, traces)),
                "mediating_glia": c.mediating_glia,
            })

        current = {"stimulus": stim, "action": action, "reward": reward}
        if reward:
            reward_memory.append(dict(current))
        previous = current

        if t < 2 or (t + 1) % 30 == 0 or t == task2.history_trials - 1:
            print(
                f"[train {seed}:{rule}] t={t:03d} S={stim} A={action} ok={is_correct} "
                f"R={reward} dD={delta:+.4f} active={sum(st.active for st in states.values())}"
            )

    pre_reset = dict(traces)
    post_reset = hard_reset_neuronal_traces(traces, base_cfg.n_neurons)
    return {
        "rule": rule,
        "evo": evo,
        "states": states,
        "traces_pre_reset": pre_reset,
        "traces_post_reset": post_reset,
        "glial_probabilities": glial_probabilities_from_traces(post_reset, base_cfg.n_glia, task2),
        "trial_log": pd.DataFrame(trial_rows),
        "node_log": pd.DataFrame(node_rows),
        "edge_log": pd.DataFrame(edge_rows),
    }


def probe_state_selection(condition: str, branch: str, engine, evo, branches):
    own = branches[branch]
    other = branches["reversal" if branch == "default" else "default"]
    pooled_p = 0.5 * (branches["default"]["glial_probabilities"] + branches["reversal"]["glial_probabilities"])
    union_states = branches["union_states"]
    if condition == "intact_glia":
        return candidate_state_copy(engine, own["states"]), own["glial_probabilities"].copy()
    if condition == "activity_equalized":
        return candidate_state_copy(engine, own["states"]), pooled_p.copy()
    if condition == "topology_equalized_union":
        return candidate_state_copy(engine, union_states), own["glial_probabilities"].copy()
    if condition == "glia_equalized":
        return candidate_state_copy(engine, union_states), pooled_p.copy()
    if condition == "glia_ablated":
        return ablated_states(engine, evo), np.zeros_like(own["glial_probabilities"])
    raise ValueError(condition)


def run_probe_condition(
    condition: str,
    branch: str,
    stimulus: int,
    repeat_idx: int,
    seed: int,
    probe_mode: str,
    core,
    engine,
    s2,
    base_cfg,
    gains,
    hw,
    evo,
    task2,
    task_s2,
    branches,
    outdir,
):
    states, gp = probe_state_selection(condition, branch, engine, evo, branches)
    events = s2.build_probe_events(engine, core, gains, stimulus, task_s2)
    cfg = engine.materialize_dynamic_config(core, base_cfg, evo.candidates, states, events)
    cfg = replace(
        cfg,
        initialize_neural_h=False,
        initialize_glial_baseline=False,
        glial_initial_probabilities=[float(x) for x in gp],
        neural_readout_mode="population",
    )
    shots = int(evo.shots_override or hw.shots or 2500)
    _, marg = engine.run_trial_backend(
        probe_mode, core, cfg, gains, shots, hw, outdir,
        f"sample2b_probe_seed{seed}_{condition}_{branch}_S{stimulus}_r{repeat_idx:02d}",
        readout_mode="population",
    )
    pL = float(marg[s2.decision_neuron(task_s2, 0)])
    pR = float(marg[s2.decision_neuron(task_s2, 1)])
    delta = pR - pL
    return {
        "seed": seed,
        "condition": condition,
        "history_rule": branch,
        "stimulus": stimulus,
        "repeat": repeat_idx,
        "P_LEFT": pL,
        "P_RIGHT": pR,
        "delta_RIGHT_minus_LEFT": delta,
        "P_choose_RIGHT_no_exploration": s2.choice_probability_right(delta, 0.0, task2.choice_gain),
        "glial_initial_probability_mean": float(np.mean(gp)),
        "glial_initial_probability_max": float(np.max(gp)),
        "active_dynamic_edges": int(sum(st.active for st in states.values())),
    }


def state_matching_rows(seed, base_cfg, evo, branches):
    a = branches["default"]
    b = branches["reversal"]
    n_pre = np.asarray([a["traces_pre_reset"][f"N{i}"] for i in range(base_cfg.n_neurons)])
    n_pre_b = np.asarray([b["traces_pre_reset"][f"N{i}"] for i in range(base_cfg.n_neurons)])
    n_post = np.asarray([a["traces_post_reset"][f"N{i}"] for i in range(base_cfg.n_neurons)])
    n_post_b = np.asarray([b["traces_post_reset"][f"N{i}"] for i in range(base_cfg.n_neurons)])
    g = np.asarray([a["traces_post_reset"][f"G{i}"] for i in range(base_cfg.n_glia)])
    g_b = np.asarray([b["traces_post_reset"][f"G{i}"] for i in range(base_cfg.n_glia)])
    topo_a = np.asarray([float(a["states"][c.id].strength) for c in evo.candidates])
    topo_b = np.asarray([float(b["states"][c.id].strength) for c in evo.candidates])
    direct_nn_candidates = [c for c in evo.candidates if c.type == "neuron_to_neuron"]
    return [{
        "seed": seed,
        "neuronal_trace_L1_before_reset": float(np.mean(np.abs(n_pre - n_pre_b))),
        "neuronal_trace_L1_after_reset": float(np.mean(np.abs(n_post - n_post_b))),
        "glial_trace_L1_after_neuronal_reset": float(np.mean(np.abs(g - g_b))),
        "glial_probability_L1": float(np.mean(np.abs(a["glial_probabilities"] - b["glial_probabilities"]))),
        "glial_topology_strength_L1": float(np.mean(np.abs(topo_a - topo_b))) if len(topo_a) else 0.0,
        "dynamic_direct_NN_candidate_count": int(len(direct_nn_candidates)),
        "base_direct_NN_edge_count": int(len(base_cfg.neuron_to_neuron or [])),
        "neuronal_match_pass": bool(np.allclose(n_post, n_post_b, atol=1e-12) and len(direct_nn_candidates) == 0),
        "glial_separation_present": bool(np.mean(np.abs(g - g_b)) > 1e-6 or (len(topo_a) and np.mean(np.abs(topo_a - topo_b)) > 1e-6)),
    }]


def snapshot_rows(seed, base_cfg, evo, branches):
    grows = []
    erows = []
    for rule in ("default", "reversal"):
        b = branches[rule]
        for i in range(base_cfg.n_neurons):
            grows.append({
                "seed": seed, "history_rule": rule, "node": f"N{i}", "node_type": "neuron",
                "slow_trace_pre_reset": float(b["traces_pre_reset"][f"N{i}"]),
                "slow_trace_post_reset": float(b["traces_post_reset"][f"N{i}"]),
                "initial_probability_at_probe": 0.0,
            })
        for g in range(base_cfg.n_glia):
            grows.append({
                "seed": seed, "history_rule": rule, "node": f"G{g}", "node_type": "glia",
                "slow_trace_pre_reset": float(b["traces_pre_reset"][f"G{g}"]),
                "slow_trace_post_reset": float(b["traces_post_reset"][f"G{g}"]),
                "initial_probability_at_probe": float(b["glial_probabilities"][g]),
            })
        for c in evo.candidates:
            st = b["states"][c.id]
            erows.append({
                "seed": seed, "history_rule": rule, "candidate_id": c.id,
                "edge_type": c.type, "strength": float(st.strength), "active": bool(st.active),
                "mediating_glia": c.mediating_glia,
                "drivers": ";".join(c.drivers),
            })
    return grows, erows


def paired_probe_summary(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if probes.empty:
        return pd.DataFrame()
    # Average technical probe repetitions first.
    m = probes.groupby(["seed", "condition", "history_rule", "stimulus"], as_index=False).agg(
        P_RIGHT=("P_RIGHT", "mean"),
        P_LEFT=("P_LEFT", "mean"),
        P_choose_RIGHT=("P_choose_RIGHT_no_exploration", "mean"),
    )
    for (seed, condition), g in m.groupby(["seed", "condition"], sort=False):
        def get(rule, stim, col="P_choose_RIGHT"):
            q = g[(g.history_rule == rule) & (g.stimulus == stim)]
            return float(q.iloc[0][col]) if len(q) else np.nan
        d0 = get("default", 0)
        r0 = get("reversal", 0)
        d1 = get("default", 1)
        r1 = get("reversal", 1)
        # Desired opposite mapping: S0 default L vs reversal R; S1 default R vs reversal L.
        s0 = r0 - d0 if np.isfinite(r0) and np.isfinite(d0) else np.nan
        s1 = d1 - r1 if np.isfinite(d1) and np.isfinite(r1) else np.nan
        rows.append({
            "seed": seed,
            "condition": condition,
            "S0_Pright_default_history": d0,
            "S0_Pright_reversal_history": r0,
            "S1_Pright_default_history": d1,
            "S1_Pright_reversal_history": r1,
            "S0_history_disambiguation": s0,
            "S1_history_disambiguation": s1,
            "bidirectional_history_disambiguation_index": float(np.nanmean([s0, s1])),
            "absolute_history_output_difference": float(np.nanmean([abs(r0-d0), abs(r1-d1)])),
        })
    return pd.DataFrame(rows)


def group_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition, g in seed_summary.groupby("condition", sort=False):
        x = pd.to_numeric(g["bidirectional_history_disambiguation_index"], errors="coerce")
        a = pd.to_numeric(g["absolute_history_output_difference"], errors="coerce")
        rows.append({
            "condition": condition,
            "n_seeds": int(g.seed.nunique()),
            "bidirectional_history_disambiguation_index_mean": float(x.mean()),
            "bidirectional_history_disambiguation_index_sd": float(x.std(ddof=1)) if x.notna().sum() > 1 else np.nan,
            "bidirectional_history_disambiguation_index_median": float(x.median()),
            "absolute_history_output_difference_mean": float(a.mean()),
            "absolute_history_output_difference_sd": float(a.std(ddof=1)) if a.notna().sum() > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def make_plots(probes, seed_summary, matching, outdir):
    if plt is None or probes.empty:
        return
    # Figure 1: primary paired outputs, seed/repeat averaged.
    m = probes.groupby(["condition", "history_rule", "stimulus"], as_index=False)["P_choose_RIGHT_no_exploration"].mean()
    conditions = list(dict.fromkeys(m.condition.tolist()))
    fig, axes = plt.subplots(len(conditions), 1, figsize=(7.0, max(3.0, 2.25*len(conditions))), sharex=True)
    if len(conditions) == 1:
        axes = [axes]
    x = np.arange(2)
    width = 0.34
    for ax, cond in zip(axes, conditions):
        q = m[m.condition == cond]
        dv = [float(q[(q.history_rule == "default") & (q.stimulus == s)].P_choose_RIGHT_no_exploration.iloc[0]) for s in (0,1)]
        rv = [float(q[(q.history_rule == "reversal") & (q.stimulus == s)].P_choose_RIGHT_no_exploration.iloc[0]) for s in (0,1)]
        ax.bar(x-width/2, dv, width, label="default history")
        ax.bar(x+width/2, rv, width, label="reversal history")
        ax.axhline(0.5, linewidth=1, linestyle="--")
        ax.set_ylim(0,1)
        ax.set_ylabel("P(RIGHT)")
        ax.set_title(cond)
    axes[-1].set_xticks(x, ["S0", "S1"])
    axes[0].legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "sample2b_paired_history_outputs.svg")
    fig.savefig(outdir / "sample2b_paired_history_outputs.png", dpi=260)
    plt.close(fig)

    # Figure 2: falsification/ablation index.
    if not seed_summary.empty:
        g = seed_summary.groupby("condition", as_index=False)["bidirectional_history_disambiguation_index"].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        xx = np.arange(len(g))
        ax.bar(xx, g["mean"].to_numpy(float), yerr=g["std"].fillna(0).to_numpy(float), capsize=3)
        ax.axhline(0.0, linewidth=1, linestyle="--")
        ax.set_xticks(xx, g["condition"], rotation=25, ha="right")
        ax.set_ylabel("History disambiguation index")
        ax.set_title("Sample 2B: glial-state necessity controls")
        fig.tight_layout()
        fig.savefig(outdir / "sample2b_ablation_disambiguation.svg")
        fig.savefig(outdir / "sample2b_ablation_disambiguation.png", dpi=260)
        plt.close(fig)

    # Figure 3: state matching diagnostics.
    if not matching.empty:
        cols = [
            "neuronal_trace_L1_before_reset",
            "neuronal_trace_L1_after_reset",
            "glial_trace_L1_after_neuronal_reset",
            "glial_topology_strength_L1",
        ]
        vals = [float(pd.to_numeric(matching[c], errors="coerce").mean()) for c in cols]
        fig, ax = plt.subplots(figsize=(7.8, 4.2))
        xx = np.arange(len(cols))
        ax.bar(xx, vals)
        ax.set_xticks(xx, ["N before reset", "N after reset", "G trace", "G topology"], rotation=20, ha="right")
        ax.set_ylabel("Mean L1 distance")
        ax.set_title("Matched neuronal state versus residual glial separation")
        fig.tight_layout()
        fig.savefig(outdir / "sample2b_state_matching.svg")
        fig.savefig(outdir / "sample2b_state_matching.png", dpi=260)
        plt.close(fig)


def resource_report(core, engine, base_cfg, evo, task2):
    print(f"Runner: {VERSION}")
    print(f"Core: {getattr(core,'VERSION','?')} | Engine: {getattr(engine,'VERSION','?')}")
    print(f"Qubits: {base_cfg.total_qubits} ({base_cfg.n_neurons} neurons + {base_cfg.n_glia} glia)")
    print(f"History trials per branch: {task2.history_trials}")
    print("Direct base N->N edges:", len(base_cfg.neuron_to_neuron or []))
    print("Dynamic candidate types before Sample2B filter:", pd.Series([c.type for c in evo.candidates]).value_counts().to_dict())
    print("Sample2B requirement: dynamic neuron_to_neuron candidates are removed during paired-history training.")
    print("Probe conditions:", list(task2.probe_conditions))
    print("Primary falsification control: glia_equalized must show ~0 history disambiguation (within sampling noise).")


def main():
    ap = argparse.ArgumentParser(description="V7.1 Sample2B astrocytic context necessity test")
    ap.add_argument("--core", default=DEFAULT_CORE)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--sample2-runner", default=DEFAULT_SAMPLE2)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--evolution-config", default=DEFAULT_EVOLUTION)
    ap.add_argument("--task-config", default=DEFAULT_TASK)
    ap.add_argument("--train-mode", choices=["surrogate", "local", "qpu"], default="surrogate")
    ap.add_argument("--probe-mode", choices=["surrogate", "local", "qpu"], default="surrogate")
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--resource", action="store_true")
    args = ap.parse_args()
    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be >=1")

    here = Path(__file__).resolve().parent
    def resolve(x):
        p = Path(x)
        return p if p.exists() else here / p

    core = load_module("sample2b_core", resolve(args.core))
    engine = load_module("sample2b_engine", resolve(args.engine))
    s2 = load_module("sample2_parent", resolve(args.sample2_runner))
    task2 = load_task(resolve(args.task_config))
    validate_task(task2)
    task_s2 = as_sample2_task(s2, task2)

    bio, base_cfg, hw, _ = core.project_from_json(resolve(args.project))
    core.validate_biological_input(bio)
    core.validate_network_config(base_cfg)
    gains = core.scores_to_layer_gains(core.compute_mechanistic_scores(bio), bio.hardware_gain)

    base_evo = engine.load_evolution_config(resolve(args.evolution_config))
    if len(base_evo.candidates) != 0:
        raise ValueError("Sample2B expects candidate_edges=[] and auto-generates the spatial pool")
    engine.finalize_spatial_candidates(base_cfg, base_evo)
    s2.prune_task_candidate_pool(base_evo, task_s2)
    s2.add_reward_eligibility_drivers(base_evo, task_s2)
    # Hard design constraint: no history-specific direct neuronal plasticity.
    base_evo.candidates = [c for c in base_evo.candidates if c.type != "neuron_to_neuron"]
    engine.validate_evolution_config(base_cfg, base_evo)

    if args.resource:
        resource_report(core, engine, base_cfg, base_evo, task2)
        return 0

    if args.train_mode == "qpu":
        print("WARNING: QPU training executes every history trial. Usually use --train-mode surrogate and --probe-mode qpu.")

    outdir = Path(args.output_dir) if args.output_dir else Path(
        f"V7_1_SAMPLE2B_train-{args.train_mode}_probe-{args.probe_mode}_{timestamp()}"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    all_train, all_nodes, all_edges = [], [], []
    all_probes, all_match, all_gsnap, all_esnap = [], [], [], []
    env_rows = []

    for rep in range(args.n_seeds):
        seed = int(task2.seed + args.seed_offset + 1009*rep)
        rng = np.random.default_rng(seed)
        stimuli = balanced_stimuli(task2.history_trials, rng)
        choice_u = rng.random(task2.history_trials)
        reward_u = rng.random(task2.history_trials)
        env_rows.append(pd.DataFrame({
            "seed": seed,
            "trial": np.arange(task2.history_trials),
            "stimulus": stimuli,
            "choice_uniform_shared_across_histories": choice_u,
            "reward_uniform_shared_across_histories": reward_u,
        }))

        print(f"\n=== Sample2B seed {seed}: paired histories ===")
        branches = {}
        for rule in ("default", "reversal"):
            branches[rule] = train_history_branch(
                rule, seed, args.train_mode, core, engine, s2, base_cfg, gains, hw,
                base_evo, task2, task_s2, stimuli, choice_u, reward_u, outdir
            )
            all_train.append(branches[rule]["trial_log"])
            all_nodes.append(branches[rule]["node_log"])
            all_edges.append(branches[rule]["edge_log"])

        branches["union_states"] = union_equalized_states(
            engine, base_evo, branches["default"]["states"], branches["reversal"]["states"]
        )
        all_match.extend(state_matching_rows(seed, base_cfg, base_evo, branches))
        gs, es = snapshot_rows(seed, base_cfg, base_evo, branches)
        all_gsnap.extend(gs); all_esnap.extend(es)

        for condition in task2.probe_conditions:
            print(f"[probe seed={seed}] {condition}")
            for branch in ("default", "reversal"):
                for stim in (0, 1):
                    for r in range(task2.probe_repeats):
                        all_probes.append(run_probe_condition(
                            condition, branch, stim, r, seed, args.probe_mode,
                            core, engine, s2, base_cfg, gains, hw, base_evo,
                            task2, task_s2, branches, outdir
                        ))

    train = pd.concat(all_train, ignore_index=True) if all_train else pd.DataFrame()
    nodes = pd.concat(all_nodes, ignore_index=True) if all_nodes else pd.DataFrame()
    edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    probes = pd.DataFrame(all_probes)
    matching = pd.DataFrame(all_match)
    gsnap = pd.DataFrame(all_gsnap)
    esnap = pd.DataFrame(all_esnap)
    env = pd.concat(env_rows, ignore_index=True) if env_rows else pd.DataFrame()

    train.to_csv(outdir / "sample2b_history_training_log.csv", index=False)
    nodes.to_csv(outdir / "sample2b_history_node_activity_and_traces.csv", index=False)
    edges.to_csv(outdir / "sample2b_history_edge_evolution.csv", index=False)
    probes.to_csv(outdir / "sample2b_paired_probe_results.csv", index=False)
    matching.to_csv(outdir / "sample2b_state_matching_diagnostics.csv", index=False)
    gsnap.to_csv(outdir / "sample2b_final_state_snapshot.csv", index=False)
    esnap.to_csv(outdir / "sample2b_final_topology_snapshot.csv", index=False)
    env.to_csv(outdir / "sample2b_matched_environment_schedule.csv", index=False)

    seed_summary = paired_probe_summary(probes)
    seed_summary.to_csv(outdir / "sample2b_seed_summary.csv", index=False)
    gsummary = group_summary(seed_summary)
    gsummary.to_csv(outdir / "sample2b_group_summary.csv", index=False)

    # Explicit paired effects relative to intact model.
    effect_rows = []
    if not seed_summary.empty:
        for seed, g in seed_summary.groupby("seed"):
            f = g[g.condition == "intact_glia"]
            if f.empty:
                continue
            fv = float(f.iloc[0].bidirectional_history_disambiguation_index)
            for _, r in g.iterrows():
                effect_rows.append({
                    "seed": seed,
                    "condition": r.condition,
                    "intact_minus_condition_disambiguation": fv - float(r.bidirectional_history_disambiguation_index),
                })
    pd.DataFrame(effect_rows).to_csv(outdir / "sample2b_paired_control_effects.csv", index=False)

    make_plots(probes, seed_summary, matching, outdir)

    manifest = {
        "version": VERSION,
        "scientific_claim_level": "falsifiable model prediction; not wet-lab proof",
        "train_mode": args.train_mode,
        "probe_mode": args.probe_mode,
        "task": asdict(task2),
        "n_seeds": args.n_seeds,
        "constraints": {
            "hidden_context_bit_passed_to_circuit": False,
            "dynamic_direct_neuron_to_neuron_candidates": 0,
            "neuronal_slow_traces_hard_matched_before_probe": True,
            "same_current_stimulus_used_across_paired_histories": True,
            "glial_state_injected_as_initial_population": True,
        },
        "primary_prediction": "intact_glia > glia_equalized ~= glia_ablated in history-disambiguation index",
        "interpretive_controls": {
            "activity_equalized": "tests history-specific astroglial topology with activity equalized",
            "topology_equalized_union": "tests carried astroglial activity with topology equalized",
            "glia_equalized": "negative control; both glial activity and topology are identical",
            "glia_ablated": "negative control; dynamic glial routes and carried glial activity are removed",
        },
    }
    (outdir / "sample2b_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(core.compute_mechanistic_scores(bio))]).to_csv(outdir / "mechanistic_scores.csv", index=False)
    pd.DataFrame([asdict(gains)]).to_csv(outdir / "layer_gains.csv", index=False)

    print("\n=== STATE MATCHING ===")
    print(matching.to_string(index=False))
    print("\n=== SAMPLE2B SEED SUMMARY ===")
    print(seed_summary.to_string(index=False))
    if args.n_seeds > 1:
        print("\n=== GROUP SUMMARY ===")
        print(gsummary.to_string(index=False))
    print("\nDONE:", outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
