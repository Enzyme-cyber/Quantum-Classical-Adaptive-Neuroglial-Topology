#!/usr/bin/env python3
"""
V7.1 SAMPLE 2 — OPTIMIZED HIDDEN-RULE BINARY CHOICE
====================================================

Publication-oriented closed-loop reversal task built on the generic V7.1
spatial evolving neuroglial framework.

Primary scientific question
---------------------------
Can reward history select between competing functional operators without an
explicit hidden-context input?

Key upgrades over V7_1_HIDDEN_RULE_BINARY_V1.py
------------------------------------------------
1. No hidden context is passed to the circuit or plasticity engine.
2. Exact future edges remain absent from the evolution JSON.
3. Rule assemblies are derived automatically from the task/node map rather
   than explicitly listed in the task config.
4. Reward history is represented by four sensory-action replay ensembles
   (S0-L, S0-R, S1-R, S1-L). Candidate plasticity receives a task-generic
   reward-eligibility driver: a candidate sensory->action route can grow only
   if the matching rewarded sensory-action replay trace is present.
5. The spatial possibility graph is intentionally near-symmetric and does not
   encode which hidden rule is currently correct.
6. Added publication controls:
      - frozen_spatial
      - no_nn_consolidation
      - no_tripartite
      - no_glia_glia
      - no_competition
      - equal_affinity
      - shuffled_affinity
      - shuffled_reward_history
      - no_reward_replay
7. Added fixed checkpoint probes showing whether the SAME sensory input maps
   to opposite decision biases after default-rule vs reversal-rule learning.
8. Added topology-switch metrics and multi-seed aggregation.

Scientific scope
----------------
This remains a hybrid hypothesis-screening model. Across-trial plasticity and
structural competition are classical update rules that rebuild quantum-circuit
analogues. The code does not claim that astrocytes are literal qubits, that
brain tissue implements microscopic quantum gates, or that the model solves
formal undecidable problems.
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

VERSION = "V7.1-SAMPLE2-HIDDEN-RULE-2.0"
DEFAULT_CORE = "V7_1_neuroglial_gate_qpu.py"
DEFAULT_ENGINE = "V7_1_SPATIAL_EVOLVING_NETWORK.py"
DEFAULT_PROJECT = "V7_1_SAMPLE2_HIDDEN_RULE_project.json"
DEFAULT_EVOLUTION = "V7_1_SAMPLE2_HIDDEN_RULE_evolution.json"
DEFAULT_TASK = "V7_1_SAMPLE2_HIDDEN_RULE_task_PRIMARY.json"
EPS = 1e-12


@dataclass
class TaskSettings:
    name: str = "V7.1 Sample 2 hidden-rule reversal"
    trials: int = 180
    reversal_trial: int = 70
    stimulus_probability: float = 0.96
    reward_replay_probability: float = 0.98
    baseline_exploration: float = 0.06
    error_exploration: float = 0.36
    choice_gain: float = 14.0
    reward_correct: float = 0.80
    reward_incorrect: float = 0.20
    seed: int = 20260814
    rolling_window: int = 12
    evaluation_window: int = 24
    recovery_accuracy: float = 0.70
    recovery_consecutive_windows: int = 2
    agents: tuple[str, ...] = (
        "full_spatial_competition",
        "no_nn_consolidation",
        "no_tripartite",
        "no_glia_glia",
        "frozen_spatial",
        "no_competition",
        "equal_affinity",
        "shuffled_affinity",
        "shuffled_reward_history",
        "no_reward_replay",
    )
    node_map: dict[str, Any] | None = None


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_int(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_task(path: Path) -> TaskSettings:
    d = json.loads(path.read_text(encoding="utf-8"))
    return TaskSettings(
        name=str(d.get("name", "V7.1 Sample 2 hidden-rule reversal")),
        trials=int(d.get("trials", 180)),
        reversal_trial=int(d.get("reversal_trial", 70)),
        stimulus_probability=float(d.get("stimulus_probability", 0.96)),
        reward_replay_probability=float(d.get("reward_replay_probability", 0.98)),
        baseline_exploration=float(d.get("baseline_exploration", 0.06)),
        error_exploration=float(d.get("error_exploration", 0.36)),
        choice_gain=float(d.get("choice_gain", 14.0)),
        reward_correct=float(d.get("reward_correct", 0.80)),
        reward_incorrect=float(d.get("reward_incorrect", 0.20)),
        seed=int(d.get("seed", 20260814)),
        rolling_window=int(d.get("rolling_window", 12)),
        evaluation_window=int(d.get("evaluation_window", 24)),
        recovery_accuracy=float(d.get("recovery_accuracy", 0.70)),
        recovery_consecutive_windows=int(d.get("recovery_consecutive_windows", 2)),
        agents=tuple(str(x) for x in d.get("agents", ["full_spatial_competition"])),
        node_map=dict(d.get("node_map", {})),
    )


def validate_task(s: TaskSettings):
    if s.trials < 30:
        raise ValueError("trials must be >=30")
    if not (8 <= s.reversal_trial <= s.trials - 8):
        raise ValueError("reversal_trial must leave sufficient trials on both sides")
    for name in ("stimulus_probability", "reward_replay_probability"):
        if not (0.0 <= getattr(s, name) <= 1.0):
            raise ValueError(f"{name} must be 0..1")
    for name in ("baseline_exploration", "error_exploration"):
        if not (0.0 <= getattr(s, name) <= 0.8):
            raise ValueError(f"{name} must be 0..0.8")
    if s.choice_gain <= 0:
        raise ValueError("choice_gain must be >0")
    if s.rolling_window < 2 or s.evaluation_window < 4:
        raise ValueError("rolling/evaluation windows are too small")
    allowed = {
        "full_spatial_competition",
        "frozen_spatial",
        "no_tripartite",
        "no_nn_consolidation",
        "no_glia_glia",
        "no_competition",
        "equal_affinity",
        "shuffled_affinity",
        "shuffled_reward_history",
        "no_reward_replay",
    }
    bad = set(s.agents) - allowed
    if bad:
        raise ValueError(f"Unknown agents: {sorted(bad)}")
    required = {"sensory", "association_replay", "decision"}
    missing = required - set((s.node_map or {}).keys())
    if missing:
        raise ValueError(f"node_map missing: {sorted(missing)}")


def node_from_map(task: TaskSettings, section: str, key: str | int) -> int:
    d = (task.node_map or {}).get(section, {})
    k = str(key)
    if k not in d:
        raise KeyError(f"node_map.{section}[{k!r}] missing")
    return int(d[k])


def sensory_neuron(task: TaskSettings, stimulus: int) -> int:
    return node_from_map(task, "sensory", stimulus)


def decision_neuron(task: TaskSettings, action: int) -> int:
    return node_from_map(task, "decision", action)


def replay_neuron(task: TaskSettings, stimulus: int, action: int) -> int:
    return node_from_map(task, "association_replay", f"{int(stimulus)},{int(action)}")


def balanced_stimuli(n: int, rng: np.random.Generator) -> np.ndarray:
    vals: list[int] = []
    while len(vals) < n:
        b = np.array([0, 0, 1, 1], dtype=int)
        rng.shuffle(b)
        vals.extend(b.tolist())
    return np.asarray(vals[:n], dtype=int)


def true_context(t: int, reversal_trial: int) -> str:
    return "default" if t < reversal_trial else "reversal"


def correct_action(context: str, stimulus: int) -> int:
    return int(stimulus) if context == "default" else 1 - int(stimulus)


def sigmoid(x: float) -> float:
    x = float(np.clip(x, -40, 40))
    return 1.0 / (1.0 + math.exp(-x))


def choice_probability_right(delta_d: float, epsilon: float, gain: float) -> float:
    p = sigmoid(gain * float(delta_d))
    return float(epsilon * 0.5 + (1.0 - epsilon) * p)


def association_glia_map(base_cfg, task: TaskSettings) -> dict[tuple[int, int], int]:
    by_source: dict[int, list[int]] = {}
    for e in base_cfg.neuron_to_glia or []:
        by_source.setdefault(int(e.source), []).append(int(e.target))
    out: dict[tuple[int, int], int] = {}
    for stim in (0, 1):
        for act in (0, 1):
            rn = replay_neuron(task, stim, act)
            targets = sorted(set(by_source.get(rn, [])))
            if len(targets) != 1:
                raise ValueError(
                    f"Reward replay N{rn} must map to exactly one glial association unit; got {targets}"
                )
            out[(stim, act)] = targets[0]
    return out


def derive_rule_assemblies(base_cfg, task: TaskSettings) -> dict[str, dict[str, Any]]:
    """Derive diagnostics from task/node roles; nothing here is fed into learning."""
    amap = association_glia_map(base_cfg, task)
    result: dict[str, dict[str, Any]] = {}
    for ctx in ("default", "reversal"):
        pairs = [(s, correct_action(ctx, s)) for s in (0, 1)]
        gs = [amap[p] for p in pairs]
        a, b = sorted(gs)
        bridge = f"AUTO_GG_G{a}_G{b}"
        tris = []
        nns = []
        for stim, act in pairs:
            g = amap[(stim, act)]
            sn = sensory_neuron(task, stim)
            dn = decision_neuron(task, act)
            tris.append(f"AUTO_TRI_N{sn}_G{g}_N{dn}")
            nns.append(f"AUTO_NN_G{g}_N{sn}_N{dn}")
        result[ctx] = {
            "pairs": pairs,
            "glia": gs,
            "bridge": bridge,
            "tripartites": tris,
            "direct_nn": nns,
        }
    return result


def candidate_role(cid: str, assemblies: dict[str, dict[str, Any]]) -> str:
    for which in ("default", "reversal"):
        a = assemblies[which]
        if cid == a["bridge"]:
            return f"{which}_bridge"
        if cid in a["tripartites"]:
            return f"{which}_tripartite"
        if cid in a["direct_nn"]:
            return f"{which}_direct_nn"
    return "competitor_or_distractor"


def state_strength(states: dict[str, Any], cid: str) -> float:
    if cid not in states:
        return float("nan")
    return float(states[cid].strength)


def state_active(states: dict[str, Any], cid: str) -> bool:
    return bool(cid in states and states[cid].active)


def assembly_metrics(states, assemblies, which: str):
    a = assemblies[which]
    primary_ids = [a["bridge"]] + list(a["tripartites"])
    vals = [state_strength(states, x) for x in primary_ids if x in states]
    acts = [state_active(states, x) for x in primary_ids if x in states]
    tri_vals = [state_strength(states, x) for x in a["tripartites"] if x in states]
    nn_vals = [state_strength(states, x) for x in a["direct_nn"] if x in states]
    return {
        "bridge_strength": state_strength(states, a["bridge"]),
        "tripartite_mean_strength": float(np.mean(tri_vals)) if tri_vals else float("nan"),
        "assembly_min_strength": float(np.min(vals)) if vals else float("nan"),
        "assembly_all_active": bool(primary_ids and all(x in states and state_active(states, x) for x in primary_ids)),
        "direct_nn_mean_strength": float(np.mean(nn_vals)) if nn_vals else float("nan"),
        "direct_nn_all_active": bool(a["direct_nn"] and all(x in states and state_active(states, x) for x in a["direct_nn"])),
    }


def rule_trace_score(traces: dict[str, float], assemblies) -> tuple[float, float, float]:
    dg = assemblies["default"]["glia"]
    rg = assemblies["reversal"]["glia"]
    d = float(np.mean([float(traces.get(f"G{x}", 0.0)) for x in dg]))
    r = float(np.mean([float(traces.get(f"G{x}", 0.0)) for x in rg]))
    return d, r, r - d


def prune_task_candidate_pool(evo, task: TaskSettings):
    """
    Keep the spatially generated G-G pool, tripartite sensory->decision routes,
    and only the biologically interpretable sensory->decision N->N consolidation
    routes. This is a role-based filter, not a future-rule filter.
    """
    sensory = {sensory_neuron(task, 0), sensory_neuron(task, 1)}
    decision = {decision_neuron(task, 0), decision_neuron(task, 1)}
    keep = []
    for c in evo.candidates:
        if c.type == "neuron_to_neuron":
            if int(c.source) not in sensory or int(c.target) not in decision:
                continue
        if c.type == "tripartite":
            if int(c.sensory_neuron) not in sensory or int(c.target_neuron) not in decision:
                continue
        keep.append(c)
    evo.candidates = keep
    return evo


def add_reward_eligibility_drivers(evo, task: TaskSettings):
    """
    Add a rewarded sensory-action replay trace to each candidate action route.
    This does not specify the hidden rule. It simply says that a route from a
    sensory identity to an action target is eligible to consolidate if that
    same sensory-action pair has previously been rewarded.
    """
    sensory_lookup = {
        sensory_neuron(task, 0): 0,
        sensory_neuron(task, 1): 1,
    }
    decision_lookup = {
        decision_neuron(task, 0): 0,
        decision_neuron(task, 1): 1,
    }
    for c in evo.candidates:
        stim = act = None
        if c.type == "tripartite":
            stim = sensory_lookup.get(int(c.sensory_neuron))
            act = decision_lookup.get(int(c.target_neuron))
        elif c.type == "neuron_to_neuron":
            stim = sensory_lookup.get(int(c.source))
            act = decision_lookup.get(int(c.target))
        if stim is None or act is None:
            continue
        rn = replay_neuron(task, stim, act)
        tag = f"N{rn}"
        if tag not in c.drivers:
            c.drivers.append(tag)
    return evo


def shuffled_affinity_copy(evo, seed: int):
    rng = np.random.default_rng(seed)
    groups: dict[str, list[Any]] = {}
    for c in evo.candidates:
        groups.setdefault(c.type, []).append(c)
    for typ, cs in groups.items():
        vals = np.asarray([float(c.affinity) for c in cs], dtype=float)
        rng.shuffle(vals)
        for c, v in zip(cs, vals):
            c.affinity = float(v)
    return evo


def agent_evolution(engine, base_cfg, base_evo, agent: str, task: TaskSettings, seed: int):
    evo = copy.deepcopy(base_evo)
    if agent == "no_tripartite":
        evo.candidates = [c for c in evo.candidates if c.type != "tripartite"]
    elif agent == "no_nn_consolidation":
        evo.candidates = [c for c in evo.candidates if c.type != "neuron_to_neuron"]
    elif agent == "no_glia_glia":
        evo.candidates = [c for c in evo.candidates if c.type != "glia_glia"]
    elif agent == "no_competition":
        evo.competition = replace(evo.competition, enabled=False)
    elif agent == "equal_affinity":
        for c in evo.candidates:
            c.affinity = 1.0
    elif agent == "shuffled_affinity":
        shuffled_affinity_copy(evo, seed + stable_int(agent))
    elif agent in {
        "full_spatial_competition",
        "frozen_spatial",
        "shuffled_reward_history",
        "no_reward_replay",
    }:
        pass
    else:
        raise ValueError(agent)
    engine.validate_evolution_config(base_cfg, evo)
    return evo


def select_replay_record(
    agent: str,
    previous: dict[str, Any] | None,
    reward_memory: list[dict[str, Any]],
    rng_history: np.random.Generator,
):
    if agent == "no_reward_replay":
        return None
    if previous is None or int(previous.get("reward", 0)) != 1:
        return None
    if agent != "shuffled_reward_history":
        return previous
    # Same number of replay opportunities as ordered history, but the identity
    # of the recalled rewarded event is permuted across the accumulated memory.
    if len(reward_memory) <= 1:
        return reward_memory[0] if reward_memory else previous
    # Exclude the most recent item when possible to specifically remove recency.
    idx = int(rng_history.integers(0, len(reward_memory) - 1))
    return reward_memory[idx]


def build_trial_events(engine, core, gains, replay_record, stimulus: int, task: TaskSettings):
    spec = []
    if replay_record is not None:
        spec.append(
            {
                "layer": 0,
                "neuron": replay_neuron(task, int(replay_record["stimulus"]), int(replay_record["action"])),
                "probability": task.reward_replay_probability,
                "event_kind": "evidence",
            }
        )
    spec.append(
        {
            "layer": 1,
            "neuron": sensory_neuron(task, int(stimulus)),
            "probability": task.stimulus_probability,
            "event_kind": "evidence",
        }
    )
    return engine.build_events(core, spec, gains)


def build_probe_events(engine, core, gains, stimulus: int, task: TaskSettings):
    spec = [
        {
            "layer": 1,
            "neuron": sensory_neuron(task, int(stimulus)),
            "probability": task.stimulus_probability,
            "event_kind": "evidence",
        }
    ]
    return engine.build_events(core, spec, gains)


def safe_nanmean(values):
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def rolling_mean(x, w):
    a = np.asarray(x, dtype=float)
    out = np.full(len(a), np.nan)
    for i in range(w - 1, len(a)):
        out[i] = float(np.mean(a[i - w + 1 : i + 1]))
    return out


def recovery_latency(df: pd.DataFrame, task: TaskSettings, start_trial: int | None = None):
    start = task.reversal_trial if start_trial is None else max(task.reversal_trial, int(start_trial))
    p = df[df.trial >= start].sort_values("trial")
    if p.empty:
        return np.nan
    # Prevent a chance high-accuracy streak from being called recovery: the
    # learned state must also remain above threshold in the terminal evaluation window.
    tail = p.tail(min(task.evaluation_window, len(p)))
    if tail.empty or float(tail.correct.mean()) < task.recovery_accuracy:
        return np.nan
    roll = rolling_mean(p.correct.to_numpy(float), task.rolling_window)
    k = task.recovery_consecutive_windows
    for i in range(len(roll)):
        if i + 1 < k:
            continue
        seg = roll[i - k + 1 : i + 1]
        if np.all(np.isfinite(seg)) and np.all(seg >= task.recovery_accuracy):
            return int(p.iloc[i].trial - task.reversal_trial + 1)
    return np.nan


def first_formed(em: pd.DataFrame, ids: list[str]) -> float:
    if em.empty or not ids:
        return np.nan
    d = em[(em.event == "formed") & (em.candidate_id.isin(ids))]
    if d.empty:
        return np.nan
    vals = []
    for cid in ids:
        q = d[d.candidate_id == cid]
        if q.empty:
            return np.nan
        vals.append(int(q.trial.min()))
    return float(max(vals))


def topology_switch_latency(g: pd.DataFrame, task: TaskSettings):
    p = g[g.trial >= task.reversal_trial].sort_values("trial")
    if p.empty:
        return np.nan
    for _, r in p.iterrows():
        # Strict rule switch: reversal rule assembly active and default rule
        # assembly not simultaneously complete.
        if bool(r.reversal_assembly_all_active) and not bool(r.default_assembly_all_active):
            return int(r.trial - task.reversal_trial + 1)
    return np.nan


def run_checkpoint_probe(
    mode,
    agent,
    checkpoint,
    seed,
    core,
    engine,
    base_cfg,
    gains,
    hw,
    evo,
    states,
    task,
    assemblies,
    shots,
    outdir,
):
    rows = []
    dm = assembly_metrics(states, assemblies, "default")
    rm = assembly_metrics(states, assemblies, "reversal")
    for stim in (0, 1):
        events = build_probe_events(engine, core, gains, stim, task)
        cfg = engine.materialize_dynamic_config(core, base_cfg, evo.candidates, states, events)
        _, marg = engine.run_trial_backend(
            mode,
            core,
            cfg,
            gains,
            shots,
            hw,
            outdir,
            f"seed{seed}_{agent}_{checkpoint}_S{stim}",
            readout_mode="population",
        )
        pL = float(marg[decision_neuron(task, 0)])
        pR = float(marg[decision_neuron(task, 1)])
        delta = pR - pL
        rows.append(
            {
                "seed": seed,
                "agent": agent,
                "checkpoint": checkpoint,
                "stimulus": stim,
                "P_LEFT": pL,
                "P_RIGHT": pR,
                "delta_RIGHT_minus_LEFT": delta,
                "P_choose_RIGHT_no_exploration": choice_probability_right(delta, 0.0, task.choice_gain),
                "default_assembly_min_strength": dm["assembly_min_strength"],
                "default_assembly_all_active": dm["assembly_all_active"],
                "reversal_assembly_min_strength": rm["assembly_min_strength"],
                "reversal_assembly_all_active": rm["assembly_all_active"],
            }
        )
    return rows


def same_stimulus_probe_metrics(probes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if probes.empty:
        return pd.DataFrame()
    for (seed, agent), g in probes.groupby(["seed", "agent"], sort=False):
        def val(cp, stim, col):
            q = g[(g.checkpoint == cp) & (g.stimulus == stim)]
            return float(q.iloc[0][col]) if len(q) else np.nan

        d_s0 = val("pre_reversal_learned", 0, "P_choose_RIGHT_no_exploration")
        d_s1 = val("pre_reversal_learned", 1, "P_choose_RIGHT_no_exploration")
        r_s0 = val("post_reversal_learned", 0, "P_choose_RIGHT_no_exploration")
        r_s1 = val("post_reversal_learned", 1, "P_choose_RIGHT_no_exploration")
        # Desired flips: S0 L->R, S1 R->L.
        flip_s0 = r_s0 - d_s0 if np.isfinite(r_s0) and np.isfinite(d_s0) else np.nan
        flip_s1 = d_s1 - r_s1 if np.isfinite(d_s1) and np.isfinite(r_s1) else np.nan
        rows.append(
            {
                "seed": seed,
                "agent": agent,
                "S0_Pright_default_learned": d_s0,
                "S0_Pright_reversal_learned": r_s0,
                "S1_Pright_default_learned": d_s1,
                "S1_Pright_reversal_learned": r_s1,
                "S0_flip_L_to_R": flip_s0,
                "S1_flip_R_to_L": flip_s1,
                "same_stimulus_opposite_rule_index": safe_nanmean([flip_s0, flip_s1]),
            }
        )
    return pd.DataFrame(rows)


def summarize(trials: pd.DataFrame, emergence: pd.DataFrame, probes: pd.DataFrame, task: TaskSettings, assemblies):
    rows = []
    probe_metrics = same_stimulus_probe_metrics(probes)
    for (seed, agent), g in trials.groupby(["seed", "agent"], sort=False):
        g = g.sort_values("trial")
        pre = g[g.trial < task.reversal_trial]
        post = g[g.trial >= task.reversal_trial]
        late_pre = pre.tail(min(task.evaluation_window, len(pre)))
        late_post = post.tail(min(task.evaluation_window, len(post)))
        em = emergence[(emergence.seed == seed) & (emergence.agent == agent)] if not emergence.empty else pd.DataFrame()
        da = assemblies["default"]
        ra = assemblies["reversal"]
        default_assembly_trial = first_formed(em, [da["bridge"]] + da["tripartites"])
        reversal_bridge_trial = first_formed(em, [ra["bridge"]])
        reversal_tri_trial = first_formed(em, ra["tripartites"])
        default_nn_trial = first_formed(em, da["direct_nn"])
        reversal_nn_trial = first_formed(em, ra["direct_nn"])
        pm = probe_metrics[(probe_metrics.seed == seed) & (probe_metrics.agent == agent)]
        flip_index = float(pm.iloc[0].same_stimulus_opposite_rule_index) if len(pm) else np.nan
        pre_topo = float(late_pre.default_assembly_min_strength.mean() - late_pre.reversal_assembly_min_strength.mean()) if len(late_pre) else np.nan
        post_topo = float(late_post.reversal_assembly_min_strength.mean() - late_post.default_assembly_min_strength.mean()) if len(late_post) else np.nan
        rows.append(
            {
                "seed": seed,
                "agent": agent,
                "overall_accuracy": float(g.correct.mean()),
                "pre_reversal_accuracy": float(pre.correct.mean()) if len(pre) else np.nan,
                "post_reversal_accuracy": float(post.correct.mean()) if len(post) else np.nan,
                "early_post_accuracy": float(post.head(min(12, len(post))).correct.mean()) if len(post) else np.nan,
                "late_pre_accuracy": float(late_pre.correct.mean()) if len(late_pre) else np.nan,
                "late_post_accuracy": float(late_post.correct.mean()) if len(late_post) else np.nan,
                "reward_rate": float(g.reward.mean()),
                "default_assembly_first_complete_trial": default_assembly_trial,
                "reversal_bridge_first_formed_trial": reversal_bridge_trial,
                "reversal_tripartites_first_complete_trial": reversal_tri_trial,
                "default_direct_NN_first_complete_trial": default_nn_trial,
                "reversal_direct_NN_first_complete_trial": reversal_nn_trial,
                "behavioral_recovery_latency_trials": recovery_latency(g, task),
                "topology_switch_latency_trials": topology_switch_latency(g, task),
                "mean_rule_trace_score_pre": float(pre.rule_trace_reversal_minus_default.mean()) if len(pre) else np.nan,
                "mean_rule_trace_score_post": float(post.rule_trace_reversal_minus_default.mean()) if len(post) else np.nan,
                "late_pre_default_minus_reversal_topology": pre_topo,
                "late_post_reversal_minus_default_topology": post_topo,
                "topology_bidirectional_selectivity": safe_nanmean([pre_topo, post_topo]),
                "same_stimulus_opposite_rule_index": flip_index,
                "final_default_assembly_min_strength": float(g.iloc[-1].default_assembly_min_strength),
                "final_reversal_assembly_min_strength": float(g.iloc[-1].reversal_assembly_min_strength),
            }
        )
    return pd.DataFrame(rows)


def group_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metrics = [
        "overall_accuracy",
        "pre_reversal_accuracy",
        "post_reversal_accuracy",
        "late_pre_accuracy",
        "late_post_accuracy",
        "behavioral_recovery_latency_trials",
        "topology_switch_latency_trials",
        "topology_bidirectional_selectivity",
        "same_stimulus_opposite_rule_index",
    ]
    rows = []
    for agent, g in seed_summary.groupby("agent", sort=False):
        row = {"agent": agent, "n_seeds": int(g.seed.nunique())}
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean())
            row[f"{m}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
            row[f"{m}_median"] = float(vals.median())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_control_effects(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    metrics = [
        "late_pre_accuracy",
        "late_post_accuracy",
        "topology_bidirectional_selectivity",
        "same_stimulus_opposite_rule_index",
    ]
    rows = []
    for seed, g in seed_summary.groupby("seed"):
        f = g[g.agent == "full_spatial_competition"]
        if f.empty:
            continue
        f = f.iloc[0]
        for _, r in g.iterrows():
            if r.agent == "full_spatial_competition":
                continue
            row = {"seed": seed, "control": r.agent}
            for m in metrics:
                fv = float(f[m]) if pd.notna(f[m]) else np.nan
                cv = float(r[m]) if pd.notna(r[m]) else np.nan
                row[f"full_minus_control_{m}"] = fv - cv if np.isfinite(fv) and np.isfinite(cv) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def save_candidate_catalog(evo, assemblies, outdir):
    rows = []
    for c in evo.candidates:
        d = asdict(c)
        d["drivers"] = ";".join(c.drivers)
        d["competition_groups"] = ";".join(c.competition_groups)
        d["role"] = candidate_role(c.id, assemblies)
        rows.append(d)
    pd.DataFrame(rows).to_csv(outdir / "sample2_candidate_edges.csv", index=False)


def plots(trials, probes, task, outdir):
    if plt is None or trials.empty:
        return
    # One plot per seed would be noisy; plot seed-averaged rolling accuracy.
    temp = []
    for (seed, agent), g in trials.groupby(["seed", "agent"], sort=False):
        g = g.sort_values("trial").copy()
        g["rolling"] = rolling_mean(g.correct.to_numpy(float), task.rolling_window)
        temp.append(g[["seed", "agent", "trial", "rolling"]])
    dd = pd.concat(temp, ignore_index=True)
    mean = dd.groupby(["agent", "trial"], as_index=False)["rolling"].mean()
    fig, ax = plt.subplots(figsize=(11, 6))
    for agent, g in mean.groupby("agent", sort=False):
        ax.plot(g["trial"], g["rolling"], label=agent)
    ax.axvline(task.reversal_trial, linestyle="--", linewidth=1)
    ax.axhline(0.5, linewidth=1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("trial")
    ax.set_ylabel(f"mean rolling accuracy (w={task.rolling_window})")
    ax.set_title("Sample 2: hidden-rule reversal")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "sample2_rolling_accuracy.png", dpi=220)
    plt.close(fig)

    # Topology selectivity: positive means current rule topology dominates.
    fig, ax = plt.subplots(figsize=(11, 6))
    for agent, g in trials.groupby("agent", sort=False):
        q = g.groupby("trial", as_index=False)[["default_assembly_min_strength", "reversal_assembly_min_strength"]].mean()
        y = np.where(
            q.trial.to_numpy() < task.reversal_trial,
            q.default_assembly_min_strength - q.reversal_assembly_min_strength,
            q.reversal_assembly_min_strength - q.default_assembly_min_strength,
        )
        ax.plot(q.trial, y, label=agent)
    ax.axvline(task.reversal_trial, linestyle="--", linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_xlabel("trial")
    ax.set_ylabel("current-rule topology selectivity")
    ax.set_title("Rule-specific topology selection")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / "sample2_topology_selectivity.png", dpi=220)
    plt.close(fig)

    if not probes.empty:
        p = same_stimulus_probe_metrics(probes)
        q = p.groupby("agent", as_index=False).same_stimulus_opposite_rule_index.mean()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(np.arange(len(q)), q.same_stimulus_opposite_rule_index)
        ax.set_xticks(np.arange(len(q)))
        ax.set_xticklabels(q.agent, rotation=35, ha="right")
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("same-stimulus opposite-rule index")
        ax.set_title("Fixed probes: same input, opposite learned rule")
        fig.tight_layout()
        fig.savefig(outdir / "sample2_same_stimulus_flip.png", dpi=220)
        plt.close(fig)


def run_agent(
    mode,
    agent,
    seed,
    core,
    engine,
    base_cfg,
    gains,
    hw,
    base_evo,
    task,
    assemblies,
    stimuli,
    reward_u,
    choice_u,
    outdir,
    reward_suite,
):
    evo = agent_evolution(engine, base_cfg, base_evo, agent, task, seed)
    traces = engine.initialize_traces(base_cfg, evo)
    states = engine.initialize_states(evo)
    shots = int(evo.shots_override or hw.shots or 2500)
    if mode == "local" and evo.shots_override is None:
        shots = max(1000, min(12000, int(getattr(hw, "shots", 1000)) * 4))

    trial_rows = []
    node_rows = []
    edge_rows = []
    emergence = []
    probe_rows = []
    previous = None
    reward_memory: list[dict[str, Any]] = []
    rng_history = np.random.default_rng(seed + stable_int(agent) + 880301)

    probe_rows.extend(
        run_checkpoint_probe(
            mode, agent, "initial", seed, core, engine, base_cfg, gains, hw,
            evo, states, task, assemblies, shots, outdir
        )
    )

    for t in range(task.trials):
        stim = int(stimuli[t])
        ctx = true_context(t, task.reversal_trial)
        correct = correct_action(ctx, stim)
        replay_record = select_replay_record(agent, previous, reward_memory, rng_history)
        events = build_trial_events(engine, core, gains, replay_record, stim, task)
        cfg = engine.materialize_dynamic_config(core, base_cfg, evo.candidates, states, events)
        _, marg = engine.run_trial_backend(
            mode,
            core,
            cfg,
            gains,
            shots,
            hw,
            outdir,
            f"seed{seed}_{agent}_trial_{t:03d}",
            readout_mode="population",
        )
        pL = float(marg[decision_neuron(task, 0)])
        pR = float(marg[decision_neuron(task, 1)])
        delta = pR - pL
        eps = task.error_exploration if (previous is not None and int(previous["reward"]) == 0) else task.baseline_exploration
        pright = choice_probability_right(delta, eps, task.choice_gain)
        action = 1 if float(choice_u[t]) < pright else 0
        is_correct = int(action == correct)
        if reward_suite == "deterministic":
            reward = is_correct
            rprob = float(is_correct)
        else:
            rprob = task.reward_correct if is_correct else task.reward_incorrect
            reward = int(float(reward_u[t]) < rprob)

        plasticity = agent != "frozen_spatial"
        if plasticity:
            engine.update_traces(traces, marg, cfg, evo.trace)
            events_new = engine.update_candidate_states(evo.candidates, states, traces, evo.competition)
            for e in events_new:
                emergence.append(
                    {
                        "seed": seed,
                        "agent": agent,
                        "trial": t,
                        "true_context": ctx,
                        "role": candidate_role(e["candidate_id"], assemblies),
                        **e,
                    }
                )

        dtrace, rtrace, trscore = rule_trace_score(traces, assemblies)
        dm = assembly_metrics(states, assemblies, "default")
        rm = assembly_metrics(states, assemblies, "reversal")
        trial_rows.append(
            {
                "seed": seed,
                "agent": agent,
                "trial": t,
                "true_context": ctx,
                "stimulus": stim,
                "correct_action": correct,
                "action": action,
                "correct": is_correct,
                "reward": reward,
                "reward_probability": rprob,
                "history_mode": "none" if agent == "no_reward_replay" else ("shuffled" if agent == "shuffled_reward_history" else "ordered"),
                "replay_stimulus": np.nan if replay_record is None else int(replay_record["stimulus"]),
                "replay_action": np.nan if replay_record is None else int(replay_record["action"]),
                "exploration_epsilon": eps,
                "P_LEFT": pL,
                "P_RIGHT": pR,
                "delta_RIGHT_minus_LEFT": delta,
                "choice_probability_RIGHT": pright,
                "default_glial_trace_mean": dtrace,
                "reversal_glial_trace_mean": rtrace,
                "rule_trace_reversal_minus_default": trscore,
                "default_bridge_strength": dm["bridge_strength"],
                "default_tripartite_mean_strength": dm["tripartite_mean_strength"],
                "default_assembly_min_strength": dm["assembly_min_strength"],
                "default_assembly_all_active": dm["assembly_all_active"],
                "default_direct_NN_mean_strength": dm["direct_nn_mean_strength"],
                "default_direct_NN_all_active": dm["direct_nn_all_active"],
                "reversal_bridge_strength": rm["bridge_strength"],
                "reversal_tripartite_mean_strength": rm["tripartite_mean_strength"],
                "reversal_assembly_min_strength": rm["assembly_min_strength"],
                "reversal_assembly_all_active": rm["assembly_all_active"],
                "reversal_direct_NN_mean_strength": rm["direct_nn_mean_strength"],
                "reversal_direct_NN_all_active": rm["direct_nn_all_active"],
                "active_candidate_edges": sum(1 for st in states.values() if st.active),
            }
        )

        for i in range(base_cfg.n_neurons):
            node_rows.append(
                {
                    "seed": seed,
                    "agent": agent,
                    "trial": t,
                    "node": f"N{i}",
                    "P_active": float(marg[i]),
                    "slow_trace": float(traces[f"N{i}"]),
                }
            )
        for g in range(base_cfg.n_glia):
            node_rows.append(
                {
                    "seed": seed,
                    "agent": agent,
                    "trial": t,
                    "node": f"G{g}",
                    "P_active": float(marg[base_cfg.n_neurons + g]),
                    "slow_trace": float(traces[f"G{g}"]),
                }
            )
        for c in evo.candidates:
            st = states[c.id]
            edge_rows.append(
                {
                    "seed": seed,
                    "agent": agent,
                    "trial": t,
                    "candidate_id": c.id,
                    "role": candidate_role(c.id, assemblies),
                    "edge_type": c.type,
                    "drivers": ";".join(c.drivers),
                    "driver": engine.driver_value(c, traces),
                    "affinity": c.affinity,
                    "strength": float(st.strength),
                    "active": bool(st.active),
                    "effective_weight": float(c.max_weight * st.strength) if st.active else 0.0,
                    "competition_groups": ";".join(c.competition_groups),
                    "generated_from": c.generated_from,
                }
            )

        current = {"stimulus": stim, "action": action, "reward": reward}
        if reward == 1:
            reward_memory.append(dict(current))
        previous = current

        if t == task.reversal_trial - 1:
            probe_rows.extend(
                run_checkpoint_probe(
                    mode, agent, "pre_reversal_learned", seed, core, engine, base_cfg,
                    gains, hw, evo, states, task, assemblies, shots, outdir
                )
            )

        if t < 3 or t in {task.reversal_trial - 1, task.reversal_trial, task.reversal_trial + 1} or (t + 1) % 30 == 0:
            print(
                f"[{seed}:{agent}] t={t:03d} C={ctx[0].upper()} S={stim} A={action} ok={is_correct} R={reward} "
                f"dD={delta:+.4f} traceR-D={trscore:+.3f} D={dm['assembly_min_strength']:.3f} R={rm['assembly_min_strength']:.3f}"
            )

    probe_rows.extend(
        run_checkpoint_probe(
            mode, agent, "post_reversal_learned", seed, core, engine, base_cfg,
            gains, hw, evo, states, task, assemblies, shots, outdir
        )
    )
    return (
        pd.DataFrame(trial_rows),
        pd.DataFrame(node_rows),
        pd.DataFrame(edge_rows),
        pd.DataFrame(emergence),
        pd.DataFrame(probe_rows),
        evo,
    )


def resource_report(core, engine, base_cfg, base_evo, task, assemblies):
    print(f"Runner: {VERSION}; core={getattr(core, 'VERSION', '?')}; engine={getattr(engine, 'VERSION', '?')}")
    print(f"Base: {base_cfg.total_qubits} qubits, {base_cfg.layers} layers")
    print(f"Task: {task.trials} trials; reversal at {task.reversal_trial}; agents={list(task.agents)}")
    print("Hidden context input: FALSE")
    print("Exact candidate_edges supplied: 0 expected")
    print(f"Auto-generated/pruned candidates: {len(base_evo.candidates)}")
    counts = pd.Series([c.type for c in base_evo.candidates]).value_counts().to_dict()
    print("candidate types:", counts)
    print("Derived rule assemblies (diagnostic only):")
    print(json.dumps(assemblies, ensure_ascii=False, indent=2))
    ids = {c.id for c in base_evo.candidates}
    for ctx in ("default", "reversal"):
        a = assemblies[ctx]
        print(f"[{ctx}] bridge present={a['bridge'] in ids}: {a['bridge']}")
        for x in a["tripartites"]:
            print(f"  tri present={x in ids}: {x}")
        for x in a["direct_nn"]:
            print(f"  nn present={x in ids}: {x}")
    print("initial circuit resources:")
    print(json.dumps(core.circuit_resource_estimate(base_cfg), indent=2))
    allstates = {c.id: engine.CandidateState(1.0, True) for c in base_evo.candidates}
    fullcfg = engine.materialize_dynamic_config(core, base_cfg, base_evo.candidates, allstates, [])
    print("all-candidates-active upper-bound resources:")
    print(json.dumps(core.circuit_resource_estimate(fullcfg), indent=2))


def write_manifest(core, engine, bio, base_cfg, gains, base_evo, task, assemblies, reward_suite, outdir, args):
    payload = {
        "runner_version": VERSION,
        "core_version": getattr(core, "VERSION", "unknown"),
        "spatial_engine_version": getattr(engine, "VERSION", "unknown"),
        "scientific_scope": "History-dependent spatial-competitive hypothesis-screening model; across-trial plasticity is classical and rebuilds quantum-circuit analogues.",
        "hidden_context_input": False,
        "future_exact_edges_explicitly_supplied": False,
        "reward_replay": "Only previously rewarded sensory-action outcomes are replayed; no rule/context label is supplied.",
        "reward_eligibility": "Candidate sensory->decision routes include the matching rewarded sensory-action replay trace as a plasticity eligibility driver.",
        "rule_assemblies": "Derived automatically from task/node mapping for diagnostics only; not used by learning.",
        "derived_rule_assemblies": assemblies,
        "task": asdict(task),
        "reward_suite": reward_suite,
        "biological_input": asdict(bio),
        "layer_gains": asdict(gains),
        "spatial_growth": base_evo.spatial_growth,
        "competition": asdict(base_evo.competition),
        "candidate_count_after_role_filter": len(base_evo.candidates),
        "base_network": {
            "total_qubits": base_cfg.total_qubits,
            "layers": base_cfg.layers,
            "neural_readout_mode": getattr(base_cfg, "neural_readout_mode", "population"),
        },
        "n_seeds": int(args.n_seeds),
        "files": {
            "project": str(Path(args.project).resolve()),
            "evolution": str(Path(args.evolution_config).resolve()),
            "task": str(Path(args.task_config).resolve()),
        },
    }
    (outdir / "sample2_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="V7.1 Sample 2 optimized hidden-rule binary-choice reversal")
    ap.add_argument("--core", default=DEFAULT_CORE)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--evolution-config", default=DEFAULT_EVOLUTION)
    ap.add_argument("--task-config", default=DEFAULT_TASK)
    ap.add_argument("--mode", choices=["resource", "surrogate", "local", "qpu"], default="surrogate")
    ap.add_argument("--reward-suite", choices=["deterministic", "probabilistic"], default="deterministic")
    ap.add_argument("--n-seeds", type=int, default=1, help="Matched independent seeds; use >1 for robustness analysis")
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--agents", default=None, help="Optional comma-separated agent override, e.g. full_spatial_competition,frozen_spatial")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be >=1")

    here = Path(__file__).resolve().parent

    def resolve(p):
        q = Path(p)
        return q if q.exists() else here / q

    core = load_module("v71_sample2_core", resolve(args.core))
    engine = load_module("v71_sample2_engine", resolve(args.engine))
    bio, base_cfg, hw, _ = core.project_from_json(resolve(args.project))
    core.validate_biological_input(bio)
    core.validate_network_config(base_cfg)
    gains = core.scores_to_layer_gains(core.compute_mechanistic_scores(bio), bio.hardware_gain)

    base_evo = engine.load_evolution_config(resolve(args.evolution_config))
    if len(base_evo.candidates) != 0:
        raise ValueError("Publication Sample 2 requires candidate_edges=[] in the evolution JSON")
    engine.finalize_spatial_candidates(base_cfg, base_evo)

    task = load_task(resolve(args.task_config))
    if args.agents:
        task.agents = tuple(x.strip() for x in args.agents.split(",") if x.strip())
    validate_task(task)
    prune_task_candidate_pool(base_evo, task)
    add_reward_eligibility_drivers(base_evo, task)
    engine.validate_evolution_config(base_cfg, base_evo)
    assemblies = derive_rule_assemblies(base_cfg, task)

    if args.mode == "resource":
        resource_report(core, engine, base_cfg, base_evo, task, assemblies)
        return 0

    if args.mode == "qpu" and base_evo.qpu_repeats_override is not None:
        hw = replace(hw, repeats=int(base_evo.qpu_repeats_override))
    if args.mode == "qpu" and base_evo.shots_override is not None:
        hw = replace(hw, shots=int(base_evo.shots_override))

    outdir = Path(args.output_dir) if args.output_dir else Path(
        f"V7_1_SAMPLE2_{args.mode}_{args.reward_suite}_{timestamp()}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    save_candidate_catalog(base_evo, assemblies, outdir)
    write_manifest(core, engine, bio, base_cfg, gains, base_evo, task, assemblies, args.reward_suite, outdir, args)
    pd.DataFrame([asdict(core.compute_mechanistic_scores(bio))]).to_csv(outdir / "mechanistic_scores.csv", index=False)
    pd.DataFrame([asdict(gains)]).to_csv(outdir / "layer_gains.csv", index=False)

    all_trials = []
    all_nodes = []
    all_edges = []
    all_em = []
    all_probes = []
    all_env = []

    for rep in range(args.n_seeds):
        seed = int(task.seed + args.seed_offset + 1009 * rep)
        rng = np.random.default_rng(seed)
        stimuli = balanced_stimuli(task.trials, rng)
        reward_u = rng.random(task.trials)
        choice_u = rng.random(task.trials)
        all_env.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "trial": np.arange(task.trials),
                    "stimulus": stimuli,
                    "true_context": [true_context(t, task.reversal_trial) for t in range(task.trials)],
                    "reward_uniform": reward_u,
                    "choice_uniform": choice_u,
                }
            )
        )
        for agent in task.agents:
            print(f"\n=== seed={seed} | {agent} ===")
            td, nd, ed, em, pr, _ = run_agent(
                args.mode,
                agent,
                seed,
                core,
                engine,
                base_cfg,
                gains,
                hw,
                base_evo,
                task,
                assemblies,
                stimuli,
                reward_u,
                choice_u,
                outdir,
                args.reward_suite,
            )
            all_trials.append(td)
            all_nodes.append(nd)
            all_edges.append(ed)
            all_em.append(em)
            all_probes.append(pr)

    trials = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
    nodes = pd.concat(all_nodes, ignore_index=True) if all_nodes else pd.DataFrame()
    edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    ems = pd.concat(all_em, ignore_index=True) if all_em else pd.DataFrame()
    probes = pd.concat(all_probes, ignore_index=True) if all_probes else pd.DataFrame()
    env = pd.concat(all_env, ignore_index=True) if all_env else pd.DataFrame()

    trials.to_csv(outdir / "sample2_trial_log.csv", index=False)
    nodes.to_csv(outdir / "sample2_node_activity_and_slow_trace.csv", index=False)
    edges.to_csv(outdir / "sample2_edge_evolution.csv", index=False)
    ems.to_csv(outdir / "sample2_emergence_events.csv", index=False)
    probes.to_csv(outdir / "sample2_checkpoint_probes.csv", index=False)
    env.to_csv(outdir / "sample2_environment_schedule.csv", index=False)

    probe_metrics = same_stimulus_probe_metrics(probes)
    probe_metrics.to_csv(outdir / "sample2_same_stimulus_flip_summary.csv", index=False)
    seed_summary = summarize(trials, ems, probes, task, assemblies)
    seed_summary.to_csv(outdir / "sample2_agent_seed_summary.csv", index=False)
    gsummary = group_summary(seed_summary)
    gsummary.to_csv(outdir / "sample2_agent_group_summary.csv", index=False)
    effects = paired_control_effects(seed_summary)
    effects.to_csv(outdir / "sample2_paired_control_effects.csv", index=False)
    if not trials.empty:
        topo = trials[[
            "seed", "agent", "trial", "true_context",
            "default_assembly_min_strength", "reversal_assembly_min_strength",
            "default_assembly_all_active", "reversal_assembly_all_active"
        ]].copy()
        topo["current_rule_topology_selectivity"] = np.where(
            topo.trial.to_numpy() < task.reversal_trial,
            topo.default_assembly_min_strength - topo.reversal_assembly_min_strength,
            topo.reversal_assembly_min_strength - topo.default_assembly_min_strength,
        )
        topo.to_csv(outdir / "sample2_rule_topology_timeseries.csv", index=False)
    plots(trials, probes, task, outdir)

    print("\n=== SEED SUMMARY ===")
    print(seed_summary.to_string(index=False))
    if args.n_seeds > 1:
        print("\n=== GROUP SUMMARY ===")
        print(gsummary.to_string(index=False))
    print("\nDONE:", outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
