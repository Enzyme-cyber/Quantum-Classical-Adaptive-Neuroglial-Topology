#!/usr/bin/env python3
"""
V7.1 SAMPLE 3 V3 — EFFORT-DRIVEN META-SELECTION OF AN EMERGENT FEEDBACK PROCEDURE
=================================================================================

Scientific question
-------------------
Can repeated primitive factor-search experience first generate a quotient re-entry
feedback topology, and can a second, effort-sensitive meta-route learn when that
new computational organization should compete with the original flat enumeration
procedure?

Key changes from Sample 3 V3
-----------------------------
1) Explicit experience replay is removed. Slow traces are driven only by actual
   ongoing computation in the current trial.
2) Arithmetic cost is counted transparently in unit primitive operations:
      - one divisor test = 1;
      - quotient-gate passage = 1;
      - quotient-buffer transfer = 1;
      - re-entry-edge transmission = 1.
3) Every problem begins with a bounded prefix of the original flat enumeration.
   The number of actual divisor tests already spent in this prefix is encoded as
   CURRENT_ENUMERATION_EFFORT; no N, log2(N), or hand-labelled complexity class is
   supplied to the circuit.
4) A second emergent tripartite route N4 + G2 -> N9 converts current computational
   effort, in the context of a learned slow effort state, into a feedback-strategy
   proposal. N8 remains the pre-existing enumeration proposal. These two routes
   compete for one execution resource.
5) The feedback procedure itself remains graph-executed: quotient re-entry occurs
   only when the learned N2+G1->N7 gate and N7->N0 edge close the cycle.

The primary V3 prediction is a true strategy crossover on held-out problems:
small/cheap cases should remain on enumeration, whereas expensive decomposable
cases should switch to the learned feedback graph. Removing strategy competition
should preserve correctness but execute both routes and increase cost.

Scope
-----
This is a hybrid hypothesis-screening model, not Shor's algorithm and not evidence
for quantum speedup or literal biological qubits. Across-trial plasticity is
classical; local/QPU modes use the V7.1 quantum-formal fast circuit. Surrogate mode
is only a deterministic software smoke test.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import zlib
from collections import deque
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

VERSION = "V7.1-SAMPLE3-FACTOR-STRATEGY-3.0"
DEFAULT_CORE = "V7_1_neuroglial_gate_qpu.py"
DEFAULT_ENGINE = "V7_1_SPATIAL_EVOLVING_NETWORK.py"
DEFAULT_PROJECT = "V7_1_SAMPLE3_FACTOR_STRATEGY_V3_project.json"
DEFAULT_EVOLUTION = "V7_1_SAMPLE3_FACTOR_STRATEGY_V3_evolution.json"
DEFAULT_TASK = "V7_1_SAMPLE3_FACTOR_STRATEGY_V3_task_PRIMARY.json"
EPS = 1e-12


@dataclass
class TaskSettings:
    name: str = "Sample 3 V3 effort-driven meta-selection"
    seed: int = 20260814
    early_numbers: tuple[int, ...] = (4, 6, 8, 9, 10, 16, 25, 49)
    early_repeats: int = 4
    structure_numbers: tuple[int, ...] = (180, 240, 300, 360, 480, 504, 540, 600, 720, 756, 840, 900)
    structure_repeats: int = 4
    power_numbers: tuple[int, ...] = (64, 128, 256, 512, 768)
    power_repeats: int = 3
    holdout_numbers: tuple[int, ...] = (4, 9, 25, 49, 420, 630, 945, 1024, 997)
    current_success_probability: float = 0.93
    current_quotient_probability: float = 0.96
    current_terminal_probability: float = 0.72
    enum_prior_probability: float = 0.78
    effort_probe_budget: int = 8
    effort_event_gain: float = 1.0
    exploration: float = 0.04
    quotient_gate_cost: float = 1.0
    buffer_transfer_cost: float = 1.0
    reentry_edge_cost: float = 1.0
    strategy_activation_threshold: float = 0.045
    token_hop_limit: int = 128
    agents: tuple[str, ...] = (
        "full_competition",
        "no_strategy_competition",
        "no_topology_competition",
        "frozen_spatial",
        "no_slow_memory",
        "no_glia_glia",
        "no_quotient_gate",
        "no_reentry_edge",
        "no_effort_signal",
        "no_meta_gate",
        "constant_effort_signal",
        "equal_affinity",
        "shuffled_affinity",
    )


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_int(text: str) -> int:
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


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
    kw = {k: d[k] for k in TaskSettings.__dataclass_fields__ if k in d}
    for k in ("early_numbers", "structure_numbers", "power_numbers", "holdout_numbers", "agents"):
        if k in kw:
            kw[k] = tuple(kw[k])
    return TaskSettings(**kw)


def validate_task(t: TaskSettings):
    nums = list(t.early_numbers) + list(t.structure_numbers) + list(t.power_numbers) + list(t.holdout_numbers)
    if not nums or min(nums) < 2:
        raise ValueError("All task integers must be >=2")
    for x in (
        t.current_success_probability, t.current_quotient_probability,
        t.current_terminal_probability, t.enum_prior_probability,
        t.effort_event_gain, t.exploration,
    ):
        if not 0 <= float(x) <= 1:
            raise ValueError("probabilities/gains/exploration must be in [0,1]")
    if int(t.effort_probe_budget) < 1:
        raise ValueError("effort_probe_budget must be >=1")
    if min(t.quotient_gate_cost, t.buffer_transfer_cost, t.reentry_edge_cost) < 0:
        raise ValueError("primitive topology costs must be >=0")
    if t.token_hop_limit < 8:
        raise ValueError("token_hop_limit must be >=8")


# =============================================================================
# Arithmetic primitives and POST-HOC scorer
# =============================================================================

def reference_prime_factors(n: int) -> list[int]:
    """POST-HOC scorer only. Never used to control topology or route selection."""
    q = int(n)
    out: list[int] = []
    d = 2
    while d * d <= q:
        while q % d == 0:
            out.append(d)
            q //= d
        d += 1
    if q > 1:
        out.append(q)
    return out


def is_prime_trial(x: int) -> tuple[bool, int]:
    """Trial-divisibility primality check used only by the fixed flat baseline."""
    x = int(x)
    if x < 2:
        return False, 0
    tests = 0
    for d in range(2, math.isqrt(x) + 1):
        tests += 1
        if x % d == 0:
            return False, tests
    return True, tests


def flat_enumeration_baseline(n: int) -> dict[str, Any]:
    """Fixed flat factor-pair scan on the ORIGINAL N, with no quotient re-entry.

    The baseline tests every candidate d=2..floor(sqrt(N)) against the original
    N and records divisor pairs (d, N/d). Prime divisors are then identified by
    trial divisibility, and multiplicity is obtained by testing powers p^k
    against the original N. No quotient token is fed back into FACTOR_TEST.
    """
    n = int(n)
    tests = 0
    raw_candidates: set[int] = set()
    successful_divisors = 0
    for d in range(2, math.isqrt(n) + 1):
        tests += 1
        if n % d == 0:
            successful_divisors += 1
            raw_candidates.add(int(d))
            raw_candidates.add(int(n // d))

    if not raw_candidates:
        # Exhausting all d <= sqrt(N) is itself a primality certificate under
        # this finite task environment; no separate reference factorizer is used.
        factors = [n]
        return {
            "route": "enumeration", "factors": factors,
            "primitive_cost": float(tests), "successful_divisions": 0,
            "nonterminal_quotients": 0, "reentries": 0, "complete": True,
        }

    prime_divisors: list[int] = []
    for d in sorted(raw_candidates):
        prime, pcost = is_prime_trial(d)
        tests += int(pcost)
        if prime:
            prime_divisors.append(int(d))

    factors: list[int] = []
    for p0 in sorted(set(prime_divisors)):
        power = int(p0)
        mult = 0
        while power <= n:
            tests += 1
            if n % power != 0:
                break
            mult += 1
            if power > n // p0:
                break
            power *= int(p0)
        factors.extend([int(p0)] * int(mult))

    factors.sort()
    return {
        "route": "enumeration", "factors": factors,
        "primitive_cost": float(tests),
        "successful_divisions": int(successful_divisors),
        "nonterminal_quotients": int(successful_divisors),
        "reentries": 0, "complete": True,
    }

def primitive_first_divisor_step(q: int) -> dict[str, Any]:
    """One invocation of the reusable FACTOR_TEST_OPERATOR.

    It finds the first proper divisor of the current token q by ascending trial
    division. If none exists, q is terminal (prime under this primitive).
    This is a single reusable primitive, not a recursive decomposition routine.
    """
    q = int(q)
    tests = 0
    for d in range(2, math.isqrt(q) + 1):
        tests += 1
        if q % d == 0:
            return {
                "input": q,
                "factor": int(d),
                "quotient": int(q // d),
                "proper_divisor": True,
                "tests": int(tests),
            }
    return {
        "input": q,
        "factor": int(q),
        "quotient": 1,
        "proper_divisor": False,
        "tests": int(tests),
    }


def first_task_event(n: int) -> dict[str, Any]:
    s = primitive_first_divisor_step(int(n))
    return {
        "N": int(n),
        "success": int(bool(s["proper_divisor"])),
        "nonterminal": int(bool(s["proper_divisor"] and int(s["quotient"]) > 1)),
        "terminal": int(not bool(s["proper_divisor"])),
        "first_factor": int(s["factor"]),
        "first_quotient": int(s["quotient"]),
        "first_step_tests": int(s["tests"]),
    }


# =============================================================================
# Generic token-flow execution: recursion arises only from a graph cycle
# =============================================================================

def emergent_graph_flags(states: dict[str, Any], assembly: dict[str, str]) -> dict[str, bool]:
    def active(role: str) -> bool:
        cid = assembly[role]
        return bool(cid in states and states[cid].active)
    return {
        "success_bridge": active("success_bridge"),
        "quotient_gate": active("quotient_gate"),
        "reentry_edge": active("reentry_edge"),
    }


def execute_token_flow(
    n: int,
    graph_flags: dict[str, bool],
    quotient_gate_cost: float,
    buffer_transfer_cost: float,
    reentry_edge_cost: float,
    hop_limit: int,
) -> dict[str, Any]:
    """Generic graph interpreter; repeated operator reuse exists only if the graph closes.

    Arithmetic and routing costs are transparent unit operations. The first divisor
    primitive is reusable but not recursive by itself. Recursion appears only when
    quotient-gate, buffer transfer and re-entry topology return a quotient token to N0.
    """
    n = int(n)
    q: deque[tuple[str, int]] = deque([("FACTOR_TEST_OPERATOR", n)])
    factors: list[int] = []
    primitive_cost = 0.0
    reentries = 0
    trace: list[dict[str, Any]] = []
    hops = 0
    complete = False

    while q and hops < int(hop_limit):
        node, value = q.popleft()
        hops += 1
        trace.append({"hop": hops, "node": node, "value": int(value), "event": "enter"})

        if node == "FACTOR_TEST_OPERATOR":
            step = primitive_first_divisor_step(value)
            primitive_cost += float(step["tests"])
            factors.append(int(step["factor"]))
            trace.append({
                "hop": hops, "node": node, "value": int(value), "event": "factor_test",
                "factor": int(step["factor"]), "quotient": int(step["quotient"]),
                "tests": int(step["tests"]), "cost_added": float(step["tests"]),
            })
            if int(step["quotient"]) == 1:
                q.append(("DONE", 1))
            else:
                q.append(("NONTERMINAL_QUOTIENT", int(step["quotient"])))

        elif node == "NONTERMINAL_QUOTIENT":
            if graph_flags.get("quotient_gate", False):
                primitive_cost += float(quotient_gate_cost)
                q.append(("QUOTIENT_BUFFER", int(value)))
                trace.append({"hop": hops, "node": node, "value": int(value), "event": "gate_pass",
                              "cost_added": float(quotient_gate_cost)})
            else:
                trace.append({"hop": hops, "node": node, "value": int(value), "event": "gate_block", "cost_added": 0.0})

        elif node == "QUOTIENT_BUFFER":
            if graph_flags.get("reentry_edge", False):
                primitive_cost += float(buffer_transfer_cost) + float(reentry_edge_cost)
                reentries += 1
                q.append(("FACTOR_TEST_OPERATOR", int(value)))
                trace.append({"hop": hops, "node": node, "value": int(value), "event": "buffer_transfer",
                              "cost_added": float(buffer_transfer_cost)})
                trace.append({"hop": hops, "node": node, "value": int(value), "event": "reenter",
                              "cost_added": float(reentry_edge_cost)})
            else:
                trace.append({"hop": hops, "node": node, "value": int(value), "event": "reentry_block", "cost_added": 0.0})

        elif node == "DONE":
            complete = True
            trace.append({"hop": hops, "node": node, "value": int(value), "event": "done", "cost_added": 0.0})

    ref = reference_prime_factors(n)
    correct = bool(complete and factors == ref)
    return {
        "route": "feedback_graph",
        "factors": factors,
        "primitive_cost": float(primitive_cost),
        "successful_divisions": int(len(factors)),
        "nonterminal_quotients": int(reentries),
        "reentries": int(reentries),
        "complete": bool(complete),
        "correct": bool(correct),
        "token_trace": trace,
    }


def ideal_feedback_benchmark(n: int, task: TaskSettings) -> dict[str, Any]:
    """POST-HOC benchmark using the same graph interpreter with feedback edges ON."""
    return execute_token_flow(
        int(n), {"success_bridge": True, "quotient_gate": True, "reentry_edge": True},
        task.quotient_gate_cost, task.buffer_transfer_cost, task.reentry_edge_cost,
        task.token_hop_limit,
    )


# =============================================================================
# Network-role helpers
# =============================================================================

def node_roles() -> dict[str, dict[str, str]]:
    return {
        "neurons": {
            "N0": "FACTOR_TEST_OPERATOR",
            "N1": "SUCCESSFUL_DIVISION_EVENT",
            "N2": "NONTERMINAL_QUOTIENT_EVENT",
            "N3": "TERMINAL_QUOTIENT_EVENT",
            "N4": "CURRENT_ENUMERATION_EFFORT",
            "N5": "META_DISTRACTOR_ROUTE",
            "N6": "ENUMERATION_PRIOR",
            "N7": "QUOTIENT_BUFFER",
            "N8": "ENUMERATION_STRATEGY_ROUTE",
            "N9": "FEEDBACK_STRATEGY_ROUTE",
        },
        "glia": {
            "G0": "DIVISION_SUCCESS_TRACE",
            "G1": "QUOTIENT_CONTINUATION_TRACE",
            "G2": "EFFORT_HISTORY_TRACE",
            "G3": "RESERVED",
            "G4": "SUCCESS_DISTRACTOR",
            "G5": "QUOTIENT_DISTRACTOR",
            "G6": "EFFORT_DISTRACTOR",
            "G7": "RESERVED",
            "G8": "RESERVED",
            "G9": "RESERVED",
        },
    }


def assembly_ids() -> dict[str, str]:
    return {
        "success_bridge": "AUTO_GG_G0_G1",
        "quotient_gate": "AUTO_TRI_N2_G1_N7",
        "reentry_edge": "AUTO_NN_G1_N7_N0",
        "effort_meta_gate": "AUTO_TRI_N4_G2_N9",
    }


def candidate_role(cid: str, assembly: dict[str, str]) -> str:
    for role, x in assembly.items():
        if cid == x:
            return role
    if any(tok in cid for tok in ("G4", "G5", "G6", "N5")):
        return "distractor"
    return "other_local_candidate"


def assembly_metrics(states: dict[str, Any], assembly: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    execution_roles = ("success_bridge", "quotient_gate", "reentry_edge")
    exec_vals, exec_acts = [], []
    for role, cid in assembly.items():
        st = states.get(cid)
        out[f"{role}_strength"] = float(st.strength) if st is not None else np.nan
        out[f"{role}_active"] = bool(st.active) if st is not None else False
        if role in execution_roles and st is not None:
            exec_vals.append(float(st.strength)); exec_acts.append(bool(st.active))
    out["feedback_assembly_min_strength"] = float(min(exec_vals)) if exec_vals else np.nan
    out["feedback_assembly_all_active"] = bool(len(exec_acts) == len(execution_roles) and all(exec_acts))
    meta_st = states.get(assembly["effort_meta_gate"])
    out["effort_meta_gate_strength"] = float(meta_st.strength) if meta_st is not None else np.nan
    out["effort_meta_gate_active"] = bool(meta_st.active) if meta_st is not None else False
    out["full_meta_architecture_active"] = bool(out["feedback_assembly_all_active"] and out["effort_meta_gate_active"])
    return out


def shuffled_affinity_copy(evo, seed: int):
    rng = np.random.default_rng(seed)
    groups: dict[str, list[Any]] = {}
    for c in evo.candidates:
        groups.setdefault(c.type, []).append(c)
    for cs in groups.values():
        vals = np.asarray([float(c.affinity) for c in cs], dtype=float)
        rng.shuffle(vals)
        for c, v in zip(cs, vals):
            c.affinity = float(v)
    return evo


def agent_evolution(engine, base_cfg, base_evo, agent: str, seed: int, assembly: dict[str, str]):
    evo = copy.deepcopy(base_evo)
    if agent == "no_topology_competition":
        evo.competition = replace(evo.competition, enabled=False)
    elif agent == "no_glia_glia":
        evo.candidates = [c for c in evo.candidates if c.type != "glia_glia"]
    elif agent == "no_quotient_gate":
        evo.candidates = [c for c in evo.candidates if c.id != assembly["quotient_gate"]]
    elif agent == "no_reentry_edge":
        evo.candidates = [c for c in evo.candidates if c.id != assembly["reentry_edge"]]
    elif agent == "no_meta_gate":
        evo.candidates = [c for c in evo.candidates if c.id != assembly["effort_meta_gate"]]
    elif agent == "equal_affinity":
        for c in evo.candidates:
            c.affinity = 1.0
    elif agent == "shuffled_affinity":
        shuffled_affinity_copy(evo, seed + stable_int(agent))
    elif agent in {
        "full_competition", "no_strategy_competition", "frozen_spatial",
        "no_slow_memory", "no_effort_signal", "constant_effort_signal",
    }:
        pass
    else:
        raise ValueError(f"Unknown agent {agent}")
    ids = {c.id for c in evo.candidates}
    for c in evo.candidates:
        c.activation_requires = [x for x in c.activation_requires if x in ids]
    engine.validate_evolution_config(base_cfg, evo)
    return evo


# =============================================================================
# Curriculum and primitive event encoding
# =============================================================================

def make_schedule(task: TaskSettings, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    def add(name: str, nums: tuple[int, ...], reps: int):
        for ep in range(int(reps)):
            arr = np.asarray(nums, dtype=int).copy(); rng.shuffle(arr)
            for n in arr:
                rows.append({"phase": name, "epoch": ep + 1, "N": int(n)})
    add("early_naive", task.early_numbers, task.early_repeats)
    add("structural_experience", task.structure_numbers, task.structure_repeats)
    add("power_chain", task.power_numbers, task.power_repeats)
    for i, r in enumerate(rows):
        r["trial"] = i
    return rows


def enumeration_effort_probe(n: int, task: TaskSettings) -> dict[str, Any]:
    """Run only a bounded prefix of the original flat divisor scan.

    This is the only task-difficulty signal available to the meta-selector. It is
    measured in actual primitive tests already spent, not computed from N or log N.
    """
    n = int(n)
    budget = int(task.effort_probe_budget)
    tests = 0
    d = 2
    tested_divisors: list[int] = []
    # Deliberately perform the same bounded candidate loop used by flat trial
    # division rather than deriving a complexity label from N.
    while tests < budget and d * d <= n:
        tested_divisors.append(int(d))
        tests += 1
        d += 1
    return {
        "tests": int(tests),
        "budget": int(budget),
        "signal": float(tests / max(1, budget)),
        "tested_divisors": tested_divisors,
    }


def effort_signal_for_agent(agent: str, raw_signal: float) -> float:
    if agent == "no_effort_signal":
        return 0.0
    if agent == "constant_effort_signal":
        return 0.65
    return clip(float(raw_signal), 0.0, 1.0)


def build_trial_events(engine, core, gains, current: dict[str, Any], effort_signal: float, task: TaskSettings):
    spec: list[dict[str, Any]] = [
        {"layer": 0, "neuron": 0, "probability": 0.90, "event_kind": "evidence"},
        {"layer": 1, "neuron": 6, "probability": task.enum_prior_probability, "event_kind": "tonic"},
    ]
    if current["success"]:
        spec.append({"layer": 0, "neuron": 1, "probability": task.current_success_probability, "event_kind": "evidence"})
    if current["nonterminal"]:
        spec.append({"layer": 0, "neuron": 2, "probability": task.current_quotient_probability, "event_kind": "evidence"})
    if current["terminal"]:
        spec.append({"layer": 0, "neuron": 3, "probability": task.current_terminal_probability, "event_kind": "evidence"})
    pe = clip(float(effort_signal) * float(task.effort_event_gain), 0.0, 0.98)
    if pe > 0:
        spec.append({"layer": 0, "neuron": 4, "probability": pe, "event_kind": "evidence"})
    return engine.build_events(core, spec, gains)


def build_probe_events(engine, core, gains, current: dict[str, Any], effort_signal: float, task: TaskSettings):
    return build_trial_events(engine, core, gains, current, effort_signal, task)


# =============================================================================
# Competition and execution
# =============================================================================

def wta_feedback_probability(p_enum: float, p_feedback: float, task: TaskSettings) -> float:
    d = float(p_feedback) - float(p_enum)
    if abs(d) < 1e-9:
        raw = 0.5
    else:
        raw = 1.0 if d > 0 else 0.0
    return float((1.0 - task.exploration) * raw + task.exploration * 0.5)


def route_outputs_for_n(n: int, flags: dict[str, bool], task: TaskSettings):
    enum = flat_enumeration_baseline(int(n))
    feedback = execute_token_flow(
        int(n), flags, task.quotient_gate_cost, task.buffer_transfer_cost,
        task.reentry_edge_cost, task.token_hop_limit,
    )
    ideal_feedback = ideal_feedback_benchmark(int(n), task)
    ref = reference_prime_factors(int(n))
    enum_correct = bool(enum["factors"] == ref)
    feedback_correct = bool(feedback["correct"])
    ideal_correct = bool(ideal_feedback["correct"])
    return enum, feedback, ideal_feedback, enum_correct, feedback_correct, ideal_correct


def execute_route_choice(
    agent: str,
    n: int,
    p_enum: float,
    p_meta_feedback: float,
    flags: dict[str, bool],
    effort_probe_cost: float,
    u: float,
    task: TaskSettings,
) -> dict[str, Any]:
    enum, feedback, ideal_feedback, enum_correct, feedback_correct, ideal_correct = route_outputs_for_n(n, flags, task)
    current = first_task_event(n)
    feedback_available = bool(flags["quotient_gate"] and flags["reentry_edge"] and current["nonterminal"])
    meta_available = bool(flags.get("effort_meta_gate", False))

    # Enumeration already contains the bounded prefix, so its total cost is not
    # increased by the effort observation. Switching to feedback pays the prefix
    # as sunk work and then executes the feedback graph from the original input.
    enum_total = float(enum["primitive_cost"])
    feedback_total = float(effort_probe_cost) + float(feedback["primitive_cost"]) if feedback_correct else np.nan
    ideal_feedback_total = float(effort_probe_cost) + float(ideal_feedback["primitive_cost"])

    can_propose_feedback = bool(feedback_available and meta_available)
    if not can_propose_feedback:
        executed = [enum]
        chosen = "enumeration"
        p_fb = 0.0
        total_cost = enum_total
    elif agent == "no_strategy_competition":
        active_fb = float(p_meta_feedback) >= float(task.strategy_activation_threshold)
        if active_fb:
            executed = [enum, feedback]
            chosen = "both"
            p_fb = 1.0
            # The effort prefix is shared with enumeration and is not double-counted.
            total_cost = enum_total + float(feedback["primitive_cost"])
        else:
            executed = [enum]
            chosen = "enumeration"
            p_fb = 0.0
            total_cost = enum_total
    else:
        p_fb = wta_feedback_probability(p_enum, p_meta_feedback, task)
        if float(u) < p_fb:
            executed = [feedback]
            chosen = "feedback_graph"
            total_cost = feedback_total
        else:
            executed = [enum]
            chosen = "enumeration"
            total_cost = enum_total

    correct = int(all(bool(x.get("complete", True)) and x["factors"] == reference_prime_factors(int(n)) for x in executed))
    candidate_costs = [enum_total]
    if ideal_correct and current["nonterminal"]:
        candidate_costs.append(ideal_feedback_total)
    best_cost = float(min(candidate_costs))
    regret = float(total_cost - best_cost)
    optimal_use = int(abs(total_cost - best_cost) < 1e-9)
    optimal = "feedback_graph" if (ideal_correct and current["nonterminal"] and ideal_feedback_total + 1e-9 < enum_total) else "enumeration"

    return {
        "chosen_strategy": chosen,
        "executed": executed,
        "correct": correct,
        "P_choose_feedback": float(p_fb),
        "enum_cost": enum_total,
        "feedback_graph_cost": float(feedback["primitive_cost"]) if feedback_correct else np.nan,
        "feedback_total_cost": feedback_total,
        "ideal_feedback_total_cost": ideal_feedback_total,
        "effort_probe_cost": float(effort_probe_cost),
        "executed_cost": float(total_cost),
        "best_cost": best_cost,
        "regret": regret,
        "optimal_strategy": optimal,
        "optimal_strategy_use": optimal_use,
        "factors": reference_prime_factors(int(n)),
        "feedback_available": feedback_available,
        "meta_available": meta_available,
        "feedback_complete": bool(feedback["complete"]),
        "feedback_reentries": int(feedback["reentries"]),
        "feedback_token_trace": feedback["token_trace"],
    }


# =============================================================================
# One agent
# =============================================================================

def run_agent(mode, agent, seed, core, engine, base_cfg, gains, hw, base_evo, task, assembly, schedule, outdir):
    evo = agent_evolution(engine, base_cfg, base_evo, agent, seed, assembly)
    traces = engine.initialize_traces(base_cfg, evo)
    states = engine.initialize_states(evo)
    shots = int(evo.shots_override or hw.shots or 2500)
    if mode == "local" and evo.shots_override is None:
        shots = max(1000, min(10000, int(getattr(hw, "shots", 1000)) * 3))

    trials, edges, emergence, token_rows = [], [], [], []
    rng_choice = np.random.default_rng(seed + stable_int(agent) + 73031)

    for rec in schedule:
        t = int(rec["trial"]); n = int(rec["N"])
        current = first_task_event(n)
        probe = enumeration_effort_probe(n, task)
        effort_signal = effort_signal_for_agent(agent, probe["signal"])
        events = build_trial_events(engine, core, gains, current, effort_signal, task)
        cfg = engine.materialize_dynamic_config(core, base_cfg, evo.candidates, states, events)
        _, marg = engine.run_trial_backend(
            mode, core, cfg, gains, shots, hw, outdir,
            f"seed{seed}_{agent}_trial_{t:03d}_N{n}", readout_mode="population"
        )
        am_before = assembly_metrics(states, assembly)
        flags_before = emergent_graph_flags(states, assembly)
        flags_before["effort_meta_gate"] = bool(am_before["effort_meta_gate_active"])
        p_enum = float(marg[8])
        p_meta = float(marg[9])
        choice = execute_route_choice(
            agent, n, p_enum, p_meta, flags_before, float(probe["tests"]),
            float(rng_choice.random()), task
        )

        if agent != "frozen_spatial":
            engine.update_traces(traces, marg, cfg, evo.trace)
            if agent == "no_slow_memory":
                for k in traces:
                    traces[k] = 0.0
            evs = engine.update_candidate_states(evo.candidates, states, traces, evo.competition)
            for e in evs:
                emergence.append({
                    "seed": seed, "agent": agent, "trial": t, "phase": rec["phase"], "N": n,
                    "role": candidate_role(e["candidate_id"], assembly), **e
                })

        am_after = assembly_metrics(states, assembly)
        trial = {
            "seed": seed, "agent": agent, "trial": t, "phase": rec["phase"], "epoch": rec["epoch"], "N": n,
            "current_success": current["success"], "current_nonterminal_quotient": current["nonterminal"],
            "first_factor": current["first_factor"], "first_quotient": current["first_quotient"],
            "first_step_tests": current["first_step_tests"],
            "effort_probe_tests": int(probe["tests"]), "effort_signal_raw": float(probe["signal"]),
            "effort_signal_used": float(effort_signal),
            "P_ENUM_ROUTE": p_enum, "P_FEEDBACK_STRATEGY": p_meta,
            "feedback_available_before_update": bool(choice["feedback_available"]),
            "meta_available_before_update": bool(choice["meta_available"]),
            "chosen_strategy": choice["chosen_strategy"], "P_choose_feedback": choice["P_choose_feedback"],
            "optimal_strategy": choice["optimal_strategy"], "optimal_strategy_use": choice["optimal_strategy_use"],
            "correct": choice["correct"], "factors": "*".join(map(str, choice["factors"])),
            "enum_cost": choice["enum_cost"], "feedback_graph_cost": choice["feedback_graph_cost"],
            "feedback_total_cost": choice["feedback_total_cost"],
            "ideal_feedback_total_cost": choice["ideal_feedback_total_cost"],
            "executed_cost": choice["executed_cost"], "best_cost": choice["best_cost"], "regret": choice["regret"],
            "feedback_reentries": choice["feedback_reentries"],
            "active_candidate_edges": sum(1 for st in states.values() if st.active),
            **am_after,
        }
        trials.append(trial)

        for x in choice["feedback_token_trace"]:
            token_rows.append({"seed": seed, "agent": agent, "stage": "training", "trial": t, "N": n, **x})

        for c in evo.candidates:
            st = states[c.id]
            edges.append({
                "seed": seed, "agent": agent, "trial": t, "phase": rec["phase"], "N": n,
                "candidate_id": c.id, "role": candidate_role(c.id, assembly), "edge_type": c.type,
                "drivers": ";".join(c.drivers), "driver": engine.driver_value(c, traces),
                "affinity": c.affinity, "strength": float(st.strength), "active": bool(st.active),
                "competition_groups": ";".join(c.competition_groups), "generated_from": c.generated_from,
            })

        if t < 3 or (t + 1) % 25 == 0 or am_after["full_meta_architecture_active"] != am_before["full_meta_architecture_active"]:
            print(
                f"[{seed}:{agent}] t={t:03d} N={n:4d} effort={effort_signal:.3f} "
                f"{choice['chosen_strategy']:<14s} cost={choice['executed_cost']:.2f}/{choice['best_cost']:.2f} "
                f"P(E,M)=({p_enum:.3f},{p_meta:.3f}) fb={am_after['feedback_assembly_min_strength']:.3f} "
                f"meta={am_after['effort_meta_gate_strength']:.3f} full={int(am_after['full_meta_architecture_active'])}"
            )

    holdout_rows = []
    rng_probe = np.random.default_rng(seed + stable_int(agent) + 99121)
    for n in task.holdout_numbers:
        current = first_task_event(int(n))
        probe = enumeration_effort_probe(int(n), task)
        effort_signal = effort_signal_for_agent(agent, probe["signal"])
        events = build_probe_events(engine, core, gains, current, effort_signal, task)
        cfg = engine.materialize_dynamic_config(core, base_cfg, evo.candidates, states, events)
        _, marg = engine.run_trial_backend(
            mode, core, cfg, gains, shots, hw, outdir,
            f"seed{seed}_{agent}_holdout_N{int(n)}", readout_mode="population"
        )
        am = assembly_metrics(states, assembly)
        flags = emergent_graph_flags(states, assembly)
        flags["effort_meta_gate"] = bool(am["effort_meta_gate_active"])
        choice = execute_route_choice(
            agent, int(n), float(marg[8]), float(marg[9]), flags,
            float(probe["tests"]), float(rng_probe.random()), task
        )
        holdout_rows.append({
            "seed": seed, "agent": agent, "N": int(n),
            "current_success": current["success"], "current_nonterminal_quotient": current["nonterminal"],
            "first_factor": current["first_factor"], "first_quotient": current["first_quotient"],
            "effort_probe_tests": int(probe["tests"]), "effort_signal_raw": float(probe["signal"]),
            "effort_signal_used": float(effort_signal),
            "P_ENUM_ROUTE": float(marg[8]), "P_FEEDBACK_STRATEGY": float(marg[9]),
            "feedback_available": bool(choice["feedback_available"]), "meta_available": bool(choice["meta_available"]),
            "chosen_strategy": choice["chosen_strategy"], "P_choose_feedback": choice["P_choose_feedback"],
            "optimal_strategy": choice["optimal_strategy"], "optimal_strategy_use": choice["optimal_strategy_use"],
            "correct": choice["correct"], "factors": "*".join(map(str, choice["factors"])),
            "enum_cost": choice["enum_cost"], "feedback_graph_cost": choice["feedback_graph_cost"],
            "feedback_total_cost": choice["feedback_total_cost"],
            "ideal_feedback_total_cost": choice["ideal_feedback_total_cost"],
            "executed_cost": choice["executed_cost"], "best_cost": choice["best_cost"], "regret": choice["regret"],
            "feedback_reentries": choice["feedback_reentries"],
            **am,
        })
        for x in choice["feedback_token_trace"]:
            token_rows.append({"seed": seed, "agent": agent, "stage": "holdout", "trial": np.nan, "N": int(n), **x})

    return (
        pd.DataFrame(trials), pd.DataFrame(edges), pd.DataFrame(emergence),
        pd.DataFrame(holdout_rows), pd.DataFrame(token_rows), evo
    )


# =============================================================================
# Summaries
# =============================================================================

def first_complete_trial(df: pd.DataFrame, column: str) -> float:
    q = df[df[column] == True]
    return float(q.trial.min()) if not q.empty else np.nan


def summarize_seed_agent(trials: pd.DataFrame, holdout: pd.DataFrame) -> dict[str, Any]:
    late = trials.tail(min(30, len(trials)))
    feedback_better = holdout[holdout.optimal_strategy == "feedback_graph"]
    enum_better = holdout[holdout.optimal_strategy == "enumeration"]
    fb_rate = float(np.mean(feedback_better.chosen_strategy == "feedback_graph")) if len(feedback_better) else np.nan
    en_rate = float(np.mean(enum_better.chosen_strategy == "enumeration")) if len(enum_better) else np.nan
    return {
        "seed": int(trials.seed.iloc[0]),
        "agent": str(trials.agent.iloc[0]),
        "training_trials": len(trials),
        "feedback_topology_complete_trial": first_complete_trial(trials, "feedback_assembly_all_active"),
        "full_meta_architecture_complete_trial": first_complete_trial(trials, "full_meta_architecture_active"),
        "late_feedback_choice_rate": float(np.mean(late.chosen_strategy == "feedback_graph")) if len(late) else np.nan,
        "late_both_choice_rate": float(np.mean(late.chosen_strategy == "both")) if len(late) else np.nan,
        "late_mean_regret": float(late.regret.mean()) if len(late) else np.nan,
        "holdout_correct_rate": float(holdout.correct.mean()) if len(holdout) else np.nan,
        "holdout_optimal_strategy_rate": float(holdout.optimal_strategy_use.mean()) if len(holdout) else np.nan,
        "holdout_mean_cost": float(holdout.executed_cost.mean()) if len(holdout) else np.nan,
        "holdout_mean_best_cost": float(holdout.best_cost.mean()) if len(holdout) else np.nan,
        "holdout_mean_regret": float(holdout.regret.mean()) if len(holdout) else np.nan,
        "feedback_better_holdout_feedback_rate": fb_rate,
        "enum_better_holdout_enum_rate": en_rate,
        "strategy_crossover_accuracy": float(np.nanmean([fb_rate, en_rate])),
        "meta_selectivity_index": float(fb_rate - (1.0-en_rate)) if np.isfinite(fb_rate) and np.isfinite(en_rate) else np.nan,
        "feedback_better_holdout_mean_regret": float(feedback_better.regret.mean()) if len(feedback_better) else np.nan,
        "enum_better_holdout_mean_regret": float(enum_better.regret.mean()) if len(enum_better) else np.nan,
        "final_feedback_assembly_min_strength": float(trials.iloc[-1].feedback_assembly_min_strength),
        "final_feedback_assembly_all_active": bool(trials.iloc[-1].feedback_assembly_all_active),
        "final_effort_meta_gate_strength": float(trials.iloc[-1].effort_meta_gate_strength),
        "final_effort_meta_gate_active": bool(trials.iloc[-1].effort_meta_gate_active),
    }


def aggregate_group(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in seed_summary.columns if c not in {"seed", "agent"} and pd.api.types.is_numeric_dtype(seed_summary[c])]
    rows = []
    for agent, g in seed_summary.groupby("agent", sort=False):
        row = {"agent": agent, "n_seeds": int(g.seed.nunique())}
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{m}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(seed_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "holdout_optimal_strategy_rate", "holdout_mean_cost", "holdout_mean_regret",
        "feedback_better_holdout_feedback_rate", "enum_better_holdout_enum_rate",
        "strategy_crossover_accuracy", "meta_selectivity_index",
        "final_feedback_assembly_min_strength", "final_effort_meta_gate_strength",
    ]
    rows = []
    for seed, g in seed_summary.groupby("seed"):
        f = g[g.agent == "full_competition"]
        if f.empty:
            continue
        f = f.iloc[0]
        for _, r in g.iterrows():
            if r.agent == "full_competition":
                continue
            d = {"seed": int(seed), "control": r.agent}
            for m in metrics:
                fv, cv = f[m], r[m]
                d[f"full_minus_control_{m}"] = float(fv - cv) if pd.notna(fv) and pd.notna(cv) else np.nan
            rows.append(d)
    return pd.DataFrame(rows)


def save_candidate_catalog(evo, assembly, outdir):
    rows = []
    for c in evo.candidates:
        d = asdict(c)
        d["drivers"] = ";".join(c.drivers)
        d["activation_requires"] = ";".join(c.activation_requires)
        d["competition_groups"] = ";".join(c.competition_groups)
        d["role"] = candidate_role(c.id, assembly)
        rows.append(d)
    pd.DataFrame(rows).to_csv(outdir / "sample3v3_candidate_edges.csv", index=False)


def make_plots(trials: pd.DataFrame, holdout: pd.DataFrame, outdir: Path):
    if plt is None or trials.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for agent, g in trials.groupby("agent", sort=False):
        q = g.groupby("trial", as_index=False)["feedback_assembly_min_strength"].mean()
        ax.plot(q.trial, q.feedback_assembly_min_strength, label=agent)
    ax.set_xlabel("training trial"); ax.set_ylabel("feedback assembly minimum strength")
    ax.set_title("Sample 3 V3: emergence of feedback execution topology")
    ax.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(outdir / "sample3v3_feedback_topology_emergence.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    q = holdout.groupby(["agent", "N"], as_index=False)["executed_cost"].mean()
    for agent in ("full_competition", "no_strategy_competition"):
        z = q[q.agent == agent]
        if not z.empty:
            ax.plot(z.N, z.executed_cost, marker="o", label=agent)
    ax.set_xscale("log", base=2); ax.set_xlabel("holdout integer N"); ax.set_ylabel("executed primitive cost")
    ax.set_title("Competition prevents redundant simultaneous procedures")
    ax.legend(); fig.tight_layout(); fig.savefig(outdir / "sample3v3_holdout_cost.png", dpi=220); plt.close(fig)


def resource_report(core, engine, base_cfg, base_evo, task, assembly):
    print(f"Runner: {VERSION}; core={getattr(core,'VERSION','?')}; engine={getattr(engine,'VERSION','?')}")
    print(f"Base: {base_cfg.total_qubits} qubits, {base_cfg.layers} layers")
    print(f"Training trials: {len(make_schedule(task, task.seed))}; holdouts={list(task.holdout_numbers)}")
    print("Direct N/log2(N)/hand-labelled complexity input: NONE")
    print("Explicit experience replay: NONE")
    print("Current meta signal: bounded actual flat-enumeration effort prefix only")
    print("Pre-written recursive decomposition strategy: NONE")
    print("Reference factorization used by controller/learning: NO; post-hoc scoring only")
    print("Cost rule: divisor test=1, quotient gate=1, buffer transfer=1, re-entry=1 (configurable but equal by default)")
    print("Exact candidate_edges supplied in JSON: 0 expected")
    print(f"Auto-generated candidates: {len(base_evo.candidates)}")
    print("Diagnostic assemblies:")
    print(json.dumps(assembly, indent=2))
    ids = {c.id for c in base_evo.candidates}
    for role, cid in assembly.items():
        print(f"  {role}: present={cid in ids}  {cid}")
    print("candidate types:", pd.Series([c.type for c in base_evo.candidates]).value_counts().to_dict())
    print("initial resources:")
    print(json.dumps(core.circuit_resource_estimate(base_cfg), indent=2))


def write_manifest(core, engine, bio, base_cfg, gains, base_evo, task, assembly, outdir, args):
    payload = {
        "runner_version": VERSION,
        "core_version": getattr(core, "VERSION", "unknown"),
        "engine_version": getattr(engine, "VERSION", "unknown"),
        "scientific_scope": "Hybrid topology-generation hypothesis screen; arithmetic environment is classical; surrogate is not quantum evidence.",
        "claim_not_made": [
            "Shor factorization", "quantum speedup", "literal astrocytic qubits",
            "biological discovery of prime factorization", "non-computable cognition",
        ],
        "v3_changes": [
            "retained graph-executed quotient re-entry from V2",
            "removed explicit experience replay",
            "added transparent equal-unit routing costs",
            "added bounded current enumeration-effort observation instead of N/log2(N) complexity",
            "added emergent effort-sensitive N4+G2->N9 meta-selection route",
            "reference prime factors remain post-hoc scoring only",
        ],
        "primary_test": "Experience generates a quotient-reentry graph; an effort-sensitive meta-route then selects enumeration versus feedback with a held-out strategy crossover.",
        "exact_candidate_edges_explicitly_supplied": False,
        "feedback_assembly": assembly,
        "task": asdict(task),
        "node_roles": node_roles(),
        "biological_input": asdict(bio),
        "layer_gains": asdict(gains),
        "competition": asdict(base_evo.competition),
        "candidate_count": len(base_evo.candidates),
        "mode": args.mode,
        "n_seeds": int(args.n_seeds),
    }
    (outdir / "sample3v3_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="V7.1 Sample 3 V3 effort-driven meta-selection")
    ap.add_argument("--core", default=DEFAULT_CORE)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--evolution-config", default=DEFAULT_EVOLUTION)
    ap.add_argument("--task-config", default=DEFAULT_TASK)
    ap.add_argument("--mode", choices=["resource", "surrogate", "local", "qpu"], default="surrogate")
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--agents", default=None, help="comma-separated agent override")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    if args.n_seeds < 1:
        raise ValueError("--n-seeds must be >=1")

    here = Path(__file__).resolve().parent
    def resolve(p):
        q = Path(p); return q if q.exists() else here / q

    core = load_module("v71_sample3v3_core", resolve(args.core))
    engine = load_module("v71_sample3v3_engine", resolve(args.engine))
    bio, base_cfg, hw, _ = core.project_from_json(resolve(args.project))
    core.validate_biological_input(bio); core.validate_network_config(base_cfg)
    gains = core.scores_to_layer_gains(core.compute_mechanistic_scores(bio), bio.hardware_gain)

    base_evo = engine.load_evolution_config(resolve(args.evolution_config))
    if len(base_evo.candidates) != 0:
        raise ValueError("Publication Sample 3 V3 requires candidate_edges=[]")
    engine.finalize_spatial_candidates(base_cfg, base_evo)
    assembly = assembly_ids()
    engine.validate_evolution_config(base_cfg, base_evo)

    task = load_task(resolve(args.task_config))
    if args.agents:
        task.agents = tuple(x.strip() for x in args.agents.split(",") if x.strip())
    validate_task(task)

    if args.mode == "resource":
        resource_report(core, engine, base_cfg, base_evo, task, assembly)
        return 0

    outdir = Path(args.output_dir) if args.output_dir else here / f"V7_1_sample3v3_{args.mode}_{timestamp()}"
    outdir.mkdir(parents=True, exist_ok=True)
    save_candidate_catalog(base_evo, assembly, outdir)
    write_manifest(core, engine, bio, base_cfg, gains, base_evo, task, assembly, outdir, args)

    all_trials, all_edges, all_emergence, all_holdout, all_tokens, seed_summaries = [], [], [], [], [], []
    for sidx in range(args.n_seeds):
        seed = int(task.seed + args.seed_offset + sidx)
        schedule = make_schedule(task, seed)
        for agent in task.agents:
            print(f"\n=== seed={seed} agent={agent} mode={args.mode} ===")
            tr, ed, em, ho, tk, _ = run_agent(
                args.mode, agent, seed, core, engine, base_cfg, gains, hw,
                base_evo, task, assembly, schedule, outdir
            )
            all_trials.append(tr); all_edges.append(ed); all_emergence.append(em); all_holdout.append(ho); all_tokens.append(tk)
            seed_summaries.append(summarize_seed_agent(tr, ho))

    trials = pd.concat(all_trials, ignore_index=True) if all_trials else pd.DataFrame()
    edges = pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    emergence = pd.concat(all_emergence, ignore_index=True) if all_emergence else pd.DataFrame()
    holdout = pd.concat(all_holdout, ignore_index=True) if all_holdout else pd.DataFrame()
    tokens = pd.concat(all_tokens, ignore_index=True) if all_tokens else pd.DataFrame()
    seed_summary = pd.DataFrame(seed_summaries)
    group_summary = aggregate_group(seed_summary)
    effects = paired_effects(seed_summary)

    trials.to_csv(outdir / "sample3v3_trial_log.csv", index=False)
    edges.to_csv(outdir / "sample3v3_rule_topology_timeseries.csv", index=False)
    emergence.to_csv(outdir / "sample3v3_emergence_events.csv", index=False)
    holdout.to_csv(outdir / "sample3v3_holdout_probe.csv", index=False)
    tokens.to_csv(outdir / "sample3v3_token_flow_trace.csv", index=False)
    seed_summary.to_csv(outdir / "sample3v3_agent_seed_summary.csv", index=False)
    group_summary.to_csv(outdir / "sample3v3_agent_group_summary.csv", index=False)
    effects.to_csv(outdir / "sample3v3_paired_control_effects.csv", index=False)
    make_plots(trials, holdout, outdir)

    print("\n=== SAMPLE 3 V3 GROUP SUMMARY ===")
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(group_summary.to_string(index=False))
    print("\nOutput:", outdir.resolve())
    print("Primary files:")
    for x in (
        "sample3v3_agent_group_summary.csv", "sample3v3_holdout_probe.csv",
        "sample3v3_emergence_events.csv", "sample3v3_rule_topology_timeseries.csv",
        "sample3v3_paired_control_effects.csv", "sample3v3_token_flow_trace.csv",
    ):
        print(" ", x)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
