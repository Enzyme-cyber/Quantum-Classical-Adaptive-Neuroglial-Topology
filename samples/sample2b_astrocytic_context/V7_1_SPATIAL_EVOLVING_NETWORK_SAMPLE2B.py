#!/usr/bin/env python3
"""
V7.1 SPATIAL-COMPETITIVE EVOLVING NEUROGLIAL NETWORK
================================
Generic across-trial structural plasticity runner for V7.0 neuroglial circuits.

Core idea
---------
Fast circuit state is reset for each trial. Slow traces and structural strengths
persist across trials:

    trace_i[k+1] = decay_i * trace_i[k] + gain_i * P_i[k]

V7.1 can still accept explicit candidate edges, but its primary mode is spatial
competition: the user supplies only local astrocytic neighborhoods (which glia
are near which neurons / glia, plus optional anatomical affinities). The runner
automatically enumerates biologically local candidate edges. Their structural
strength evolves from measured population activity and local competition. When a
strength crosses an ON threshold the edge becomes part of the next trial's
circuit; OFF threshold provides hysteresis.

Nothing in this file hard-codes a specific brain topology. N->N, N->G, G<->G,
G->N and tripartite candidate edges are all data-driven.

The bundled SPATIAL_ADDITION_TO_MULTIPLICATION example is only one demonstration:
repeated activation of a simple neural loop accumulates slow glial traces,
recruits new glial coupling / feedback, and finally enables a mixed N x G
tripartite operator that can be probed for multiplicative interaction.

Scientific scope
----------------
This is a hybrid computational hypothesis model. Classical across-trial
plasticity updates rebuild quantum-circuit analogues between trials. It does
not claim that biological astrocytes are literal qubits or quantum gates.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

VERSION = "V7.1-SPATIAL-EVOLVING-NG-1"
DEFAULT_CORE = "V7_1_neuroglial_gate_qpu.py"
EPS = 1e-12


# =============================================================================
# Utilities
# =============================================================================

def clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_core(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"V7.1 core not found: {path}")
    spec = importlib.util.spec_from_file_location("v70_core", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import V7.0 core from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def all_qubit_marginals(dist: dict[int, float], n_qubits: int) -> np.ndarray:
    p = np.zeros(int(n_qubits), dtype=float)
    for idx, prob in dist.items():
        ii = int(idx)
        pp = float(prob)
        for q in range(int(n_qubits)):
            if (ii >> q) & 1:
                p[q] += pp
    return p


def node_key(kind: str, idx: int) -> str:
    return ("N" if kind.lower().startswith("n") else "G") + str(int(idx))


def parse_node(s: str, n_neurons: int, n_glia: int) -> tuple[str, int]:
    x = str(s).strip().upper()
    if len(x) < 2 or x[0] not in {"N", "G"}:
        raise ValueError(f"Node driver must look like N3 or G2, got {s!r}")
    idx = int(x[1:])
    if x[0] == "N" and not (0 <= idx < n_neurons):
        raise ValueError(f"Neuron driver out of range: {s}")
    if x[0] == "G" and not (0 <= idx < n_glia):
        raise ValueError(f"Glia driver out of range: {s}")
    return x[0], idx


def event_probability_to_amplitude(probability: float, neuronal_input_gain: float) -> float:
    """Clean |0> RY input conversion. Exact only when initialize_neural_h=False."""
    p = clip(float(probability), 0.0, 1.0)
    theta = 2.0 * math.asin(math.sqrt(p))
    if abs(neuronal_input_gain) < EPS:
        return 0.0
    return clip(theta / float(neuronal_input_gain), -4.0, 4.0)


# =============================================================================
# Plasticity schema
# =============================================================================

@dataclass
class TraceSettings:
    neuron_decay: float = 0.82
    neuron_gain: float = 0.22
    glia_decay: float = 0.88
    glia_gain: float = 0.28
    trace_max: float = 1.0


@dataclass
class CandidateEdge:
    id: str
    type: str
    drivers: list[str]
    driver_mode: str = "min"
    growth_threshold: float = 0.45
    learning_rate: float = 0.18
    structural_decay: float = 0.025
    on_threshold: float = 0.55
    off_threshold: float = 0.30
    initial_strength: float = 0.0
    active_initial: bool = False
    activation_requires: list[str] = field(default_factory=list)
    max_weight: float = 1.0
    max_exchange_weight: float = 1.0
    require_phasic_sensory: bool = True
    # V7.1 spatial-generation metadata. These values constrain competition but
    # do not force an edge to form.
    affinity: float = 1.0
    competition_groups: list[str] = field(default_factory=list)
    requires_glial_network: bool = False
    mediating_glia: int | None = None
    generated_from: str = "explicit"
    # topology payload (only fields relevant to `type` are used)
    source: int | None = None
    target: int | None = None
    a: int | None = None
    b: int | None = None
    sensory_neuron: int | None = None
    context_glia: int | None = None
    target_neuron: int | None = None


@dataclass
class CandidateState:
    strength: float
    active: bool


@dataclass
class CompetitionSettings:
    enabled: bool = True
    default_winners_per_group: int = 1
    winners_by_prefix: dict[str, int] = field(default_factory=dict)
    loser_decay_multiplier: float = 1.5


@dataclass
class EvolutionConfig:
    trace: TraceSettings
    candidates: list[CandidateEdge]
    training_phases: list[dict[str, Any]]
    probes: list[dict[str, Any]]
    initial_traces: dict[str, float] = field(default_factory=dict)
    shots_override: int | None = None
    qpu_repeats_override: int | None = None
    run_probes_before_training: bool = False
    spatial_growth: dict[str, Any] = field(default_factory=dict)
    competition: CompetitionSettings = field(default_factory=CompetitionSettings)


def candidate_from_dict(d: dict[str, Any]) -> CandidateEdge:
    return CandidateEdge(
        id=str(d["id"]),
        type=str(d["type"]).lower(),
        drivers=[str(x) for x in d.get("drivers", [])],
        driver_mode=str(d.get("driver_mode", "min")).lower(),
        growth_threshold=float(d.get("growth_threshold", 0.45)),
        learning_rate=float(d.get("learning_rate", 0.18)),
        structural_decay=float(d.get("structural_decay", 0.025)),
        on_threshold=float(d.get("on_threshold", 0.55)),
        off_threshold=float(d.get("off_threshold", 0.30)),
        initial_strength=float(d.get("initial_strength", 0.0)),
        active_initial=bool(d.get("active_initial", False)),
        activation_requires=[str(x) for x in d.get("activation_requires", [])],
        max_weight=float(d.get("max_weight", 1.0)),
        max_exchange_weight=float(d.get("max_exchange_weight", d.get("max_weight", 1.0))),
        require_phasic_sensory=bool(d.get("require_phasic_sensory", True)),
        affinity=float(d.get("affinity", 1.0)),
        competition_groups=[str(x) for x in d.get("competition_groups", [])],
        requires_glial_network=bool(d.get("requires_glial_network", False)),
        mediating_glia=None if d.get("mediating_glia") is None else int(d["mediating_glia"]),
        generated_from=str(d.get("generated_from", "explicit")),
        source=None if d.get("source") is None else int(d["source"]),
        target=None if d.get("target") is None else int(d["target"]),
        a=None if d.get("a") is None else int(d["a"]),
        b=None if d.get("b") is None else int(d["b"]),
        sensory_neuron=None if d.get("sensory_neuron") is None else int(d["sensory_neuron"]),
        context_glia=None if d.get("context_glia") is None else int(d["context_glia"]),
        target_neuron=None if d.get("target_neuron") is None else int(d["target_neuron"]),
    )


def load_evolution_config(path: Path) -> EvolutionConfig:
    d = json.loads(path.read_text(encoding="utf-8"))
    tr = d.get("trace", {})
    trace = TraceSettings(
        neuron_decay=float(tr.get("neuron_decay", 0.82)),
        neuron_gain=float(tr.get("neuron_gain", 0.22)),
        glia_decay=float(tr.get("glia_decay", 0.88)),
        glia_gain=float(tr.get("glia_gain", 0.28)),
        trace_max=float(tr.get("trace_max", 1.0)),
    )
    comp=d.get("competition", {})
    competition=CompetitionSettings(
        enabled=bool(comp.get("enabled", True)),
        default_winners_per_group=max(1,int(comp.get("default_winners_per_group",1))),
        winners_by_prefix={str(k):max(1,int(v)) for k,v in comp.get("winners_by_prefix",{}).items()},
        loser_decay_multiplier=max(1.0,float(comp.get("loser_decay_multiplier",1.5))),
    )
    return EvolutionConfig(
        trace=trace,
        candidates=[candidate_from_dict(x) for x in d.get("candidate_edges", [])],
        training_phases=list(d.get("training_phases", [])),
        probes=list(d.get("probes", [])),
        initial_traces={str(k): float(v) for k, v in d.get("initial_traces", {}).items()},
        shots_override=None if d.get("shots_override") is None else int(d["shots_override"]),
        qpu_repeats_override=None if d.get("qpu_repeats_override") is None else int(d["qpu_repeats_override"]),
        run_probes_before_training=bool(d.get("run_probes_before_training", False)),
        spatial_growth=dict(d.get("spatial_growth", {})),
        competition=competition,
    )



def _relation_affinity(rec: dict[str, Any], key: str) -> float:
    return clip(float(rec.get(key, rec.get("affinity", 1.0))), 0.0, 1.0)


def _template(spatial: dict[str, Any], typ: str) -> dict[str, Any]:
    d=dict(spatial.get("edge_templates", {}).get(typ, {}))
    defaults={
        "growth_threshold":0.18,"learning_rate":0.28,"structural_decay":0.02,
        "on_threshold":0.38,"off_threshold":0.18,"max_weight":1.0,
        "max_exchange_weight":1.0,"driver_mode":"min",
    }
    for k,v in defaults.items(): d.setdefault(k,v)
    return d


def _mk_candidate(typ: str, cid: str, payload: dict[str, Any], drivers: list[str], affinity: float,
                  groups: list[str], template: dict[str, Any], generated_from: str, **extra) -> CandidateEdge:
    d={
        "id":cid,"type":typ,"drivers":drivers,"driver_mode":template.get("driver_mode","min"),
        "growth_threshold":template.get("growth_threshold",0.18),
        "learning_rate":template.get("learning_rate",0.28),
        "structural_decay":template.get("structural_decay",0.02),
        "on_threshold":template.get("on_threshold",0.38),
        "off_threshold":template.get("off_threshold",0.18),
        "initial_strength":0.0,"active_initial":False,
        "activation_requires":[],"max_weight":template.get("max_weight",1.0),
        "max_exchange_weight":template.get("max_exchange_weight",template.get("max_weight",1.0)),
        "require_phasic_sensory":template.get("require_phasic_sensory",True),
        "affinity":affinity,"competition_groups":groups,
        "requires_glial_network":bool(template.get("requires_glial_network",False)),
        "generated_from":generated_from,
        **payload,**extra,
    }
    return candidate_from_dict(d)


def generate_spatial_candidates(cfg, evo: EvolutionConfig) -> list[CandidateEdge]:
    spatial=evo.spatial_growth or {}
    if not bool(spatial.get("enabled", False)):
        return list(evo.candidates)
    nN,nG=cfg.n_neurons,cfg.n_glia
    domains={}
    for d in spatial.get("local_domains", []):
        g=int(d["glia"])
        if not (0<=g<nG): raise ValueError(f"spatial domain glia out of range: G{g}")
        neurons={}
        for r in d.get("nearby_neurons",[]):
            n=int(r["neuron"])
            if not (0<=n<nN): raise ValueError(f"nearby neuron out of range: N{n}")
            neurons[n]=dict(r)
        glia={}
        for r in d.get("nearby_glia",[]):
            h=int(r["glia"])
            if not (0<=h<nG) or h==g: continue
            glia[h]=dict(r)
        domains[g]={"neurons":neurons,"glia":glia}
    enabled=set(str(x) for x in spatial.get("generate_edge_types",["glia_glia","glia_to_neuron","neuron_to_neuron","tripartite"]))
    min_aff=float(spatial.get("minimum_affinity",0.0))
    out=list(evo.candidates); seen={c.id for c in out}
    existing_nn={(int(e.source),int(e.target)) for e in cfg.neuron_to_neuron or []}
    existing_ng={(int(e.source),int(e.target)) for e in cfg.neuron_to_glia or []}
    existing_gn={(int(e.source),int(e.target)) for e in cfg.glia_to_neuron or []}
    existing_gg={tuple(sorted((int(e.a),int(e.b)))) for e in cfg.glia_glia or []}
    existing_tri={(int(t.sensory_neuron),int(t.context_glia),int(t.target_neuron)) for t in cfg.tripartite_synapses or []}

    if "glia_glia" in enabled:
        tpl=_template(spatial,"glia_glia")
        for g,d in domains.items():
            for h,r in d["glia"].items():
                a,b=sorted((g,h)); cid=f"AUTO_GG_G{a}_G{b}"
                if cid in seen or (a,b) in existing_gg: continue
                # reciprocal listing, if present, is averaged rather than duplicated
                aff=_relation_affinity(r,"gg_affinity")
                if h in domains and g in domains[h]["glia"]:
                    aff=0.5*(aff+_relation_affinity(domains[h]["glia"][g],"gg_affinity"))
                if aff<min_aff: continue
                out.append(_mk_candidate("glia_glia",cid,{"a":a,"b":b},[f"G{a}",f"G{b}"],aff,
                    [f"GG:G{a}",f"GG:G{b}"],tpl,f"local_domains:G{g}"))
                seen.add(cid)

    if "neuron_to_glia" in enabled:
        tpl=_template(spatial,"neuron_to_glia")
        for g,d in domains.items():
            for n,r in d["neurons"].items():
                if (n,g) in existing_ng: continue
                aff=_relation_affinity(r,"ng_affinity"); cid=f"AUTO_NG_N{n}_G{g}"
                if cid in seen or aff<min_aff: continue
                out.append(_mk_candidate("neuron_to_glia",cid,{"source":n,"target":g},[f"N{n}"],aff,
                    [f"NG:G{g}"],tpl,f"local_domain:G{g}",mediating_glia=g)); seen.add(cid)

    if "glia_to_neuron" in enabled:
        tpl=_template(spatial,"glia_to_neuron")
        for g,d in domains.items():
            for n,r in d["neurons"].items():
                if (g,n) in existing_gn: continue
                aff=_relation_affinity(r,"gn_affinity"); cid=f"AUTO_GN_G{g}_N{n}"
                if cid in seen or aff<min_aff: continue
                out.append(_mk_candidate("glia_to_neuron",cid,{"source":g,"target":n},[f"G{g}"],aff,
                    [f"GN_IN:N{n}",f"GN_OUT:G{g}"],tpl,f"local_domain:G{g}",mediating_glia=g)); seen.add(cid)

    # Neurons sharing a glial microdomain are locally eligible for N->N growth.
    if "neuron_to_neuron" in enabled:
        tpl=_template(spatial,"neuron_to_neuron")
        for g,d in domains.items():
            ns=sorted(d["neurons"])
            for src in ns:
                for dst in ns:
                    if src==dst or (src,dst) in existing_nn: continue
                    rs,rd=d["neurons"][src],d["neurons"][dst]
                    aff=_relation_affinity(rs,"nn_affinity")*_relation_affinity(rd,"nn_affinity")
                    cid=f"AUTO_NN_G{g}_N{src}_N{dst}"
                    if cid in seen or aff<min_aff: continue
                    out.append(_mk_candidate("neuron_to_neuron",cid,{"source":src,"target":dst},
                        [f"N{src}",f"G{g}"],aff,[f"NN_IN:N{dst}",f"NN_OUT:N{src}"],tpl,
                        f"shared_glial_domain:G{g}",mediating_glia=g)); seen.add(cid)

    if "tripartite" in enabled:
        tpl=_template(spatial,"tripartite")
        sensory_pool=set(int(x) for x in tpl.get("sensory_pool",range(nN)))
        target_pool=set(int(x) for x in tpl.get("target_pool",range(nN)))
        for g,d in domains.items():
            for sn,rs in d["neurons"].items():
                if sn not in sensory_pool: continue
                for tn,rt in d["neurons"].items():
                    if tn not in target_pool or sn==tn: continue
                    if (sn,g,tn) in existing_tri: continue
                    aff=_relation_affinity(rs,"tripartite_sensory_affinity")*_relation_affinity(rt,"tripartite_target_affinity")
                    cid=f"AUTO_TRI_N{sn}_G{g}_N{tn}"
                    if cid in seen or aff<min_aff: continue
                    out.append(_mk_candidate("tripartite",cid,
                        {"sensory_neuron":sn,"context_glia":g,"target_neuron":tn},[f"N{sn}",f"G{g}"],aff,
                        [f"TRI_T:N{tn}",f"TRI_G:G{g}"],tpl,f"local_domain:G{g}",mediating_glia=g)); seen.add(cid)
    return out


def finalize_spatial_candidates(cfg, evo: EvolutionConfig):
    evo.candidates=generate_spatial_candidates(cfg,evo)
    return evo


def validate_evolution_config(cfg, evo: EvolutionConfig):
    nN, nG = cfg.n_neurons, cfg.n_glia
    t = evo.trace
    for name in ("neuron_decay", "glia_decay"):
        v = float(getattr(t, name))
        if not (0 <= v <= 1):
            raise ValueError(f"{name} must be 0..1")
    for name in ("neuron_gain", "glia_gain", "trace_max"):
        if float(getattr(t, name)) < 0:
            raise ValueError(f"{name} must be >=0")
    ids = [c.id for c in evo.candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate edge IDs must be unique.")
    allowed = {"neuron_to_neuron", "neuron_to_glia", "glia_glia", "glia_to_neuron", "tripartite"}
    idset = set(ids)
    for c in evo.candidates:
        if c.type not in allowed:
            raise ValueError(f"Unknown candidate edge type: {c.type}")
        if c.driver_mode not in {"min", "mean", "max", "product"}:
            raise ValueError(f"Unknown driver_mode for {c.id}: {c.driver_mode}")
        for d in c.drivers:
            parse_node(d, nN, nG)
        for dep in c.activation_requires:
            if dep not in idset:
                raise ValueError(f"{c.id} requires unknown candidate {dep}")
        for name in ("growth_threshold", "on_threshold", "off_threshold", "initial_strength"):
            v = float(getattr(c, name))
            if not (0 <= v <= 1):
                raise ValueError(f"{c.id}.{name} must be 0..1")
        if c.off_threshold > c.on_threshold:
            raise ValueError(f"{c.id}: off_threshold should be <= on_threshold")
        if c.learning_rate < 0 or c.structural_decay < 0:
            raise ValueError(f"{c.id}: learning/decay must be >=0")
        if not (0.0 <= float(c.affinity) <= 1.0):
            raise ValueError(f"{c.id}: affinity must be 0..1")
        if c.mediating_glia is not None and not (0 <= int(c.mediating_glia) < nG):
            raise ValueError(f"{c.id}: mediating_glia out of range")
        if abs(c.max_weight) > 4 or c.max_exchange_weight < 0 or c.max_exchange_weight > 4:
            raise ValueError(f"{c.id}: edge weights must be within core limits")
        if c.type in {"neuron_to_neuron", "neuron_to_glia", "glia_to_neuron"}:
            if c.source is None or c.target is None:
                raise ValueError(f"{c.id}: directed edge needs source,target")
        elif c.type == "glia_glia":
            if c.a is None or c.b is None:
                raise ValueError(f"{c.id}: glia_glia needs a,b")
        elif c.type == "tripartite":
            if None in (c.sensory_neuron, c.context_glia, c.target_neuron):
                raise ValueError(f"{c.id}: tripartite needs sensory_neuron, context_glia,target_neuron")


# =============================================================================
# Dynamic topology
# =============================================================================

def driver_value(c: CandidateEdge, traces: dict[str, float]) -> float:
    if not c.drivers:
        return 0.0
    vals = [clip(float(traces.get(k.upper(), 0.0)), 0.0, 1.0) for k in c.drivers]
    if c.driver_mode == "min": raw=float(min(vals))
    elif c.driver_mode == "mean": raw=float(np.mean(vals))
    elif c.driver_mode == "max": raw=float(max(vals))
    else:
        raw=1.0
        for v in vals: raw*=v
    return clip(raw*float(c.affinity),0.0,1.0)


def active_glial_degree(candidates, states, gidx: int) -> int:
    deg=0
    for c in candidates:
        if c.type!="glia_glia" or not states[c.id].active: continue
        if int(c.a)==int(gidx) or int(c.b)==int(gidx): deg+=1
    return deg


def competition_budget(settings: CompetitionSettings, group: str) -> int:
    prefix=str(group).split(":",1)[0]
    return int(settings.winners_by_prefix.get(prefix, settings.default_winners_per_group))


def competition_winners(candidates, states, traces, settings: CompetitionSettings) -> dict[str,set[str]]:
    groups={}
    if not settings.enabled: return groups
    for c in candidates:
        score=driver_value(c,traces)
        for g in c.competition_groups:
            groups.setdefault(g,[]).append((score,float(c.affinity),float(states[c.id].strength),c.id))
    out={}
    for g,vals in groups.items():
        vals=sorted(vals,key=lambda x:(x[0],x[1],x[2],x[3]),reverse=True)
        out[g]=set(v[3] for v in vals[:competition_budget(settings,g)])
    return out


def update_candidate_states(
    candidates: list[CandidateEdge],
    states: dict[str, CandidateState],
    traces: dict[str, float],
    competition: CompetitionSettings,
) -> list[dict[str, Any]]:
    """Simultaneous V7.1 growth update with local winner competition."""
    events=[]
    winners=competition_winners(candidates,states,traces,competition)
    proposed={}
    for c in candidates:
        st=states[c.id]
        dep_ok=all(states[d].active for d in c.activation_requires)
        if c.requires_glial_network:
            g=c.mediating_glia
            dep_ok=dep_ok and (g is not None) and active_glial_degree(candidates,states,int(g))>0
        wins=all((not competition.enabled) or c.id in winners.get(grp,set()) for grp in c.competition_groups)
        drv=driver_value(c,traces) if dep_ok else 0.0
        thr=clip(c.growth_threshold,0.0,1.0)
        if dep_ok and wins and drv>thr:
            growth=c.learning_rate*(drv-thr)/max(EPS,1.0-thr); decay=0.0
        else:
            growth=0.0
            base=c.structural_decay*(thr-drv)/max(EPS,thr if thr>0 else 1.0) if drv<thr else c.structural_decay
            decay=base*(competition.loser_decay_multiplier if dep_ok and (not wins) else 1.0)
        proposed[c.id]=(clip(st.strength+growth-decay,0.0,1.0),drv,wins,dep_ok)
    # Apply simultaneously, then threshold with hysteresis.
    for c in candidates:
        st=states[c.id]; old_strength=st.strength; old_active=st.active
        new_strength,drv,wins,dep_ok=proposed[c.id]
        st.strength=new_strength
        if (not st.active) and st.strength>=c.on_threshold and dep_ok and wins:
            st.active=True
        elif st.active and (st.strength<c.off_threshold or (competition.enabled and not wins)):
            st.active=False
        if st.active!=old_active:
            events.append({"candidate_id":c.id,"edge_type":c.type,"event":"formed" if st.active else "removed",
                           "driver":drv,"affinity":c.affinity,"competition_win":wins,
                           "strength_before":old_strength,"strength_after":st.strength,
                           "generated_from":c.generated_from})
    return events


def merge_directed(core, edges, src, dst, weight):
    out = list(edges or [])
    for i, e in enumerate(out):
        if int(e.source) == int(src) and int(e.target) == int(dst):
            out[i] = core.DirectedEdge(int(src), int(dst), float(e.weight) + float(weight))
            return out
    out.append(core.DirectedEdge(int(src), int(dst), float(weight)))
    return out


def merge_glia_glia(core, edges, a, b, weight, exchange_weight):
    aa, bb = sorted((int(a), int(b)))
    out = list(edges or [])
    for i, e in enumerate(out):
        x, y = sorted((int(e.a), int(e.b)))
        if (x, y) == (aa, bb):
            ex0 = core.effective_glia_exchange_weight(e)
            out[i] = core.UndirectedEdge(
                int(e.a), int(e.b), float(e.weight) + float(weight),
                max(0.0, float(ex0) + float(exchange_weight)),
            )
            return out
    out.append(core.UndirectedEdge(aa, bb, float(weight), max(0.0, float(exchange_weight))))
    return out


def merge_tripartite(core, edges, c: CandidateEdge, weight: float):
    out = list(edges or [])
    for i, e in enumerate(out):
        if (int(e.sensory_neuron), int(e.context_glia), int(e.target_neuron)) == (
            int(c.sensory_neuron), int(c.context_glia), int(c.target_neuron)
        ):
            out[i] = core.TripartiteSynapse(
                int(c.sensory_neuron), int(c.context_glia), int(c.target_neuron),
                float(e.weight) + float(weight), bool(c.require_phasic_sensory)
            )
            return out
    out.append(core.TripartiteSynapse(
        int(c.sensory_neuron), int(c.context_glia), int(c.target_neuron),
        float(weight), bool(c.require_phasic_sensory)
    ))
    return out


def materialize_dynamic_config(core, base_cfg, candidates, states, neural_inputs=None):
    nn = list(base_cfg.neuron_to_neuron or [])
    ng = list(base_cfg.neuron_to_glia or [])
    gg = list(base_cfg.glia_glia or [])
    gn = list(base_cfg.glia_to_neuron or [])
    tri = list(base_cfg.tripartite_synapses or [])

    for c in candidates:
        st = states[c.id]
        if not st.active:
            continue
        frac = clip(st.strength, 0.0, 1.0)
        w = clip(c.max_weight * frac, -4.0, 4.0)
        if c.type == "neuron_to_neuron":
            nn = merge_directed(core, nn, c.source, c.target, max(0.0, w))
        elif c.type == "neuron_to_glia":
            ng = merge_directed(core, ng, c.source, c.target, w)
        elif c.type == "glia_to_neuron":
            gn = merge_directed(core, gn, c.source, c.target, w)
        elif c.type == "glia_glia":
            ex = clip(c.max_exchange_weight * frac, 0.0, 4.0)
            gg = merge_glia_glia(core, gg, c.a, c.b, w, ex)
        elif c.type == "tripartite":
            tri = merge_tripartite(core, tri, c, w)

    cfg = replace(
        base_cfg,
        neural_inputs=list(neural_inputs if neural_inputs is not None else base_cfg.neural_inputs or []),
        neuron_to_neuron=nn,
        neuron_to_glia=ng,
        glia_glia=gg,
        glia_to_neuron=gn,
        tripartite_synapses=tri,
    )
    core.validate_network_config(cfg)
    return cfg


# =============================================================================
# Trial input and trace updates
# =============================================================================

def build_events(core, spec: list[dict[str, Any]], gains) -> list[Any]:
    out = []
    for x in spec:
        if "probability" in x:
            amp = event_probability_to_amplitude(float(x["probability"]), gains.neuronal_input_gain)
        else:
            amp = float(x.get("amplitude", 0.0))
        out.append(core.NeuralInputEvent(
            int(x.get("layer", 0)), int(x["neuron"]), amp,
            float(x.get("phase_rad", 0.0)), str(x.get("event_kind", "evidence"))
        ))
    return out


def update_traces(traces, marg, cfg, settings: TraceSettings):
    for i in range(cfg.n_neurons):
        k = f"N{i}"
        traces[k] = clip(
            settings.neuron_decay * traces.get(k, 0.0) + settings.neuron_gain * float(marg[i]),
            0.0, settings.trace_max,
        )
    for g in range(cfg.n_glia):
        k = f"G{g}"
        traces[k] = clip(
            settings.glia_decay * traces.get(k, 0.0) + settings.glia_gain * float(marg[cfg.n_neurons + g]),
            0.0, settings.trace_max,
        )


# =============================================================================
# Backends
# =============================================================================

def run_local_trial(core, cfg, gains, shots, max_qubits, outdir, label, readout_mode="population") -> tuple[dict[int,float], np.ndarray]:
    if cfg.total_qubits > max_qubits:
        raise RuntimeError(
            f"Local CPUQVM blocked at {cfg.total_qubits} qubits; local_safe_qubits={max_qubits}."
        )
    qp = core.import_qpanda_core()
    qvm = qp["CPUQVM"]()
    prog = core.build_layered_program(cfg, gains, qp=qp, measure_glia=True, neural_readout_mode=readout_mode)
    qvm.run(prog, shots=int(shots))
    counts = qvm.result().get_counts()
    dist = core.normalized_sparse_distribution(counts, cfg.total_qubits)
    marg = all_qubit_marginals(dist, cfg.total_qubits)
    return dist, marg


def run_qpu_trial(core, cfg, gains, hw, outdir, label, readout_mode="population") -> tuple[dict[int,float], np.ndarray]:
    trial_dir = outdir / "qpu_trials" / label
    trial_dir.mkdir(parents=True, exist_ok=True)
    mean, _, _ = core.run_qpu(
        {label: cfg}, gains, hw, trial_dir, print, measure_glia=True, neural_readout_mode=readout_mode
    )
    dist = mean[label]
    return dist, all_qubit_marginals(dist, cfg.total_qubits)


def surrogate_trial(core, cfg, gains) -> tuple[dict[int,float], np.ndarray]:
    """Fast deterministic structural smoke test, NOT a quantum simulator."""
    nN, nG = cfg.n_neurons, cfg.n_glia
    pN = np.zeros(nN, dtype=float)
    pG = np.zeros(nG, dtype=float)
    if getattr(cfg, "initialize_neural_h", True):
        pN[:] = 0.5
    glial_init = getattr(cfg, "glial_initial_probabilities", None)
    if glial_init is not None:
        if len(glial_init) != nG:
            raise ValueError(
                f"glial_initial_probabilities must have length {nG}; got {len(glial_init)}"
            )
        pG[:] = np.asarray([clip(float(x), 0.0, 1.0) for x in glial_init], dtype=float)
    elif getattr(cfg, "initialize_glial_baseline", True):
        for g in range(nG):
            th = gains.glial_local_bias * (1 + 0.05*math.sin(2*math.pi*g/max(1,nG)))
            pG[g] = math.sin(th/2)**2
    by_layer = core.input_events_by_layer(cfg)
    evidence = core.evidence_neurons_by_layer(cfg)
    for layer in range(cfg.layers):
        for ev in by_layer.get(layer, []):
            th = clip(ev.amplitude * gains.neuronal_input_gain, -math.pi, math.pi)
            pe = math.sin(th/2)**2
            # independent RY composition surrogate
            pN[ev.neuron] = 1.0 - (1.0-pN[ev.neuron])*(1.0-pe)
        for e in cfg.neuron_to_neuron or []:
            th = clip(e.weight*gains.neuron_to_neuron_gain, 0.0, 1.5)
            pe = pN[e.source] * math.sin(th/2)**2
            pN[e.target] = 1-(1-pN[e.target])*(1-pe)
        for e in cfg.neuron_to_glia or []:
            th = clip(e.weight*gains.neuron_to_glia_gain, -1.5, 1.5)
            pe = pN[e.source] * math.sin(th/2)**2
            pG[e.target] = 1-(1-pG[e.target])*(1-pe)
        if cfg.enable_glia_exchange:
            for e in cfg.glia_glia or []:
                th = abs(core.effective_glia_exchange_weight(e) * gains.glia_glia_exchange_gain)
                mix = min(0.5, math.sin(th)**2 * 0.5)
                a, b = int(e.a), int(e.b)
                pa, pb = pG[a], pG[b]
                pG[a] = (1-mix)*pa + mix*pb
                pG[b] = (1-mix)*pb + mix*pa
        if cfg.enable_tripartite_synapses:
            active = evidence.get(layer, set())
            for t in cfg.tripartite_synapses or []:
                if t.require_phasic_sensory and int(t.sensory_neuron) not in active:
                    continue
                th = clip(math.pi*gains.threshold_gate_gain*t.weight, -math.pi, math.pi)
                pe = pN[t.sensory_neuron]*pG[t.context_glia]*math.sin(th/2)**2
                pN[t.target_neuron] = 1-(1-pN[t.target_neuron])*(1-pe)
        for e in cfg.glia_to_neuron or []:
            th = clip(e.weight*gains.glia_to_neuron_gain, -1.5, 1.5)
            pe = pG[e.source] * math.sin(th/2)**2
            pN[e.target] = 1-(1-pN[e.target])*(1-pe)
    marg = np.concatenate([pN,pG])
    return {}, marg


# =============================================================================
# Evolution engine
# =============================================================================

def run_trial_backend(mode, core, cfg, gains, shots, hw, outdir, label, readout_mode="population"):
    if mode == "surrogate":
        return surrogate_trial(core, cfg, gains)
    if mode == "local":
        return run_local_trial(core, cfg, gains, shots, hw.local_safe_qubits, outdir, label, readout_mode)
    if mode == "qpu":
        return run_qpu_trial(core, cfg, gains, hw, outdir, label, readout_mode)
    raise ValueError(mode)


def initialize_traces(cfg, evo: EvolutionConfig):
    traces = {f"N{i}": 0.0 for i in range(cfg.n_neurons)}
    traces.update({f"G{i}": 0.0 for i in range(cfg.n_glia)})
    for k, v in evo.initial_traces.items():
        parse_node(k, cfg.n_neurons, cfg.n_glia)
        traces[k.upper()] = clip(float(v), 0.0, evo.trace.trace_max)
    return traces


def initialize_states(evo: EvolutionConfig):
    return {
        c.id: CandidateState(clip(c.initial_strength,0,1), bool(c.active_initial))
        for c in evo.candidates
    }


def trial_rows_from_marg(trial_no, phase, rep, cfg, marg, traces, candidates, states):
    rows = []
    for i in range(cfg.n_neurons):
        rows.append({"trial":trial_no,"phase":phase,"phase_repeat":rep,"node":f"N{i}",
                     "P_active":float(marg[i]),"slow_trace":float(traces[f"N{i}"])})
    for g in range(cfg.n_glia):
        rows.append({"trial":trial_no,"phase":phase,"phase_repeat":rep,"node":f"G{g}",
                     "P_active":float(marg[cfg.n_neurons+g]),"slow_trace":float(traces[f"G{g}"])})
    return rows


def edge_rows(trial_no, phase, candidates, states, traces):
    rows=[]
    for c in candidates:
        st=states[c.id]
        rows.append({
            "trial":trial_no,"phase":phase,"candidate_id":c.id,"edge_type":c.type,
            "driver":driver_value(c,traces),"strength":st.strength,"active":st.active,
            "effective_weight":c.max_weight*st.strength if st.active else 0.0,
            "effective_exchange_weight":c.max_exchange_weight*st.strength if st.active and c.type=="glia_glia" else 0.0,
            "drivers":";".join(c.drivers),"requires":";".join(c.activation_requires),
            "affinity":c.affinity,"competition_groups":";".join(c.competition_groups),
            "requires_glial_network":c.requires_glial_network,"mediating_glia":c.mediating_glia,
            "generated_from":c.generated_from,
        })
    return rows


def run_training(mode, core, base_cfg, gains, hw, evo, outdir, max_training_trials=None, freeze_plasticity=False):
    traces = initialize_traces(base_cfg, evo)
    states = initialize_states(evo)
    trial_log=[]; node_log=[]; edge_log=[]; emergence=[]
    trial_no=0
    shots = int(evo.shots_override or hw.shots or 2000)
    if mode == "local" and evo.shots_override is None:
        shots = max(1000, min(20000, int(getattr(hw,"shots",1000))*5))

    for phase in evo.training_phases:
        name=str(phase.get("name","phase"))
        reps=int(phase.get("repetitions",1))
        plasticity=bool(phase.get("plasticity_update",True)) and (not freeze_plasticity)
        for r in range(reps):
            if max_training_trials is not None and trial_no >= int(max_training_trials):
                return traces,states,pd.DataFrame(trial_log),pd.DataFrame(node_log),pd.DataFrame(edge_log),pd.DataFrame(emergence)
            trial_no += 1
            events=build_events(core,list(phase.get("neural_inputs",[])),gains)
            cfg=materialize_dynamic_config(core,base_cfg,evo.candidates,states,events)
            label=f"trial_{trial_no:03d}_{name}_{r+1:03d}"
            _,marg=run_trial_backend(mode,core,cfg,gains,shots,hw,outdir,label,readout_mode="population")
            active_before=sum(1 for x in states.values() if x.active)
            if plasticity:
                update_traces(traces,marg,cfg,evo.trace)
                evs=update_candidate_states(evo.candidates,states,traces,evo.competition)
                for e in evs:
                    emergence.append({"trial":trial_no,"phase":name,**e})
            active_after=sum(1 for x in states.values() if x.active)
            trial_log.append({
                "trial":trial_no,"phase":name,"phase_repeat":r+1,"plasticity_update":plasticity,
                "active_candidate_edges_before":active_before,"active_candidate_edges_after":active_after,
            })
            node_log.extend(trial_rows_from_marg(trial_no,name,r+1,cfg,marg,traces,evo.candidates,states))
            edge_log.extend(edge_rows(trial_no,name,evo.candidates,states,traces))
            print(f"[V7.1 {mode}] trial {trial_no:03d} {name} {r+1}/{reps}: active_edges={active_after}")
    return traces,states,pd.DataFrame(trial_log),pd.DataFrame(node_log),pd.DataFrame(edge_log),pd.DataFrame(emergence)


# =============================================================================
# Probes
# =============================================================================

def fit_product_models(df: pd.DataFrame):
    y=df["P_output"].to_numpy(float)
    x=df["P_x"].to_numpy(float); g=df["P_context_glia"].to_numpy(float)
    X0=np.column_stack([np.ones(len(df)),x,g])
    X1=np.column_stack([np.ones(len(df)),x,g,x*g])
    b0=np.linalg.lstsq(X0,y,rcond=None)[0]
    b1=np.linalg.lstsq(X1,y,rcond=None)[0]
    p0=X0@b0; p1=X1@b1
    def met(p):
        rm=float(np.sqrt(np.mean((y-p)**2)))
        mae=float(np.mean(np.abs(y-p)))
        ssr=float(np.sum((y-p)**2)); sst=float(np.sum((y-y.mean())**2))
        r2=1-ssr/sst if sst>EPS else float("nan")
        return rm,mae,r2
    rm0,ma0,r20=met(p0); rm1,ma1,r21=met(p1)
    return pd.DataFrame([
        {"model":"additive","intercept":b0[0],"beta_x":b0[1],"beta_g":b0[2],"beta_xg":0.0,"RMSE":rm0,"MAE":ma0,"R2":r20},
        {"model":"interaction","intercept":b1[0],"beta_x":b1[1],"beta_g":b1[2],"beta_xg":b1[3],"RMSE":rm1,"MAE":ma1,"R2":r21},
    ])


def run_product_probe(mode, core, base_cfg, gains, hw, evo, states, probe, outdir, prefix=""):

    name=str(probe.get("name","product_probe"))
    xspec=probe["x"]; yspec=probe["y"]
    xn=int(xspec["neuron"]); yn=int(yspec["neuron"])
    context_g=int(probe["context_glia"]); outn=int(probe["output_neuron"])
    xs=[float(v) for v in xspec.get("values",[0,0.25,0.5,0.75,1])]
    ys=[float(v) for v in yspec.get("values",[0,0.25,0.5,0.75,1])]
    rows=[]; shots=int(probe.get("shots",evo.shots_override or hw.shots or 3000))
    for xv in xs:
        for yv in ys:
            ev=[
                core.NeuralInputEvent(int(xspec.get("layer",0)),xn,event_probability_to_amplitude(xv,gains.neuronal_input_gain),0.0,"evidence"),
                core.NeuralInputEvent(int(yspec.get("layer",0)),yn,event_probability_to_amplitude(yv,gains.neuronal_input_gain),0.0,"evidence"),
            ]
            cfg=materialize_dynamic_config(core,base_cfg,evo.candidates,states,ev)
            label=f"probe_{name}_x{xv:.3f}_y{yv:.3f}".replace('.','p')
            _,marg=run_trial_backend(mode,core,cfg,gains,shots,hw,outdir,label,readout_mode="population")
            rows.append({
                "probe":name,"x_requested":xv,"y_requested":yv,
                "P_x":float(marg[xn]),"P_y_driver":float(marg[yn]),
                "P_context_glia":float(marg[cfg.n_neurons+context_g]),
                "P_output":float(marg[outn]),
                "Px_times_Pg":float(marg[xn]*marg[cfg.n_neurons+context_g]),
            })
            print(f"[probe] x={xv:.2f} y={yv:.2f} -> Pout={marg[outn]:.4f}")
    df=pd.DataFrame(rows)
    stem=f"{prefix}{name}"
    df.to_csv(outdir/f"{stem}_grid.csv",index=False)
    reg=fit_product_models(df); reg.to_csv(outdir/f"{stem}_regression.csv",index=False)
    # Falsifiable interaction diagnostics. Use the four extreme requested corners.
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
    def corner(xv,yv):
        d=df[(np.isclose(df["x_requested"],xv)) & (np.isclose(df["y_requested"],yv))]
        return float(d.iloc[0]["P_output"]) if len(d) else float("nan")
    f00,f10,f01,f11=corner(xmin,ymin),corner(xmax,ymin),corner(xmin,ymax),corner(xmax,ymax)
    second_difference=f11-f10-f01+f00
    r0=float(reg.loc[reg["model"]=="additive","R2"].iloc[0]) if len(reg) else float("nan")
    r1=float(reg.loc[reg["model"]=="interaction","R2"].iloc[0]) if len(reg) else float("nan")
    rm0=float(reg.loc[reg["model"]=="additive","RMSE"].iloc[0]) if len(reg) else float("nan")
    rm1=float(reg.loc[reg["model"]=="interaction","RMSE"].iloc[0]) if len(reg) else float("nan")
    beta=float(reg.loc[reg["model"]=="interaction","beta_xg"].iloc[0]) if len(reg) else float("nan")
    val=pd.DataFrame([{
        "stage":"pre_evolution" if prefix.startswith("pre_") else "post_evolution",
        "second_difference":second_difference,"beta_xg":beta,
        "additive_R2":r0,"interaction_R2":r1,"delta_R2":r1-r0 if np.isfinite(r0) and np.isfinite(r1) else float("nan"),
        "additive_RMSE":rm0,"interaction_RMSE":rm1,
        "rmse_ratio":rm1/rm0 if rm0>EPS else float("nan"),
        "f00":f00,"f10":f10,"f01":f01,"f11":f11,
    }])
    val.to_csv(outdir/f"{stem}_validation.csv",index=False)
    if plt is not None:
        fig,ax=plt.subplots(figsize=(7.4,5.2))
        ax.scatter(df["Px_times_Pg"],df["P_output"])
        ax.set_xlabel("measured P(X) × P(context glia)")
        ax.set_ylabel("P(output)")
        stage="pre-evolution" if prefix.startswith("pre_") else "post-evolution"
        ax.set_title(f"V7.1 {stage}: multiplicative interaction probe")
        fig.tight_layout(); fig.savefig(outdir/f"{stem}_product_scatter.png",dpi=220); plt.close(fig)
    return df,reg


def run_case_probe(mode,core,base_cfg,gains,hw,evo,states,probe,outdir,prefix=""):

    name=str(probe.get("name","case_probe")); rows=[]
    shots=int(probe.get("shots",evo.shots_override or hw.shots or 3000))
    for ci,case in enumerate(probe.get("cases",[]),1):
        ev=build_events(core,case.get("neural_inputs",[]),gains)
        cfg=materialize_dynamic_config(core,base_cfg,evo.candidates,states,ev)
        readout_mode=str(probe.get("readout_mode","population")).lower()
        _,marg=run_trial_backend(mode,core,cfg,gains,shots,hw,outdir,f"probe_{name}_{ci:03d}",readout_mode=readout_mode)
        row={"probe":name,"case":case.get("name",f"case{ci}")}
        for n in case.get("read_neurons",[]): row[f"P_N{int(n)}"]=float(marg[int(n)])
        for g in case.get("read_glia",[]): row[f"P_G{int(g)}"]=float(marg[cfg.n_neurons+int(g)])
        rows.append(row)
    df=pd.DataFrame(rows); df.to_csv(outdir/f"{prefix}{name}.csv",index=False); return df


def run_probes(mode,core,base_cfg,gains,hw,evo,states,outdir,prefix=""):

    outputs={}
    for p in evo.probes:
        typ=str(p.get("type","case")).lower()
        if typ=="product_grid":
            outputs[p.get("name","product_probe")]=run_product_probe(mode,core,base_cfg,gains,hw,evo,states,p,outdir,prefix=prefix)
        elif typ=="case":
            outputs[p.get("name","case_probe")]=run_case_probe(mode,core,base_cfg,gains,hw,evo,states,p,outdir,prefix=prefix)
        else:
            raise ValueError(f"Unknown probe type: {typ}")
    return outputs


# =============================================================================
# Reporting
# =============================================================================

def save_topology(core,cfg,outdir,prefix="initial"):
    pd.DataFrame([asdict(x) for x in cfg.neuron_to_neuron or []]).to_csv(outdir/f"{prefix}_N_to_N.csv",index=False)
    pd.DataFrame([asdict(x) for x in cfg.neuron_to_glia or []]).to_csv(outdir/f"{prefix}_N_to_G.csv",index=False)
    pd.DataFrame([asdict(x) for x in cfg.glia_glia or []]).to_csv(outdir/f"{prefix}_G_to_G.csv",index=False)
    pd.DataFrame([asdict(x) for x in cfg.glia_to_neuron or []]).to_csv(outdir/f"{prefix}_G_to_N.csv",index=False)
    pd.DataFrame([asdict(x) for x in cfg.tripartite_synapses or []]).to_csv(outdir/f"{prefix}_tripartite.csv",index=False)


def save_candidate_table(evo,outdir):
    rows=[]
    for c in evo.candidates:
        d=asdict(c); d["drivers"]=";".join(c.drivers); d["activation_requires"]=";".join(c.activation_requires); d["competition_groups"]=";".join(c.competition_groups); rows.append(d)
    pd.DataFrame(rows).to_csv(outdir/"candidate_edges.csv",index=False)


def plot_evolution(edge_df,node_df,outdir):
    if plt is None or edge_df.empty: return
    fig,ax=plt.subplots(figsize=(9,5.5))
    for cid,g in edge_df.groupby("candidate_id"):
        ax.plot(g["trial"],g["strength"],label=cid)
    ax.set_xlabel("trial"); ax.set_ylabel("structural strength")
    ax.set_ylim(-0.03,1.03); ax.set_title("V7.1 spatial candidate-edge evolution"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outdir/"edge_strength_evolution.png",dpi=220); plt.close(fig)

    # Plot only glial slow traces to keep readable.
    gd=node_df[node_df["node"].str.startswith("G")]
    if not gd.empty:
        fig,ax=plt.subplots(figsize=(9,5.5))
        for node,g in gd.groupby("node"):
            ax.plot(g["trial"],g["slow_trace"],label=node)
        ax.set_xlabel("trial"); ax.set_ylabel("slow trace"); ax.set_title("V7.1 glial accumulation traces")
        ax.legend(fontsize=8,ncol=2); fig.tight_layout(); fig.savefig(outdir/"glial_trace_evolution.png",dpi=220); plt.close(fig)


def final_manifest(core,base_cfg,gains,bio,hw,evo,traces,states,outdir,args):
    payload={
        "runner_version":VERSION,"core_version":getattr(core,"VERSION","unknown"),
        "scientific_scope":"Hybrid across-trial plasticity rebuilds circuit analogues; no claim of literal biological quantum gates.",
        "mode":args.mode,"base_project":str(Path(args.project).resolve()),"evolution_config":str(Path(args.evolution_config).resolve()),
        "biological_input":asdict(bio),"layer_gains":asdict(gains),"hardware":asdict(hw),
        "trace_settings":asdict(evo.trace),"final_traces":traces,
        "competition":asdict(evo.competition),"spatial_growth":evo.spatial_growth,
        "candidate_states":{k:asdict(v) for k,v in states.items()},
        "base_network":{
            "total_qubits":base_cfg.total_qubits,"layers":base_cfg.layers,
            "initialize_neural_h":getattr(base_cfg,"initialize_neural_h",True),
            "initialize_glial_baseline":getattr(base_cfg,"initialize_glial_baseline",True),
            "neural_readout_mode":getattr(base_cfg,"neural_readout_mode","phase"),
        },
    }
    (outdir/"V7_1_manifest.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap=argparse.ArgumentParser(description="V7.1 spatial-competitive evolving neuroglial network")
    ap.add_argument("--core",default=DEFAULT_CORE)
    ap.add_argument("--project",required=True,help="V7.0/V6.9-style editable biological topology project JSON")
    ap.add_argument("--evolution-config",required=True,help="Slow traces + candidate edge plasticity + trial schedule JSON")
    ap.add_argument("--mode",choices=["resource","surrogate","local","qpu"],default="surrogate")
    ap.add_argument("--output-dir",default=None)
    ap.add_argument("--max-training-trials",type=int,default=None)
    ap.add_argument("--skip-probes",action="store_true")
    ap.add_argument("--freeze-plasticity",action="store_true",help="Run the same trial schedule but do not update slow traces/edges")
    args=ap.parse_args()

    core=load_core(Path(args.core))
    bio,base_cfg,hw,local_shots=core.project_from_json(Path(args.project))
    core.validate_biological_input(bio); core.validate_network_config(base_cfg)
    scores=core.compute_mechanistic_scores(bio); gains=core.scores_to_layer_gains(scores,bio.hardware_gain)
    evo=load_evolution_config(Path(args.evolution_config)); finalize_spatial_candidates(base_cfg,evo); validate_evolution_config(base_cfg,evo)
    if evo.qpu_repeats_override is not None:
        hw=replace(hw,repeats=int(evo.qpu_repeats_override))
    if evo.shots_override is not None and args.mode=="qpu":
        hw=replace(hw,shots=int(evo.shots_override))

    outdir=Path(args.output_dir) if args.output_dir else Path(f"V7_1_evolution_{args.mode}_{timestamp()}")
    outdir.mkdir(parents=True,exist_ok=True)
    save_topology(core,base_cfg,outdir,"initial"); save_candidate_table(evo,outdir)
    pd.DataFrame([asdict(scores)]).to_csv(outdir/"mechanistic_scores.csv",index=False)
    pd.DataFrame([asdict(gains)]).to_csv(outdir/"layer_gains.csv",index=False)

    print(f"Runner: {VERSION}\nCore: {getattr(core,'VERSION','unknown')}\nMode: {args.mode}")
    print(f"Base network: {base_cfg.total_qubits} qubits, {base_cfg.layers} layers")
    print(f"Candidate edges: {len(evo.candidates)}; training phases: {len(evo.training_phases)}")
    print(f"Spatial growth: {bool(evo.spatial_growth.get('enabled',False))}; competition: {evo.competition.enabled}")

    if args.mode=="resource":
        init=core.circuit_resource_estimate(base_cfg)
        tmp_states={c.id:CandidateState(1.0,True) for c in evo.candidates}
        full=materialize_dynamic_config(core,base_cfg,evo.candidates,tmp_states,base_cfg.neural_inputs)
        final=core.circuit_resource_estimate(full)
        (outdir/"resource_initial.json").write_text(json.dumps(init,indent=2),encoding="utf-8")
        (outdir/"resource_all_candidates_active.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
        print("Initial resources:\n",json.dumps(init,indent=2))
        print("All-candidates-active resources:\n",json.dumps(final,indent=2))
        print("DONE:",outdir.resolve()); return 0

    if evo.run_probes_before_training and (not args.skip_probes):
        initial_states=initialize_states(evo)
        run_probes(args.mode,core,base_cfg,gains,hw,evo,initial_states,outdir,prefix="pre_evolution_")

    traces,states,trial_df,node_df,edge_df,em_df=run_training(
        args.mode,core,base_cfg,gains,hw,evo,outdir,args.max_training_trials,args.freeze_plasticity
    )
    trial_df.to_csv(outdir/"trial_summary.csv",index=False)
    node_df.to_csv(outdir/"node_activity_and_slow_trace.csv",index=False)
    edge_df.to_csv(outdir/"edge_evolution.csv",index=False)
    em_df.to_csv(outdir/"emergence_events.csv",index=False)

    final_cfg=materialize_dynamic_config(core,base_cfg,evo.candidates,states,base_cfg.neural_inputs)
    save_topology(core,final_cfg,outdir,"final")
    plot_evolution(edge_df,node_df,outdir)
    if not args.skip_probes:
        run_probes(args.mode,core,base_cfg,gains,hw,evo,states,outdir,prefix="post_evolution_")
    final_manifest(core,base_cfg,gains,bio,hw,evo,traces,states,outdir,args)

    print("\nFinal candidate states:")
    for c in evo.candidates:
        st=states[c.id]
        print(f"  {c.id:24s} {c.type:18s} active={st.active} strength={st.strength:.4f}")
    print("DONE:",outdir.resolve())
    return 0


if __name__=="__main__":
    raise SystemExit(main())
