#!/usr/bin/env python3
"""
V7.1 SAMPLE 4 V1.3 — BUKALO 2026 BLA FEAR / EXTINCTION / RENEWAL
================================================================
Experimentally constrained reduced neuroglial circuit built on V7.1.

Source biology:
Bukalo O et al. Astrocytes enable amygdala neural representations supporting
memory. Nature (2026), doi:10.1038/s41586-025-10068-0.

This program constrains the task schedule and validation targets to the published
source data. It does NOT claim that G0..G9 are literal astrocyte cell types, that
V7.1 protein-to-gate mappings were measured in BLA, or that brain astrocytes are
physical qubits. Slow structural plasticity and history-dependent responsiveness are classical;
local/qpu modes execute the V7.1 quantum-formal fast circuit sequentially.

V1.3 competing-memory upgrade:
- separates retained fear-memory storage from its current astrocytic/neural expression;
- extinction learns a second context-dependent inhibitory/safety memory rather than rapidly erasing the fear memory;
- Context-B is required for downstream extinction expression, preventing a CS-only bypass;
- Context-A provides a modest reinstatement bias through the previously introduced G1-G3 context/fear association;
- DREADD perturbations continue to alter both fast signalling and extinction-plasticity eligibility.
These additions are MODEL HYPOTHESES constrained by the published conditioning/extinction/renewal and perturbation data.
Bukalo et al. did not measure literal G1-G3 edges, memory-state variables, or quantum-gate parameters.
"""
from __future__ import annotations
import argparse, copy, csv, importlib.util, json, math, sys, zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

VERSION = "V7.1-SAMPLE4-BLA-1.3"
DEFAULT_CORE = "V7_1_neuroglial_gate_qpu.py"
DEFAULT_ENGINE = "V7_1_SPATIAL_EVOLVING_NETWORK.py"
DEFAULT_PROJECT = "V7_1_SAMPLE4_BLA_project.json"
DEFAULT_EVOLUTION = "V7_1_SAMPLE4_BLA_evolution.json"
DEFAULT_TASK = "V7_1_SAMPLE4_BLA_task_PRIMARY.json"
EPS=1e-12

# Node map: population-level coarse graining, NOT literal cell identities.
NODE_MAP = {
 "N0":"auditory conditioned stimulus (CS)",
 "N1":"aversive unconditioned stimulus (US)",
 "N2":"conditioning context A representation",
 "N3":"extinction context B representation",
 "N4":"BLA fear-memory / conditionally CS-excited neuronal ensemble",
 "N5":"BLA extinction-plastic / safety neuronal ensemble",
 "N6":"BLA->PL fear-readout population",
 "N7":"extinction/safety relay",
 "N8":"FREEZE output",
 "N9":"fear-suppression / NO-FREEZE output",
 "G0":"US-responsive BLA astrocyte state",
 "G1":"learned CS/fear-associated astrocyte state",
 "G2":"extinction-adapting astrocyte state",
 "G3":"conditioning-context-associated astrocyte state used for renewal/reinstatement",
 "G4":"astrocyte-supported BLA->PL microdomain",
 "G5":"extinction-context-associated astrocyte state",
 "G6":"local distractor astrocyte domain",
 "G7":"local distractor astrocyte domain",
 "G8":"reserved distractor astrocyte domain",
 "G9":"reserved distractor astrocyte domain",
}

ROLE_IDS = {
 "fear_gg":"AUTO_GG_G0_G1",
 "fear_context_gg":"AUTO_GG_G1_G3",
 "fear_operator":"AUTO_TRI_N0_G1_N4",
 "ext_gg":"AUTO_GG_G2_G5",
 "ext_operator":"AUTO_TRI_N3_G2_N5",
 "ext_context":"AUTO_TRI_N3_G2_N9",
}

@dataclass
class TaskSettings:
    name: str="Sample4 Bukalo2026 BLA fear-extinction-renewal"
    seed: int=20260817
    conditioning_pairings: int=3
    ext_day1_trials: int=25
    ext_day2_trials: int=25
    retrieval_trials: int=5
    renewal_trials: int=5
    early_extinction_block: int=5
    late_extinction_block: int=5
    cs_probability: float=0.96
    us_probability: float=0.96
    context_probability: float=0.92
    behavior_floor: float=0.04
    # MODEL_HYPOTHESIS parameters. They regulate expression of stored assemblies;
    # empirical stage values are never inserted into the decoder.
    fear_memory_capacity: float=3.0
    extinction_memory_capacity: float=2.6
    fear_memory_expression_floor: float=0.55
    extinction_memory_expression_floor: float=0.30
    cross_state_suppression_gain: float=0.75
    context_a_reinstatement_gain: float=0.35
    reactivation_weight_cap: float=3.4
    agents: tuple[str,...]=("full_biological",)
    empirical_constraints: str="bukalo2026_empirical_constraints.csv"

def load_module(name:str,path:Path):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise RuntimeError(f"Cannot import {path}")
    m=importlib.util.module_from_spec(spec); sys.modules[name]=m; spec.loader.exec_module(m); return m

def load_task(path:Path)->TaskSettings:
    d=json.loads(path.read_text(encoding="utf-8"))
    kw={k:d[k] for k in TaskSettings.__dataclass_fields__ if k in d}
    if "agents" in kw: kw["agents"]=tuple(kw["agents"])
    return TaskSettings(**kw)

def resolve(here:Path,p:str|Path)->Path:
    q=Path(p); return q if q.exists() else here/q

def stable_int(s:str)->int: return int(zlib.crc32(s.encode()) & 0xffffffff)

def load_empirical(path:Path)->pd.DataFrame:
    return pd.read_csv(path)

def empirical_value(emp:pd.DataFrame,category:str,metric:str,stage:str)->float:
    x=emp[(emp.category==category)&(emp.metric==metric)&(emp.stage==stage)]
    return float(x.iloc[0].value) if len(x) else float('nan')

def derived_scalars(emp:pd.DataFrame)->dict[str,float]:
    x=emp[emp.category=="derived_scalar"]
    return {str(r.metric):float(r.value) for _,r in x.iterrows()}

def patch_candidates(evo, agent:str, scalars:dict[str,float]):
    """Assign history eligibility after spatial candidate generation.

    V1.3 separates four levels:
      1) anatomical/causal possibility graph from the JSON;
      2) history-dependent structural selection;
      3) retained structural memory (fear and extinction assemblies);
      4) context-dependent expression of those stored assemblies.

    The new G1<->G3 link is a model prediction motivated by context-dependent
    renewal; Bukalo et al. did not directly measure this specific astrocyte edge.
    """
    ids={c.id:c for c in evo.candidates}
    missing=[v for v in ROLE_IDS.values() if v not in ids]
    if missing: raise RuntimeError(f"Missing diagnostic spatial candidates: {missing}")
    for c in evo.candidates:
        c.growth_threshold=max(c.growth_threshold,0.88)
        c.learning_rate=min(c.learning_rate,0.025)
        c.on_threshold=max(c.on_threshold,0.95)
        c.off_threshold=min(c.off_threshold,0.24)
        c.structural_decay=max(c.structural_decay,0.018)

    # Fear acquisition: three CS/US pairings. Formation is driven by cue/shock/context
    # history rather than requiring a large pre-existing G1 CS response.
    c=ids[ROLE_IDS['fear_gg']]
    c.drivers=["N0","N1","G0"]; c.driver_mode="min"
    c.growth_threshold=0.005; c.learning_rate=4.0; c.on_threshold=0.08; c.off_threshold=0.015; c.structural_decay=0.0004
    c.max_weight=0.8; c.max_exchange_weight=1.25; c.competition_groups=[]

    # Context-A reinstatement memory: spatially eligible G1<->G3 is selected during
    # conditioning. This is intentionally an explicit model prediction.
    c=ids[ROLE_IDS['fear_context_gg']]
    c.drivers=["N0","N1","G3"]; c.driver_mode="min"
    c.growth_threshold=0.005; c.learning_rate=3.2; c.on_threshold=0.08; c.off_threshold=0.015; c.structural_decay=0.00035
    c.max_weight=0.45; c.max_exchange_weight=0.35; c.competition_groups=[]

    c=ids[ROLE_IDS['fear_operator']]
    c.drivers=["N0","G1","N1"]; c.driver_mode="min"; c.activation_requires=[ROLE_IDS['fear_gg']]
    c.requires_glial_network=False; c.growth_threshold=0.005; c.learning_rate=10.0; c.on_threshold=0.10; c.off_threshold=0.015; c.structural_decay=0.0003; c.max_weight=1.7; c.competition_groups=[]

    # Extinction: Context-B (G5) plus repeated CS supplies eligibility for recruiting G2.
    # This avoids assuming that G2 is already strongly CS-responsive before extinction.
    eligibility=1.0
    if agent=="hm3dq_like": eligibility=float(scalars['hM3Dq_extinction_plasticity_gain'])
    if agent=="hm4di_like": eligibility=float(scalars['hM4Di_extinction_plasticity_gain'])
    # fast_only controls isolate acute signalling from the cross-session plasticity effect.
    if agent in {"hm3dq_fast_only","hm4di_fast_only"}: eligibility=1.0

    c=ids[ROLE_IDS['ext_gg']]
    c.drivers=["N0","N3","G5"]; c.driver_mode="min"
    c.growth_threshold=0.040; c.learning_rate=0.50*eligibility; c.on_threshold=0.20; c.off_threshold=0.06; c.structural_decay=0.0015
    c.max_weight=0.8; c.max_exchange_weight=1.20*eligibility; c.competition_groups=[]
    for role,drivers in [
        ('ext_operator',["N0","G2","N3"]),
        ('ext_context',["N3","G2","N0"]),
    ]:
        c=ids[ROLE_IDS[role]]; c.drivers=drivers; c.driver_mode="min"; c.activation_requires=[ROLE_IDS['ext_gg']]
        c.requires_glial_network=False; c.growth_threshold=0.045; c.learning_rate=0.28*eligibility; c.on_threshold=0.32; c.off_threshold=0.08; c.structural_decay=0.0012
        c.max_weight=1.35*eligibility; c.competition_groups=[]

    evo.competition.enabled=False
    return evo

def schedule(task:TaskSettings, agent:str)->list[dict[str,Any]]:
    rows=[]
    def add(stage,day,context,cs=True,us=False,plastic=True,stage_index=0):
        if agent=="no_context_input": context=None
        rows.append(dict(stage=stage,day=day,context=context,cs=cs,us=us,plasticity=plastic,stage_index=stage_index))
    # F-Con exact schedule: 3 co-presentations CS+US in conditioning context A.
    for i in range(task.conditioning_pairings): add("F-Con",1,"A",True,True,True,i+1)
    # Ext1 and Ext2: 25 CS-alone trials/day in novel context B.
    for i in range(task.ext_day1_trials): add("Ext1",2,"B",True,False,True,i+1)
    for i in range(task.ext_day2_trials): add("Ext2",3,"B",True,False,True,i+1)
    # Retrieval and renewal are readout tests; structural plasticity is frozen.
    for i in range(task.retrieval_trials): add("E-Ret",4,"B",True,False,False,i+1)
    ren_context="A"
    if agent=="context_swap": ren_context="B"
    for i in range(task.renewal_trials): add("F-Ren",5,ren_context,True,False,False,i+1)
    return rows

def input_spec(task:TaskSettings,rec:dict)->list[dict[str,Any]]:
    x=[]
    if rec.get('cs'): x.append({"layer":0,"neuron":0,"probability":task.cs_probability,"event_kind":"evidence"})
    if rec.get('us'): x.append({"layer":0,"neuron":1,"probability":task.us_probability,"event_kind":"evidence"})
    if rec.get('context')=='A': x.append({"layer":0,"neuron":2,"probability":task.context_probability,"event_kind":"evidence"})
    if rec.get('context')=='B': x.append({"layer":0,"neuron":3,"probability":task.context_probability,"event_kind":"evidence"})
    return x

EDGE_WEIGHT_LIMIT = 4.0

def scale_edge_list(edges, predicate, factor):
    """Scale selected directed-edge weights by a common bounded factor.

    The empirical perturbation ratio is kept separately as the biological target.
    Circuit weights are an encoding and the V7.1 core requires |weight| <= 4.
    For gain > 1, one common effective factor is therefore capped by the least
    available headroom among all selected edges, preserving their relative ratios.
    """
    edges=list(edges or [])
    chosen=[e for e in edges if predicate(e)]
    if not chosen:
        return edges
    requested=float(factor)
    effective=requested
    max_abs=max(abs(float(e.weight)) for e in chosen)
    if requested>1.0 and max_abs>0:
        effective=min(requested, EDGE_WEIGHT_LIMIT/max_abs)
    out=[]
    for e in edges:
        if predicate(e):
            w=float(e.weight)*effective
            # Numerical safety only; the common factor above should already satisfy this.
            w=max(-EDGE_WEIGHT_LIMIT,min(EDGE_WEIGHT_LIMIT,w))
            out.append(replace(e,weight=w))
        else:
            out.append(e)
    return out

def bounded_fast_gain(cfg, requested_factor: float) -> float:
    """Return one common N->G/G->N perturbation gain compatible with core bounds."""
    selected=[]
    for e in (cfg.neuron_to_glia or []):
        if int(e.target) in {1,2,4}: selected.append(abs(float(e.weight)))
    for e in (cfg.glia_to_neuron or []):
        if int(e.source)==4 and int(e.target)==6: selected.append(abs(float(e.weight)))
    if not selected or requested_factor<=1.0:
        return float(requested_factor)
    return float(min(float(requested_factor), EDGE_WEIGHT_LIMIT/max(selected)))

def causal_prune_dormant_edges(cfg):
    """Ideal-equivalent QPU compilation pass for clean |0> baselines.
    A controlled operation whose control/source is provably unreachable from any
    current external input is an identity in the ideal circuit. Omitting it avoids
    hardware error accumulation on dormant controls (observed in Sample1 QPU).
    This does not alter the mathematical model.
    """
    active_n=set(int(e.neuron) for e in (cfg.neural_inputs or []))
    active_g=set()
    changed=True
    while changed:
        changed=False
        for e in cfg.neuron_to_neuron or []:
            if int(e.source) in active_n and int(e.target) not in active_n:
                active_n.add(int(e.target)); changed=True
        for e in cfg.neuron_to_glia or []:
            if int(e.source) in active_n and int(e.target) not in active_g:
                active_g.add(int(e.target)); changed=True
        if cfg.enable_glia_exchange:
            for e in cfg.glia_glia or []:
                a,b=int(e.a),int(e.b)
                if a in active_g and b not in active_g: active_g.add(b); changed=True
                if b in active_g and a not in active_g: active_g.add(a); changed=True
        for t in cfg.tripartite_synapses or []:
            if int(t.sensory_neuron) in active_n and int(t.context_glia) in active_g and int(t.target_neuron) not in active_n:
                active_n.add(int(t.target_neuron)); changed=True
        for e in cfg.glia_to_neuron or []:
            if int(e.source) in active_g and int(e.target) not in active_n:
                active_n.add(int(e.target)); changed=True
    return replace(
        cfg,
        neuron_to_neuron=[e for e in (cfg.neuron_to_neuron or []) if int(e.source) in active_n],
        neuron_to_glia=[e for e in (cfg.neuron_to_glia or []) if int(e.source) in active_n],
        glia_glia=[e for e in (cfg.glia_glia or []) if int(e.a) in active_g or int(e.b) in active_g],
        glia_to_neuron=[e for e in (cfg.glia_to_neuron or []) if int(e.source) in active_g],
        tripartite_synapses=[t for t in (cfg.tripartite_synapses or []) if int(t.sensory_neuron) in active_n and int(t.context_glia) in active_g],
    )


def state_strength(states, cid:str)->float:
    st=states.get(cid)
    return float(st.strength) if st is not None else 0.0

def replace_ng_weight(cfg, source:int, target:int, new_weight:float):
    out=[]
    for e in cfg.neuron_to_glia or []:
        if int(e.source)==int(source) and int(e.target)==int(target):
            out.append(replace(e,weight=float(np.clip(new_weight,-EDGE_WEIGHT_LIMIT,EDGE_WEIGHT_LIMIT))))
        else:
            out.append(e)
    return replace(cfg,neuron_to_glia=out)

def history_responsive_config(cfg,traces,states,task:TaskSettings,agent:str,rec:dict):
    """Map stored assemblies into current stimulus-evoked responsiveness.

    V1.3 deliberately distinguishes *memory storage* from *memory expression*.
    The persistent structural assemblies are the coarse-grained storage proxies:
        M_F = min(fear G-G, fear tripartite)
        M_E = min(extinction G-G, extinction tripartite)
    Extinction does not rapidly destroy M_F. Instead, Context B permits the
    extinction assembly to suppress current fear expression. Context A removes
    that suppression and, if the learned G1<->G3 association exists, gives only
    a modest reinstatement bias. This is a model-level implementation of the
    empirical facts that extinction reduces fear without preventing later renewal.

    No empirical freezing percentage or stage-specific Ca2+ value is inserted
    into these equations.
    """
    if agent=="no_memory_responsiveness":
        return cfg, {
            'fear_responsive_weight':np.nan,'ext_responsive_weight':np.nan,
            'fear_memory_storage':0.0,'extinction_memory_storage':0.0,
            'fear_expression_gate':np.nan,'extinction_expression_gate':np.nan,
        }

    sf=min(state_strength(states,ROLE_IDS['fear_gg']),state_strength(states,ROLE_IDS['fear_operator']))
    se=min(state_strength(states,ROLE_IDS['ext_gg']),state_strength(states,ROLE_IDS['ext_operator']))
    tf=float(np.clip(traces.get('G1',0.0),0.0,1.0))
    te=float(np.clip(traces.get('G2',0.0),0.0,1.0))
    sctx=float(np.clip(state_strength(states,ROLE_IDS['fear_context_gg']),0.0,1.0))

    base_f=next((float(e.weight) for e in cfg.neuron_to_glia or [] if int(e.source)==0 and int(e.target)==1),0.0)
    base_e=next((float(e.weight) for e in cfg.neuron_to_glia or [] if int(e.source)==0 and int(e.target)==2),0.0)

    # Stored memory is structural. The online Ca2+-like trace modulates expression
    # but cannot erase the stored association merely because the current state is low.
    ff=float(np.clip(task.fear_memory_expression_floor,0.0,1.0))
    ef=float(np.clip(task.extinction_memory_expression_floor,0.0,1.0))
    if agent=="no_retained_fear_memory":
        # Matched negative control: expression follows only the current G1 trace;
        # the persistent structural assembly contributes no independent memory floor.
        fear_expression=sf*tf
    else:
        fear_expression=sf*(ff+(1.0-ff)*tf)
    ext_expression=se*(ef+(1.0-ef)*te)

    # Extinction is expressed specifically in Context B. This modulates the
    # current fear response; it does not reduce the stored fear assembly itself.
    in_ctx_b = 1.0 if rec.get('context')=='B' else 0.0
    ext_suppression = float(task.cross_state_suppression_gain)*ext_expression*in_ctx_b

    # Context A reinstatement is deliberately modest. The G1-G3 edge is a model
    # prediction motivated by renewal, not a measured BLA astrocyte connection.
    in_ctx_a = 1.0 if rec.get('context')=='A' else 0.0
    ctx_gain = 1.0
    if agent!="no_context_reinstatement":
        ctx_gain += float(task.context_a_reinstatement_gain)*sctx*in_ctx_a

    fear_gate = ctx_gain/(1.0+ext_suppression)
    ext_gate = in_ctx_b
    wf=(base_f + float(task.fear_memory_capacity)*fear_expression)*fear_gate
    # G2 may be reactivated by CS through the fixed N0->G2 input, but downstream
    # safety output remains Context-B gated by the tripartite topology.
    we=base_e + float(task.extinction_memory_capacity)*ext_expression

    cap=float(task.reactivation_weight_cap)
    wf=float(np.clip(wf,0.0,cap)); we=float(np.clip(we,0.0,cap))
    cfg=replace_ng_weight(cfg,0,1,wf)
    cfg=replace_ng_weight(cfg,0,2,we)
    return cfg, {
        'fear_responsive_weight':wf,'ext_responsive_weight':we,
        'fear_memory_storage':float(sf),'extinction_memory_storage':float(se),
        'fear_expression_gate':float(fear_gate),'extinction_expression_gate':float(ext_gate),
    }

def agent_fast_config(core,cfg,agent:str,stage:str,scalars:dict[str,float]):
    # Causal controls.
    if agent=="neural_only":
        return replace(cfg,neuron_to_glia=[],glia_glia=[],glia_to_neuron=[],tripartite_synapses=[],enable_tripartite_synapses=False,enable_glia_exchange=False)
    if agent=="no_tripartite": return replace(cfg,tripartite_synapses=[],enable_tripartite_synapses=False)
    if agent=="no_glia_glia": return replace(cfg,glia_glia=[],enable_glia_exchange=False)
    # CalEx anchor: Fig4g phototagged BLA->PL CS-responsive fraction 83% -> 31%.
    if agent=="calex_like":
        r=float(scalars['CalEx_BLA_PL_gain'])
        # Fig.4g is a response-FRACTION ratio (31/83), whereas controlled-rotation
        # weights are amplitude-like encoding parameters. In the small-angle limit
        # p~theta^2, so the corresponding encoding gain is sqrt(r), while r itself
        # remains the held-out biological validation target.
        f=float(math.sqrt(max(0.0,r)))
        nn=scale_edge_list(cfg.neuron_to_neuron,lambda e:int(e.target)==6,f)
        gn=scale_edge_list(cfg.glia_to_neuron,lambda e:int(e.target)==6,f)
        return replace(cfg,neuron_to_neuron=nn,glia_to_neuron=gn)
    # Pre-extinction DREADD perturbations are applied only during extinction training.
    if stage in {"Ext1","Ext2"} and agent in {"hm3dq_like","hm4di_like","hm3dq_fast_only","hm4di_fast_only"}:
        is_dq=agent in {"hm3dq_like","hm3dq_fast_only"}
        f_requested=float(scalars['hM3Dq_early_retrieval_fast_gain'] if is_dq else scalars['hM4Di_early_retrieval_fast_gain'])
        # The empirical ratio is a biological perturbation target, not a raw gate weight.
        # Map it into the bounded V7.1 circuit encoding with one common gain so that
        # relative weights across the affected astrocytic route are preserved.
        f=bounded_fast_gain(cfg,f_requested)
        ng=scale_edge_list(cfg.neuron_to_glia,lambda e:int(e.target) in {1,2,4},f)
        gn=scale_edge_list(cfg.glia_to_neuron,lambda e:int(e.source)==4 and int(e.target)==6,f)
        return replace(cfg,neuron_to_glia=ng,glia_to_neuron=gn)
    return cfg

def behavior_readout(marg, task:TaskSettings, agent:str)->dict[str,float]:
    # A population-level behavioral readout is more faithful to the source study
    # than treating one qubit as literal freezing. The three fear-side populations
    # are BLA fear representation (N4), BLA->PL readout (N6), and downstream N8;
    # the extinction side is N5/N7/N9. No fitted coefficients are used.
    p8=float(marg[8]); p9=float(marg[9]); floor=float(task.behavior_floor)
    fear_drive=float(max(float(marg[4]),float(marg[6]),p8))
    suppress_drive=float(max(float(marg[5]),float(marg[7]),p9))
    if agent=="no_behavior_competition":
        freeze=fear_drive/(fear_drive+floor)
    else:
        freeze=fear_drive/(fear_drive+suppress_drive+floor)
    suppress=suppress_drive/(fear_drive+suppress_drive+floor)
    return {
        'P_FREEZE_NODE':p8,'P_SUPPRESS_NODE':p9,'fear_route_drive':fear_drive,'suppression_route_drive':suppress_drive,
        'freeze_score':float(np.clip(freeze,0,1)),
        'suppression_score':float(np.clip(suppress,0,1)),
        'route_conflict':float(min(fear_drive,suppress_drive)),
    }

def active_strength(states,cid):
    st=states.get(cid); return float(st.strength) if st else float('nan')

def stage_label(rec,task:TaskSettings):
    if rec['stage']=='Ext1' and rec['stage_index']<=task.early_extinction_block: return 'E-Ext'
    if rec['stage']=='Ext2' and rec['stage_index']>task.ext_day2_trials-task.late_extinction_block: return 'L-Ext'
    return rec['stage']

def precon_probe(mode,core,engine,base_cfg,gains,hw,task,outdir,agent,scalars):
    rec={'stage':'Pre-Con','day':0,'context':None if agent=='no_context_input' else 'A','cs':True,'us':False,'plasticity':False,'stage_index':1}
    evo_inputs=engine.build_events(core,input_spec(task,rec),gains)
    cfg=replace(base_cfg,neural_inputs=evo_inputs)
    cfg=agent_fast_config(core,cfg,agent,'Pre-Con',scalars)
    if mode=='qpu': cfg=causal_prune_dormant_edges(cfg)
    _,m=engine.run_trial_backend(mode,core,cfg,gains,int(hw.shots),hw,outdir,f"{agent}_precon",readout_mode='population')
    b=behavior_readout(m,task,agent)
    return m,b

def summarize_agent(agent, trials:pd.DataFrame, emp:pd.DataFrame, candidates, states, task:TaskSettings):
    stages=[]
    for stage in ['Pre-Con','F-Con','E-Ext','L-Ext','E-Ret','F-Ren']:
        g=trials[trials.analysis_stage==stage]
        if not len(g): continue
        rec={'agent':agent,'stage':stage,'n_model_trials':len(g),'model_freezing_percent':100*float(g.freeze_score.mean()),
             'model_P_BLA_PL':float(g.P_N6.mean()),'model_astro_fear_proxy':float(g.astro_fear_proxy.mean()),
             'model_route_conflict':float(g.route_conflict.mean()),
             'model_fear_memory_storage':float(g.fear_memory_storage.mean()) if 'fear_memory_storage' in g else np.nan,
             'model_extinction_memory_storage':float(g.extinction_memory_storage.mean()) if 'extinction_memory_storage' in g else np.nan,
             'model_fear_expression_gate':float(g.fear_expression_gate.mean()) if 'fear_expression_gate' in g else np.nan,
             'model_extinction_expression_gate':float(g.extinction_expression_gate.mean()) if 'extinction_expression_gate' in g else np.nan}
        ev=empirical_value(emp,'primary_stage','freezing_percent',stage)
        ea=empirical_value(emp,'primary_stage','cs_related_astrocyte_ca_transients',stage)
        rec['empirical_freezing_percent']=ev; rec['empirical_astro_ca_transients']=ea
        stages.append(rec)
    sdf=pd.DataFrame(stages)
    # Shape-only astrocyte validation: empirical transient counts are not used to drive
    # the model. P(G1) is the preregistered learned fear-astrocyte proxy.
    for col,newcol in [('model_astro_fear_proxy','model_astro_relative_to_EExt'),('empirical_astro_ca_transients','empirical_astro_relative_to_EExt')]:
        z=sdf[sdf.stage=='E-Ext']
        den=float(z.iloc[0][col]) if len(z) and np.isfinite(float(z.iloc[0][col])) and float(z.iloc[0][col])>0 else np.nan
        sdf[newcol]=sdf[col]/den if np.isfinite(den) else np.nan
    # Directional signature counts are primary; correlations are descriptive.
    def val(stage,col):
        x=sdf[sdf.stage==stage]; return float(x.iloc[0][col]) if len(x) else np.nan
    sigs={
      'freeze_EExt_gt_LExt':val('E-Ext','model_freezing_percent')>val('L-Ext','model_freezing_percent'),
      'freeze_EExt_gt_ERet':val('E-Ext','model_freezing_percent')>val('E-Ret','model_freezing_percent'),
      'freeze_FRen_gt_ERet':val('F-Ren','model_freezing_percent')>val('E-Ret','model_freezing_percent'),
      'freeze_FRen_gt_LExt':val('F-Ren','model_freezing_percent')>val('L-Ext','model_freezing_percent'),
    }
    astro_sigs={
      'astro_EExt_gt_LExt':val('E-Ext','model_astro_fear_proxy')>val('L-Ext','model_astro_fear_proxy'),
      'astro_EExt_gt_ERet':val('E-Ext','model_astro_fear_proxy')>val('E-Ret','model_astro_fear_proxy'),
      'astro_FRen_gt_ERet':val('F-Ren','model_astro_fear_proxy')>val('E-Ret','model_astro_fear_proxy'),
      'astro_FRen_gt_LExt':val('F-Ren','model_astro_fear_proxy')>val('L-Ext','model_astro_fear_proxy'),
    }
    comparable=sdf[sdf.stage.isin(['E-Ext','L-Ext','E-Ret','F-Ren'])].dropna(subset=['empirical_freezing_percent'])
    pearson=np.nan
    if len(comparable)>=3 and comparable.model_freezing_percent.std()>0:
        pearson=float(np.corrcoef(comparable.model_freezing_percent,comparable.empirical_freezing_percent)[0,1])
    acomp=sdf[sdf.stage.isin(['E-Ext','L-Ext','E-Ret','F-Ren'])].dropna(subset=['empirical_astro_ca_transients'])
    astro_pearson=np.nan; astro_norm_rmse=np.nan
    if len(acomp)>=3 and acomp.model_astro_fear_proxy.std()>0:
        astro_pearson=float(np.corrcoef(acomp.model_astro_fear_proxy,acomp.empirical_astro_ca_transients)[0,1])
        if acomp.model_astro_relative_to_EExt.notna().all():
            astro_norm_rmse=float(np.sqrt(np.mean((acomp.model_astro_relative_to_EExt-acomp.empirical_astro_relative_to_EExt)**2)))
    out={
      'agent':agent,'directional_freezing_passes':sum(bool(v) for v in sigs.values()),'directional_freezing_total':len(sigs),
      'directional_astro_passes':sum(bool(v) for v in astro_sigs.values()),'directional_astro_total':len(astro_sigs),
      'freezing_stage_pearson':pearson,'astro_stage_pearson':astro_pearson,'astro_normalized_rmse':astro_norm_rmse,'final_fear_assembly_min_strength':min(active_strength(states,ROLE_IDS['fear_gg']),active_strength(states,ROLE_IDS['fear_context_gg']),active_strength(states,ROLE_IDS['fear_operator'])),
      'final_extinction_assembly_min_strength':min(active_strength(states,ROLE_IDS['ext_gg']),active_strength(states,ROLE_IDS['ext_operator']),active_strength(states,ROLE_IDS['ext_context'])),
    }
    out.update({k:int(v) for k,v in sigs.items()}); out.update({k:int(v) for k,v in astro_sigs.items()})
    return sdf,out

def run_one_agent(mode,core,engine,bio,base_cfg,hw,gains,evo_template,task,emp,agent,outdir,seed):
    scalars=derived_scalars(emp)
    evo=copy.deepcopy(evo_template); evo=patch_candidates(evo,agent,scalars)
    # Initialize the complete state dictionary before pruning so dependency IDs
    # remain defined (inactive) in matched ablations such as no_glia_glia.
    states=engine.initialize_states(evo); traces=engine.initialize_traces(base_cfg,evo)
    candidates=evo.candidates
    # Remove structural mechanisms for matched ablations.
    if agent=='no_tripartite': candidates=[c for c in candidates if c.type!='tripartite']; evo.candidates=candidates
    if agent=='no_glia_glia': candidates=[c for c in candidates if c.type!='glia_glia']; evo.candidates=candidates
    if agent=='no_context_reinstatement': candidates=[c for c in candidates if c.id!=ROLE_IDS['fear_context_gg']]; evo.candidates=candidates
    if agent=='neural_only': candidates=[]; evo.candidates=[]
    rows=[]; events=[]; edge_rows=[]
    # Pre-conditioning CS-only checkpoint.
    pm,pb=precon_probe(mode,core,engine,base_cfg,gains,hw,task,outdir,agent,scalars)
    rows.append({'agent':agent,'trial':0,'stage':'Pre-Con','analysis_stage':'Pre-Con','day':0,'stage_index':1,'context':'A','plasticity':False,
                 **pb,'P_N4':float(pm[4]),'P_N5':float(pm[5]),'P_N6':float(pm[6]),'P_N7':float(pm[7]),
                 'P_G0':float(pm[10]),'P_G1':float(pm[11]),'P_G2':float(pm[12]),'P_G3':float(pm[13]),'P_G4':float(pm[14]),'P_G5':float(pm[15]),
                 'astro_fear_proxy':float(pm[11]),'fear_responsive_weight':np.nan,'ext_responsive_weight':np.nan,
                 'fear_memory_storage':0.0,'extinction_memory_storage':0.0,'fear_expression_gate':np.nan,'extinction_expression_gate':np.nan,
                 'fear_assembly_min_strength':0.0,'extinction_assembly_min_strength':0.0})
    for t,rec in enumerate(schedule(task,agent),start=1):
        specs=input_spec(task,rec); inp=engine.build_events(core,specs,gains)
        cfg=engine.materialize_dynamic_config(core,base_cfg,candidates,states,inp)
        cfg,resp=history_responsive_config(cfg,traces,states,task,agent,rec)
        cfg=agent_fast_config(core,cfg,agent,rec['stage'],scalars)
        if mode=='qpu': cfg=causal_prune_dormant_edges(cfg)
        _,marg=engine.run_trial_backend(mode,core,cfg,gains,int(hw.shots),hw,outdir,f"{agent}_t{t:03d}_{rec['stage']}",readout_mode='population')
        b=behavior_readout(marg,task,agent)
        fear_min=min(active_strength(states,ROLE_IDS['fear_gg']) if ROLE_IDS['fear_gg'] in states else 0,
                     active_strength(states,ROLE_IDS['fear_operator']) if ROLE_IDS['fear_operator'] in states else 0,
                     active_strength(states,ROLE_IDS['fear_context_gg']) if ROLE_IDS['fear_context_gg'] in states else 0)
        ext_min=min(active_strength(states,ROLE_IDS['ext_gg']) if ROLE_IDS['ext_gg'] in states else 0,
                    active_strength(states,ROLE_IDS['ext_operator']) if ROLE_IDS['ext_operator'] in states else 0,
                    active_strength(states,ROLE_IDS['ext_context']) if ROLE_IDS['ext_context'] in states else 0)
        ar=stage_label(rec,task)
        rows.append({'agent':agent,'trial':t,'stage':rec['stage'],'analysis_stage':ar,'day':rec['day'],'stage_index':rec['stage_index'],'context':rec['context'],'plasticity':rec['plasticity'],
                     **b,'P_N4':float(marg[4]),'P_N5':float(marg[5]),'P_N6':float(marg[6]),'P_N7':float(marg[7]),
                     'P_G0':float(marg[10]),'P_G1':float(marg[11]),'P_G2':float(marg[12]),'P_G3':float(marg[13]),'P_G4':float(marg[14]),'P_G5':float(marg[15]),
                     'astro_fear_proxy':float(marg[11]),**resp,
                     'fear_assembly_min_strength':fear_min,'extinction_assembly_min_strength':ext_min})
        # Update traces and structural state only during training phases.
        if rec['plasticity'] and agent not in {'frozen_plasticity','neural_only'}:
            engine.update_traces(traces,marg,base_cfg,evo.trace)
            ev=engine.update_candidate_states(candidates,states,traces,evo.competition)
            for e in ev: e.update({'agent':agent,'trial':t,'stage':rec['stage']}); events.append(e)
        # Trace dynamics may continue without structural updates at test only for diagnostics.
        elif rec['plasticity'] and agent=='frozen_plasticity':
            engine.update_traces(traces,marg,base_cfg,evo.trace)
        for c in candidates:
            st=states[c.id]
            if c.id in ROLE_IDS.values():
                edge_rows.append({'agent':agent,'trial':t,'stage':rec['stage'],'candidate_id':c.id,'edge_type':c.type,'strength':st.strength,'active':st.active,'driver':engine.driver_value(c,traces)})
    tdf=pd.DataFrame(rows); edf=pd.DataFrame(events); xdf=pd.DataFrame(edge_rows)
    sdf,summary=summarize_agent(agent,tdf,emp,candidates,states,task)
    # Perturbation anchors.
    if agent=='calex_like': summary['empirical_target_BLA_PL_ratio']=scalars['CalEx_BLA_PL_gain']
    if agent=='hm3dq_like':
        summary['empirical_target_EExt_fast_ratio']=scalars['hM3Dq_early_retrieval_fast_gain']
        summary['circuit_effective_fast_gain']=bounded_fast_gain(base_cfg,float(scalars['hM3Dq_early_retrieval_fast_gain']))
    if agent=='hm4di_like':
        summary['empirical_target_EExt_fast_ratio']=scalars['hM4Di_early_retrieval_fast_gain']
        summary['circuit_effective_fast_gain']=bounded_fast_gain(base_cfg,float(scalars['hM4Di_early_retrieval_fast_gain']))
    return tdf,edf,xdf,sdf,summary

def resource_report(core,engine,base_cfg,evo,task):
    evo=engine.finalize_spatial_candidates(base_cfg,evo)
    ids=set(c.id for c in evo.candidates)
    print(f"Sample4 resource check: total_qubits={base_cfg.total_qubits}, neurons={base_cfg.n_neurons}, glia={base_cfg.n_glia}")
    print(f"Exact candidate_edges supplied in JSON: 0; auto-generated candidates={len(evo.candidates)}")
    for k,v in ROLE_IDS.items(): print(f"  {k:16s} {v:26s} present={v in ids}")
    bypass=["AUTO_TRI_N0_G2_N5","AUTO_TRI_N0_G2_N9"]
    for bid in bypass: print(f"  context_bypass    {bid:26s} present={bid in ids} (must be False in V1.3)")
    if any(b in ids for b in bypass):
        raise RuntimeError("V1.3 causal topology violation: context-independent extinction-output candidate was generated.")
    print(f"Biological schedule: 3 CS/US + {task.ext_day1_trials}+{task.ext_day2_trials} CS-alone + {task.retrieval_trials} E-Ret + {task.renewal_trials} F-Ren")
    print("IMPORTANT: schedule/effect-size targets are empirical. Retained-memory expression and G1<->G3 reinstatement are explicit model hypotheses; Context-B gating is a reduced causal representation of context-dependent extinction, not a measured single synapse.")

def main():
    ap=argparse.ArgumentParser(description="V7.1 Sample4 V1.3 Bukalo2026 BLA competing-memory fear-extinction-renewal")
    ap.add_argument('--core',default=DEFAULT_CORE); ap.add_argument('--engine',default=DEFAULT_ENGINE)
    ap.add_argument('--project',default=DEFAULT_PROJECT); ap.add_argument('--evolution-config',default=DEFAULT_EVOLUTION); ap.add_argument('--task-config',default=DEFAULT_TASK)
    ap.add_argument('--mode',choices=['resource','surrogate','local','qpu'],default='surrogate')
    ap.add_argument('--agents',default=None,help='comma-separated agent override'); ap.add_argument('--n-seeds',type=int,default=1); ap.add_argument('--seed-offset',type=int,default=0)
    ap.add_argument('--output-dir',default=None)
    args=ap.parse_args(); here=Path(__file__).resolve().parent
    core=load_module('sample4_core',resolve(here,args.core)); engine=load_module('sample4_engine',resolve(here,args.engine))
    bio,base_cfg,hw,local_shots=core.project_from_json(resolve(here,args.project)); core.validate_biological_input(bio); core.validate_network_config(base_cfg)
    gains=core.scores_to_layer_gains(core.compute_mechanistic_scores(bio),bio.hardware_gain)
    evo0=engine.load_evolution_config(resolve(here,args.evolution_config)); evo0=engine.finalize_spatial_candidates(base_cfg,evo0)
    task=load_task(resolve(here,args.task_config)); emp=load_empirical(resolve(here,task.empirical_constraints))
    if args.agents: task.agents=tuple(x.strip() for x in args.agents.split(',') if x.strip())
    if evo0.shots_override is not None: hw=replace(hw,shots=int(evo0.shots_override))
    if evo0.qpu_repeats_override is not None: hw=replace(hw,repeats=int(evo0.qpu_repeats_override))
    if args.mode=='resource': resource_report(core,engine,base_cfg,evo0,task); return
    if args.n_seeds<1: raise ValueError('--n-seeds must be >=1')
    root=Path(args.output_dir) if args.output_dir else here/f"sample4_{args.mode}_output"
    root.mkdir(parents=True,exist_ok=True)
    all_trials=[]; all_events=[]; all_edges=[]; all_stages=[]; summaries=[]
    for si in range(args.n_seeds):
        seed=task.seed+args.seed_offset+si
        for agent in task.agents:
            print(f"[{args.mode}] seed={seed} agent={agent}")
            t,e,x,s,sm=run_one_agent(args.mode,core,engine,bio,base_cfg,hw,gains,evo0,task,emp,agent,root,seed)
            for df in (t,e,x,s):
                if len(df): df['seed']=seed
            sm['seed']=seed
            all_trials.append(t); all_events.append(e); all_edges.append(x); all_stages.append(s); summaries.append(sm)
    trials=pd.concat(all_trials,ignore_index=True) if all_trials else pd.DataFrame(); events=pd.concat(all_events,ignore_index=True) if any(len(x) for x in all_events) else pd.DataFrame()
    edges=pd.concat(all_edges,ignore_index=True) if all_edges else pd.DataFrame(); stages=pd.concat(all_stages,ignore_index=True) if all_stages else pd.DataFrame(); seed_summary=pd.DataFrame(summaries)
    trials.to_csv(root/'sample4_trial_log.csv',index=False); events.to_csv(root/'sample4_emergence_events.csv',index=False); edges.to_csv(root/'sample4_topology_timeseries.csv',index=False); stages.to_csv(root/'sample4_stage_validation.csv',index=False); seed_summary.to_csv(root/'sample4_seed_summary.csv',index=False)
    # aggregate group summary
    numeric=[c for c in seed_summary.columns if c not in {'agent','seed'} and pd.api.types.is_numeric_dtype(seed_summary[c])]
    gr=[]
    for agent,g in seed_summary.groupby('agent'):
        r={'agent':agent,'n_seeds':len(g)}
        for c in numeric: r[c+'_mean']=float(g[c].mean()); r[c+'_sd']=float(g[c].std(ddof=1)) if len(g)>1 else np.nan
        gr.append(r)
    pd.DataFrame(gr).to_csv(root/'sample4_agent_group_summary.csv',index=False)
    # Empirical perturbation validation: source-data effect sizes vs matched model ratios.
    pv=[]
    st=stages.copy()
    def stage_mean(agent,stage,col):
        z=st[(st.agent==agent)&(st.stage==stage)]
        return float(z[col].mean()) if len(z) else np.nan
    full_e=stage_mean('full_biological','E-Ext','model_freezing_percent')
    full_r=stage_mean('full_biological','E-Ret','model_freezing_percent')
    full_pl=stage_mean('full_biological','E-Ext','model_P_BLA_PL')
    sc=derived_scalars(emp)
    for a,target_key,target_stage,metric in [
        ('hm3dq_like','hM3Dq_early_retrieval_fast_gain','E-Ext','model_freezing_percent'),
        ('hm4di_like','hM4Di_early_retrieval_fast_gain','E-Ext','model_freezing_percent'),
        ('calex_like','CalEx_BLA_PL_gain','E-Ext','model_P_BLA_PL'),
    ]:
        den=full_pl if metric=='model_P_BLA_PL' else full_e
        obs=stage_mean(a,target_stage,metric)
        if a=='calex_like':
            ec=empirical_value(emp,'BLA_to_PL','phototagged_cs_responsive_fraction','Control')
            ep=empirical_value(emp,'BLA_to_PL','phototagged_cs_responsive_fraction','CalEx')
        else:
            cat='hM3Dq_preExt' if a=='hm3dq_like' else 'hM4Di_preExt'
            pertm='freezing_percent_hM3Dq' if a=='hm3dq_like' else 'freezing_percent_hM4Di'
            ec=empirical_value(emp,cat,'freezing_percent_control',target_stage)
            ep=empirical_value(emp,cat,pertm,target_stage)
        pv.append({'agent':a,'stage':target_stage,'metric':metric,'model_value':obs,'model_full_value':den,
                   'model_ratio_to_full':obs/den if den>0 else np.nan,'empirical_control_value':ec,'empirical_perturbed_value':ep,
                   'empirical_target_ratio':sc[target_key]})
    # E-Ret tests whether an extinction-period astrocyte perturbation leaves a next-day memory deficit.
    for a in ['hm3dq_like','hm4di_like']:
        obs=stage_mean(a,'E-Ret','model_freezing_percent')
        cat='hM3Dq_preExt' if a=='hm3dq_like' else 'hM4Di_preExt'
        pertm='freezing_percent_hM3Dq' if a=='hm3dq_like' else 'freezing_percent_hM4Di'
        ec=empirical_value(emp,cat,'freezing_percent_control','E-Ret')
        ep=empirical_value(emp,cat,pertm,'E-Ret')
        pv.append({'agent':a,'stage':'E-Ret','metric':'model_freezing_percent','model_value':obs,'model_full_value':full_r,
                   'model_ratio_to_full':obs/full_r if full_r>0 else np.nan,'empirical_control_value':ec,'empirical_perturbed_value':ep,
                   'empirical_target_ratio':ep/ec if ec>0 else np.nan})
    pvdf=pd.DataFrame(pv)
    # Add absolute values beside ratios so cohort-specific perturbation effects can be audited.
    if len(pvdf):
        pvdf.to_csv(root/'sample4_perturbation_validation.csv',index=False)
    # Transition-level validation is insensitive to a global behavioral scale offset.
    tv=[]
    def ev(stage): return empirical_value(emp,'primary_stage','freezing_percent',stage)
    def mv(stage): return stage_mean('full_biological',stage,'model_freezing_percent')
    for name,a,b in [
        ('extinction_drop','E-Ext','L-Ext'),
        ('retrieval_drop','E-Ext','E-Ret'),
        ('renewal_return','F-Ren','E-Ret'),
    ]:
        model_delta=mv(a)-mv(b) if name!='renewal_return' else mv(a)-mv(b)
        empirical_delta=ev(a)-ev(b) if name!='renewal_return' else ev(a)-ev(b)
        tv.append({'transition':name,'stage_a':a,'stage_b':b,'model_delta_percentage_points':model_delta,'empirical_delta_percentage_points':empirical_delta,'absolute_error_pp':abs(model_delta-empirical_delta)})
    pd.DataFrame(tv).to_csv(root/'sample4_transition_validation.csv',index=False)
    # Copy empirical table into output for direct audit.
    emp.to_csv(root/'sample4_empirical_targets_used.csv',index=False)
    manifest={'version':VERSION,'mode':args.mode,'task':task.__dict__,'node_map':NODE_MAP,'role_ids':ROLE_IDS,
              'circuit_gain_mapping':{
                  'hM3Dq_requested_empirical_ratio':float(sc['hM3Dq_early_retrieval_fast_gain']),
                  'hM3Dq_effective_circuit_gain':bounded_fast_gain(base_cfg,float(sc['hM3Dq_early_retrieval_fast_gain'])),
                  'hM4Di_requested_empirical_ratio':float(sc['hM4Di_early_retrieval_fast_gain']),
                  'hM4Di_effective_circuit_gain':bounded_fast_gain(base_cfg,float(sc['hM4Di_early_retrieval_fast_gain'])),
                  'CalEx_empirical_response_fraction_ratio':float(sc['CalEx_BLA_PL_gain']),
                  'CalEx_circuit_amplitude_gain':float(math.sqrt(max(0.0,float(sc['CalEx_BLA_PL_gain'])))),
                  'directed_edge_weight_limit':EDGE_WEIGHT_LIMIT,
                  'mapping_note':'Empirical response ratios remain validation targets. CalEx response-fraction ratio is mapped to an amplitude-like gate gain by sqrt(r); >1 DREADD gains are common-factor capped to satisfy |weight|<=4.'
              },
              'model_hypotheses':{'history_dependent_responsiveness':True,'retained_fear_memory_storage':'persistent fear assembly strength','context_A_reinstatement_edge':'G1<->G3','context_B_required_for_extinction_output':True,'astrocyte_dependent_extinction_plasticity_eligibility':True},
              'scientific_scope':'Experimentally constrained reduced causal model. Retained-memory expression, Context-B gating, and G1<->G3 reinstatement are model hypotheses/predictions constrained by extinction/renewal data; they are not measured BLA microscopic wiring. Not evidence for microscopic quantum brain computation.',
              'source_doi':'10.1038/s41586-025-10068-0'}
    (root/'sample4_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,default=list),encoding='utf-8')
    print(f"Saved Sample4 outputs to {root}")

if __name__=='__main__': main()
