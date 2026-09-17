#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7.1 SAMPLE 1B — ARITHMETIC COMPILATION BENCHMARK
==================================================

Goal
----
Test a literal, falsifiable version of the "addition -> multiplication" idea:

1) A fixed shallow neural population circuit is first validated as an ADD primitive.
2) Multiplication is initially executed only as a recurrent repeated-addition procedure:
       s_0 = 0;  s_{j+1} = ADD(s_j, a), j=0..b-1.
3) Repeated use of that procedure updates ordinary V7.1 slow traces and spatially
   generated candidate strengths. No product error, product label, reward, or target
   is supplied to plasticity.
4) If local history recruits a G-G bridge and then a tripartite N+G->N route, the
   network is probed with (a,b) once and tested for a direct multiplicative readout.
5) Held-out operand value 4 is never used in topology-training episodes. A simple
   post-hoc affine readout calibrated only on values 1..3 must generalize to pairs
   containing 4.
6) A targeted product-edge ablation and a complete learned-topology reset test whether
   the direct capability depends on the newly generated topology.

Biological scope
----------------
The arithmetic task is an abstract adapter. N0..N6 are population-level functional
roles, not literal "number neurons". The biological claim being tested is narrower:
repeated activity in a re-entrant neuronal procedure can accumulate astrocytic slow
state and recruit a new local higher-order route. The fast layer uses the same V7.1
CRY / G-G / CC-RY analogues; across-trial plasticity remains classical.

QPU scope
---------
This benchmark is deliberately 14 qubits and one circuit layer to reduce hardware
flattening. Sequential training calls cannot be parallelized because each measured
population updates the next slow state/topology. "One-call multiplication" means one
fast-circuit invocation within this bounded population-code benchmark; it is NOT an
asymptotic quantum speedup claim.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "V7.1-SAMPLE1B-ARITHMETIC-COMPILATION-1.0"
DEFAULT_CORE = "V7_1_neuroglial_gate_qpu.py"
DEFAULT_ENGINE = "V7_1_SPATIAL_EVOLVING_NETWORK.py"
DEFAULT_PROJECT = "V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_project.json"
DEFAULT_EVOLUTION = "V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_evolution.json"
DEFAULT_TASK = "V7_1_SAMPLE1B_ARITHMETIC_COMPILATION_task_PRIMARY.json"
EPS = 1e-12
PRODUCT_EDGE_ID = "AUTO_TRI_N1_G1_N6"
REQUIRED_GG_ID = "AUTO_GG_G0_G1"

NODE_MAP = {
    "N0": "re-entrant partial-sum input population",
    "N1": "current addend / operand-a population",
    "N2": "partial-sum transmission population",
    "N3": "addend transmission population",
    "N4": "maintained repetition-demand / operand-b context population",
    "N5": "local alternative/distractor target population",
    "N6": "candidate compiled product readout population",
    "G0": "astrocytic state integrating repeated addition-circuit activity",
    "G1": "astrocytic state coupled to maintained repetition/context activity",
    "G2": "local competing astrocytic domain",
}


@dataclass
class TaskSettings:
    name: str = "Sample1B arithmetic compilation primary"
    seed: int = 20260824
    operand_scale: float = 5.0
    partial_sum_scale: float = 25.0
    addition_anchor_probability: float = 0.8
    training_epochs: int = 3
    training_pairs: list[list[int]] | None = None
    addition_validation_pairs: list[list[float]] | None = None
    product_calibration_pairs: list[list[int]] | None = None
    product_holdout_pairs: list[list[int]] | None = None
    run_targeted_product_ablation: bool = True
    run_topology_reset: bool = True
    shots: int = 5000
    early_stop_after_product_strength: float | None = None
    minimum_post_compile_consolidation_calls: int = 0
    validation_profile: str = "primary"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve(here: Path, p: str | Path) -> Path:
    q = Path(p)
    return q if q.exists() else here / q


def load_task(path: Path) -> TaskSettings:
    d = json.loads(path.read_text(encoding="utf-8"))
    return TaskSettings(**d)


def validate_task(t: TaskSettings) -> None:
    if t.operand_scale <= 0 or t.partial_sum_scale <= 0:
        raise ValueError("operand_scale and partial_sum_scale must be positive")
    if not (0 < t.addition_anchor_probability <= 1):
        raise ValueError("addition_anchor_probability must be in (0,1]")
    if t.training_epochs < 1:
        raise ValueError("training_epochs must be >=1")
    for name in ("training_pairs", "product_calibration_pairs", "product_holdout_pairs"):
        pairs = getattr(t, name) or []
        if not pairs:
            raise ValueError(f"{name} must not be empty")
        for a, b in pairs:
            if a < 0 or b < 0:
                raise ValueError(f"negative arithmetic operand in {name}")
    train_values = {int(x) for pair in (t.training_pairs or []) for x in pair}
    if 4 in train_values:
        raise ValueError("Primary benchmark reserves operand value 4 for held-out generalization; remove 4 from training_pairs")
    holdout = [tuple(map(int, p)) for p in (t.product_holdout_pairs or [])]
    if not any(4 in p for p in holdout):
        raise ValueError("product_holdout_pairs should contain unseen operand value 4")


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    sst = float(np.sum((y-y.mean())**2)); ssr = float(np.sum((y-pred)**2))
    return 1.0-ssr/sst if sst > EPS else float("nan")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y=np.asarray(y,float); pred=np.asarray(pred,float)
    err=pred-y
    rmse=float(np.sqrt(np.mean(err**2)))
    mae=float(np.mean(np.abs(err)))
    denom=float(np.max(y)-np.min(y))
    return {"R2":r2_score(y,pred),"RMSE":rmse,"MAE":mae,"NRMSE_range":rmse/denom if denom>EPS else float("nan")}


def state_strength(states, cid: str) -> float:
    st=states.get(cid)
    return float(st.strength) if st is not None else 0.0


def state_active(states, cid: str) -> bool:
    st=states.get(cid)
    return bool(st.active) if st is not None else False


def run_cfg(mode, core, engine, base_cfg, gains, hw, evo, states, specs, outdir, label, shots):
    events=engine.build_events(core, specs, gains)
    cfg=engine.materialize_dynamic_config(core, base_cfg, evo.candidates, states, events)
    _,marg=engine.run_trial_backend(mode,core,cfg,gains,int(shots),hw,outdir,label,readout_mode="population")
    return marg,cfg


def calibrate_addition(mode,core,engine,base_cfg,gains,hw,evo,states,task,outdir,shots,prefix="pre"):
    p=float(task.addition_anchor_probability)
    rows=[]
    m0,_=run_cfg(mode,core,engine,base_cfg,gains,hw,evo,states,[],outdir,f"{prefix}_addcal_baseline",shots)
    rows.append({"case":"baseline","P_N2":m0[2],"P_N3":m0[3]})
    mp,_=run_cfg(mode,core,engine,base_cfg,gains,hw,evo,states,[{"layer":0,"neuron":0,"probability":p,"event_kind":"evidence"}],outdir,f"{prefix}_addcal_partial",shots)
    rows.append({"case":"partial_anchor","P_N2":mp[2],"P_N3":mp[3]})
    ma,_=run_cfg(mode,core,engine,base_cfg,gains,hw,evo,states,[{"layer":0,"neuron":1,"probability":p,"event_kind":"evidence"}],outdir,f"{prefix}_addcal_addend",shots)
    rows.append({"case":"addend_anchor","P_N2":ma[2],"P_N3":ma[3]})
    b2=float(m0[2]); b3=float(m0[3])
    k2=(float(mp[2])-b2)/p
    k3=(float(ma[3])-b3)/p
    if k2 <= EPS or k3 <= EPS:
        raise RuntimeError(f"Addition calibration failed: k_partial={k2}, k_addend={k3}")
    cal={"baseline_N2":b2,"baseline_N3":b3,"gain_partial":k2,"gain_addend":k3,"anchor_probability":p}
    pd.DataFrame(rows).to_csv(outdir/f"{prefix}_addition_calibration.csv",index=False)
    return cal


def decode_addition(marg,cal,task):
    p2=max(0.0,float(marg[2])-cal["baseline_N2"])
    p3=max(0.0,float(marg[3])-cal["baseline_N3"])
    partial_hat=p2/max(EPS,cal["gain_partial"])*float(task.partial_sum_scale)
    addend_hat=p3/max(EPS,cal["gain_addend"])*float(task.operand_scale)
    return partial_hat,addend_hat,partial_hat+addend_hat


def addition_validation(mode,core,engine,base_cfg,gains,hw,evo,states,task,cal,outdir,shots,prefix):
    rows=[]
    for i,(x,a) in enumerate(task.addition_validation_pairs or [],1):
        px=float(x)/float(task.partial_sum_scale); pa=float(a)/float(task.operand_scale)
        if px>1+EPS or pa>1+EPS:
            raise ValueError(f"Addition validation pair {(x,a)} exceeds encoding scale")
        specs=[{"layer":0,"neuron":0,"probability":px,"event_kind":"evidence"},
               {"layer":0,"neuron":1,"probability":pa,"event_kind":"evidence"}]
        m,_=run_cfg(mode,core,engine,base_cfg,gains,hw,evo,states,specs,outdir,f"{prefix}_addval_{i:02d}",shots)
        xh,ah,sh=decode_addition(m,cal,task)
        rows.append({"partial_sum":x,"addend":a,"true_sum":float(x)+float(a),"decoded_partial":xh,"decoded_addend":ah,"predicted_sum":sh,"P_N2":m[2],"P_N3":m[3]})
    df=pd.DataFrame(rows); df.to_csv(outdir/f"{prefix}_addition_primitive_validation.csv",index=False)
    mm=metrics(df.true_sum.to_numpy(float),df.predicted_sum.to_numpy(float))
    return df,mm


def product_specs(a,b,task):
    pa=float(a)/float(task.operand_scale); pb=float(b)/float(task.operand_scale)
    if pa>1+EPS or pb>1+EPS:
        raise ValueError(f"Product pair {(a,b)} exceeds operand_scale={task.operand_scale}")
    return [
        {"layer":0,"neuron":1,"probability":pa,"event_kind":"evidence"},
        {"layer":0,"neuron":4,"probability":pb,"event_kind":"tonic"},
    ]


def direct_product_probe(mode,core,engine,base_cfg,gains,hw,evo,states,task,outdir,shots,phase,pairs):
    rows=[]
    for i,(a,b) in enumerate(pairs,1):
        m,_=run_cfg(mode,core,engine,base_cfg,gains,hw,evo,states,product_specs(a,b,task),outdir,f"{phase}_prod_{i:02d}_a{a}_b{b}",shots)
        rows.append({"phase":phase,"a":int(a),"b":int(b),"true_product":int(a)*int(b),
                     "P_a_input":float(m[1]),"P_context_input":float(m[4]),
                     "P_context_glia":float(m[base_cfg.n_neurons+1]),"P_product":float(m[6])})
    return pd.DataFrame(rows)


def fit_interaction_models(df: pd.DataFrame) -> dict[str,float]:
    y=df.P_product.to_numpy(float); x=df.P_a_input.to_numpy(float); g=df.P_context_glia.to_numpy(float)
    X0=np.column_stack([np.ones(len(df)),x,g]); X1=np.column_stack([np.ones(len(df)),x,g,x*g])
    b0=np.linalg.lstsq(X0,y,rcond=None)[0]; b1=np.linalg.lstsq(X1,y,rcond=None)[0]
    p0=X0@b0; p1=X1@b1
    return {"additive_R2":r2_score(y,p0),"interaction_R2":r2_score(y,p1),"delta_R2":r2_score(y,p1)-r2_score(y,p0),
            "beta_xg":float(b1[3]),"additive_RMSE":float(np.sqrt(np.mean((y-p0)**2))),
            "interaction_RMSE":float(np.sqrt(np.mean((y-p1)**2)))}


def mixed_second_difference(df: pd.DataFrame, low=1, high=4) -> float:
    def val(a,b):
        z=df[(df.a==a)&(df.b==b)]
        return float(z.P_product.mean()) if len(z) else float("nan")
    fhh, fhl, flh, fll = val(high,high), val(high,low), val(low,high), val(low,low)
    if not all(math.isfinite(v) for v in (fhh,fhl,flh,fll)):
        return float("nan")
    return fhh-fhl-flh+fll


def fit_posthoc_product_decoder(post_cal: pd.DataFrame):
    x=post_cal.P_product.to_numpy(float); y=post_cal.true_product.to_numpy(float)
    X=np.column_stack([np.ones(len(x)),x]); beta=np.linalg.lstsq(X,y,rcond=None)[0]
    return float(beta[0]),float(beta[1])


def apply_product_decoder(df,intercept,slope):
    z=df.copy(); z["predicted_product"]=intercept+slope*z.P_product.astype(float)
    return z


def clone_reset_states(engine,evo):
    return engine.initialize_states(evo)


def clone_product_ablation(states):
    z=copy.deepcopy(states)
    if PRODUCT_EDGE_ID in z:
        z[PRODUCT_EDGE_ID].active=False; z[PRODUCT_EDGE_ID].strength=0.0
    return z


def train_repeated_addition(mode,core,engine,base_cfg,gains,hw,evo,task,cal,outdir,shots,agent="full"):
    local_evo=copy.deepcopy(evo)
    if agent=="no_tripartite_growth":
        local_evo.candidates=[c for c in local_evo.candidates if c.type!="tripartite"]
    states=engine.initialize_states(local_evo); traces=engine.initialize_traces(base_cfg,local_evo)
    trial_rows=[]; edge_rows=[]; events=[]; trial=0; compile_trial=None; calls_after_compile=0; stop=False
    for epoch in range(1,int(task.training_epochs)+1):
        for pair_i,(a,b) in enumerate(task.training_pairs or [],1):
            s=0.0
            step_records=[]
            for rep in range(1,int(b)+1):
                trial+=1
                specs=[
                    {"layer":0,"neuron":0,"probability":min(1.0,max(0.0,s/float(task.partial_sum_scale))),"event_kind":"evidence"},
                    {"layer":0,"neuron":1,"probability":float(a)/float(task.operand_scale),"event_kind":"evidence"},
                    {"layer":0,"neuron":4,"probability":float(b)/float(task.operand_scale),"event_kind":"tonic"},
                ]
                m,cfg=run_cfg(mode,core,engine,base_cfg,gains,hw,local_evo,states,specs,outdir,f"train_t{trial:03d}_e{epoch}_a{a}_b{b}_r{rep}",shots)
                xh,ah,sh=decode_addition(m,cal,task)
                s=float(np.clip(sh,0.0,float(task.partial_sum_scale)))
                plastic=(agent!="frozen_plasticity")
                evs=[]
                if plastic:
                    engine.update_traces(traces,m,cfg,local_evo.trace)
                    evs=engine.update_candidate_states(local_evo.candidates,states,traces,local_evo.competition)
                    for ev in evs: events.append({"trial":trial,"epoch":epoch,"a":a,"b":b,"repeat":rep,**ev})
                if compile_trial is None and state_active(states,PRODUCT_EDGE_ID):
                    compile_trial=trial
                if compile_trial is not None: calls_after_compile=trial-compile_trial
                row={"trial":trial,"epoch":epoch,"pair_index":pair_i,"a":a,"b":b,"repeat_index":rep,
                     "partial_before":float(specs[0]["probability"])*float(task.partial_sum_scale),
                     "decoded_partial":xh,"decoded_addend":ah,"partial_after":s,
                     "P_N2":float(m[2]),"P_N3":float(m[3]),"P_N6":float(m[6]),
                     "P_G0":float(m[base_cfg.n_neurons+0]),"P_G1":float(m[base_cfg.n_neurons+1]),
                     "trace_G0":float(traces.get("G0",0)),"trace_G1":float(traces.get("G1",0)),
                     "GG_strength":state_strength(states,REQUIRED_GG_ID),"product_edge_strength":state_strength(states,PRODUCT_EDGE_ID),
                     "product_edge_active":state_active(states,PRODUCT_EDGE_ID),"plasticity_update":plastic}
                step_records.append(row); trial_rows.append(row)
                for c in local_evo.candidates:
                    st=states[c.id]
                    edge_rows.append({"trial":trial,"epoch":epoch,"candidate_id":c.id,"edge_type":c.type,
                                      "strength":float(st.strength),"active":bool(st.active),"driver":engine.driver_value(c,traces),
                                      "affinity":float(c.affinity),"generated_from":c.generated_from})
                target=task.early_stop_after_product_strength
                if target is not None and state_strength(states,PRODUCT_EDGE_ID)>=float(target) and calls_after_compile>=int(task.minimum_post_compile_consolidation_calls):
                    stop=True; break
            # Product is appended only to the output table after the neural procedure.
            if step_records:
                step_records[-1]["episode_true_product_posthoc"]=int(a)*int(b)
                step_records[-1]["episode_procedural_product"]=s
                step_records[-1]["episode_error"]=s-int(a)*int(b)
                step_records[-1]["target_used_for_plasticity"]=False
            if stop: break
        if stop: break
    trdf=pd.DataFrame(trial_rows); edf=pd.DataFrame(edge_rows); evdf=pd.DataFrame(events)
    trdf.to_csv(outdir/"procedural_repeated_addition_training.csv",index=False)
    edf.to_csv(outdir/"arithmetic_edge_evolution.csv",index=False)
    evdf.to_csv(outdir/"arithmetic_emergence_events.csv",index=False)
    return local_evo,traces,states,trdf,edf,evdf,compile_trial


def procedural_episode_metrics(training_df: pd.DataFrame):
    if training_df.empty or "episode_procedural_product" not in training_df:
        return {"R2":float("nan"),"RMSE":float("nan"),"MAE":float("nan"),"NRMSE_range":float("nan")}
    z=training_df.dropna(subset=["episode_procedural_product","episode_true_product_posthoc"])
    return metrics(z.episode_true_product_posthoc.to_numpy(float),z.episode_procedural_product.to_numpy(float)) if len(z) else {}


def resource_report(core,engine,base_cfg,evo):
    ids={c.id for c in evo.candidates}
    print(f"{VERSION}")
    print(f"Base circuit: {base_cfg.total_qubits} qubits, {base_cfg.layers} layer")
    print(f"Explicit candidate_edges in JSON: 0 expected")
    print(f"Auto-generated candidates: {len(evo.candidates)}")
    print(f"Required G-G bridge {REQUIRED_GG_ID}: {REQUIRED_GG_ID in ids}")
    print(f"Required product candidate {PRODUCT_EDGE_ID}: {PRODUCT_EDGE_ID in ids}")
    print("Tripartite candidates:")
    for c in evo.candidates:
        if c.type=="tripartite": print(f"  {c.id} affinity={c.affinity:.3f}")
    init=core.circuit_resource_estimate(base_cfg)
    allstates={c.id:engine.CandidateState(1.0,True) for c in evo.candidates}
    full=engine.materialize_dynamic_config(core,base_cfg,evo.candidates,allstates,[])
    print("Initial resources:",json.dumps(init,indent=2))
    print("All-candidates-active upper bound:",json.dumps(core.circuit_resource_estimate(full),indent=2))


def run_agent(mode,core,engine,base_cfg,gains,hw,evo,task,root,agent,seed):
    outdir=root if agent=="full" else root/agent
    outdir.mkdir(parents=True,exist_ok=True)
    shots=int(task.shots)
    initial_states=engine.initialize_states(evo)

    # 1. Calibrate and falsify/validate the pre-existing ADD primitive before any plasticity.
    cal=calibrate_addition(mode,core,engine,base_cfg,gains,hw,evo,initial_states,task,outdir,shots,"pre")
    add_pre,add_pre_metrics=addition_validation(mode,core,engine,base_cfg,gains,hw,evo,initial_states,task,cal,outdir,shots,"pre")

    probe_pairs=[]
    for p in (task.product_calibration_pairs or [])+(task.product_holdout_pairs or []):
        t=tuple(map(int,p))
        if t not in probe_pairs: probe_pairs.append(t)
    pre_prod=direct_product_probe(mode,core,engine,base_cfg,gains,hw,evo,initial_states,task,outdir,shots,"pre",probe_pairs)

    # 2. Repeated ADD procedure drives slow-trace/topology evolution. No product label enters updates.
    local_evo,traces,states,train_df,edge_df,event_df,compile_trial=train_repeated_addition(
        mode,core,engine,base_cfg,gains,hw,evo,task,cal,outdir,shots,agent=agent)

    # 3. Verify fixed ADD primitive remains readable after evolution using the same calibration.
    add_post,add_post_metrics=addition_validation(mode,core,engine,base_cfg,gains,hw,local_evo,states,task,cal,outdir,shots,"post")

    # 4. Direct one-call product probe after history-dependent topology change.
    post_prod=direct_product_probe(mode,core,engine,base_cfg,gains,hw,local_evo,states,task,outdir,shots,"post",probe_pairs)
    post_cal=post_prod.merge(pd.DataFrame(task.product_calibration_pairs,columns=["a","b"]),on=["a","b"],how="inner")
    if len(post_cal)<2 or float(post_cal.P_product.std())<EPS:
        dec_intercept=float("nan"); dec_slope=float("nan")
    else:
        dec_intercept,dec_slope=fit_posthoc_product_decoder(post_cal)

    hold_pairs=pd.DataFrame(task.product_holdout_pairs,columns=["a","b"])
    def hold(df): return df.merge(hold_pairs,on=["a","b"],how="inner")
    pre_hold=hold(pre_prod); post_hold=hold(post_prod)
    if math.isfinite(dec_intercept) and math.isfinite(dec_slope):
        pre_hold=apply_product_decoder(pre_hold,dec_intercept,dec_slope)
        post_hold=apply_product_decoder(post_hold,dec_intercept,dec_slope)
        pre_num=metrics(pre_hold.true_product.to_numpy(float),pre_hold.predicted_product.to_numpy(float))
        post_num=metrics(post_hold.true_product.to_numpy(float),post_hold.predicted_product.to_numpy(float))
    else:
        pre_num=post_num={"R2":float("nan"),"RMSE":float("nan"),"MAE":float("nan"),"NRMSE_range":float("nan")}

    # 5. Cheap QPU-friendly causal tests: same trained run, then remove learned topology.
    ablated_df=pd.DataFrame(); reset_df=pd.DataFrame(); ablated_num={}; reset_num={}
    if task.run_targeted_product_ablation:
        astates=clone_product_ablation(states)
        ablated_df=direct_product_probe(mode,core,engine,base_cfg,gains,hw,local_evo,astates,task,outdir,shots,"product_edge_ablation",task.product_holdout_pairs)
        if math.isfinite(dec_intercept):
            ablated_df=apply_product_decoder(ablated_df,dec_intercept,dec_slope)
            ablated_num=metrics(ablated_df.true_product.to_numpy(float),ablated_df.predicted_product.to_numpy(float))
    if task.run_topology_reset:
        rstates=clone_reset_states(engine,local_evo)
        reset_df=direct_product_probe(mode,core,engine,base_cfg,gains,hw,local_evo,rstates,task,outdir,shots,"topology_reset",task.product_holdout_pairs)
        if math.isfinite(dec_intercept):
            reset_df=apply_product_decoder(reset_df,dec_intercept,dec_slope)
            reset_num=metrics(reset_df.true_product.to_numpy(float),reset_df.predicted_product.to_numpy(float))

    all_prod=pd.concat([pre_prod,post_prod,ablated_df,reset_df],ignore_index=True,sort=False)
    all_prod.to_csv(outdir/"direct_product_probes.csv",index=False)
    pre_hold.to_csv(outdir/"pre_heldout_product_validation.csv",index=False)
    post_hold.to_csv(outdir/"post_heldout_product_validation.csv",index=False)
    if len(ablated_df): ablated_df.to_csv(outdir/"product_edge_ablation_validation.csv",index=False)
    if len(reset_df): reset_df.to_csv(outdir/"topology_reset_validation.csv",index=False)

    # Interaction regressions use raw measured input/glial populations and cannot be created by the arithmetic decoder.
    pre_int=fit_interaction_models(pre_prod); post_int=fit_interaction_models(post_prod)
    pre_d2=mixed_second_difference(pre_prod); post_d2=mixed_second_difference(post_prod)
    proc_met=procedural_episode_metrics(train_df)

    post_mean=float(post_hold.P_product.mean()) if len(post_hold) else float("nan")
    pre_mean=float(pre_hold.P_product.mean()) if len(pre_hold) else float("nan")
    abl_mean=float(ablated_df.P_product.mean()) if len(ablated_df) else float("nan")
    reset_mean=float(reset_df.P_product.mean()) if len(reset_df) else float("nan")

    # Bounded call-count comparison: repeated addition needs b fast-circuit calls;
    # the compiled route is queried once. This is a benchmark-level compression
    # measure, not an asymptotic computational-complexity claim.
    call_rows=[]
    for a,b in (task.product_holdout_pairs or []):
        call_rows.append({"a":int(a),"b":int(b),"true_product":int(a)*int(b),
                          "recurrent_addition_calls":int(b),"compiled_direct_calls":1,
                          "call_reduction_factor":float(b)})
    pd.DataFrame(call_rows).to_csv(outdir/"call_count_comparison.csv",index=False)

    is_qpu=mode=="qpu"
    add_thr=0.80 if is_qpu else 0.95
    prod_thr=0.50 if is_qpu else 0.80
    summary={
        "version":VERSION,"agent":agent,"mode":mode,"seed":seed,
        "addition_pre_R2":add_pre_metrics.get("R2"),"addition_post_R2":add_post_metrics.get("R2"),
        "procedural_repeated_addition_product_R2":proc_met.get("R2"),
        "required_GG_active":state_active(states,REQUIRED_GG_ID),"required_GG_strength":state_strength(states,REQUIRED_GG_ID),
        "product_edge_active":state_active(states,PRODUCT_EDGE_ID),"product_edge_strength":state_strength(states,PRODUCT_EDGE_ID),
        "product_compile_trial":compile_trial,
        "pre_interaction_delta_R2":pre_int["delta_R2"],"post_interaction_delta_R2":post_int["delta_R2"],
        "pre_beta_xg":pre_int["beta_xg"],"post_beta_xg":post_int["beta_xg"],
        "pre_mixed_second_difference":pre_d2,"post_mixed_second_difference":post_d2,
        "product_decoder_intercept":dec_intercept,"product_decoder_slope":dec_slope,
        "heldout_pre_product_R2":pre_num.get("R2"),"heldout_post_product_R2":post_num.get("R2"),
        "heldout_post_product_RMSE":post_num.get("RMSE"),"heldout_post_product_NRMSE_range":post_num.get("NRMSE_range"),
        "mean_raw_product_signal_pre":pre_mean,"mean_raw_product_signal_post":post_mean,
        "mean_raw_product_signal_targeted_ablation":abl_mean,"mean_raw_product_signal_topology_reset":reset_mean,
        "addition_primitive_status":"PASS" if add_pre_metrics.get("R2",-999)>=add_thr else "FAIL",
        "topology_compilation_status":"PASS" if state_active(states,PRODUCT_EDGE_ID) else "FAIL",
        "heldout_direct_product_status":"PASS" if post_num.get("R2",-999)>=prod_thr else "FAIL",
        "interaction_emergence_status":"PASS" if (post_int["delta_R2"] > (pre_int["delta_R2"] if math.isfinite(pre_int["delta_R2"]) else 0.0) + 0.05 and post_int["beta_xg"]>0) else "FAIL",
        "targeted_ablation_status":(
            "NOT_APPLICABLE" if not state_active(states,PRODUCT_EDGE_ID) else
            ("PASS" if (math.isfinite(abl_mean) and math.isfinite(post_mean) and post_mean>EPS and abl_mean < 0.5*post_mean) else ("NOT_RUN" if not len(ablated_df) else "FAIL"))
        ),
        "topology_reset_status":(
            "NOT_APPLICABLE" if not state_active(states,PRODUCT_EDGE_ID) else
            ("PASS" if (math.isfinite(reset_mean) and math.isfinite(post_mean) and post_mean>EPS and reset_mean < 0.5*post_mean) else ("NOT_RUN" if not len(reset_df) else "FAIL"))
        ),
    }
    if str(task.validation_profile).lower()=="diagnostic":
        for k in ("topology_compilation_status","heldout_direct_product_status","interaction_emergence_status","targeted_ablation_status","topology_reset_status"):
            summary[k]="DIAGNOSTIC_ONLY"
    pd.DataFrame([summary]).to_csv(outdir/"arithmetic_validation_summary.csv",index=False)
    manifest={
        "version":VERSION,"scientific_scope":"Population-level hybrid computational hypothesis test; not a literal neural arithmetic map and not evidence of microscopic quantum brain computation.",
        "node_map":NODE_MAP,"task":asdict(task),"mode":mode,"agent":agent,
        "training_target_used_for_plasticity":False,
        "plasticity_inputs":"Measured V7.1 populations -> slow traces -> spatial candidate competition/hysteresis only",
        "candidate_edges_explicitly_supplied":0,
        "required_generated_candidates":{"GG":REQUIRED_GG_ID,"product":PRODUCT_EDGE_ID},
        "qpu_note":"Sequential training calls are intentionally not parallelized. The circuit is 14 qubits and one layer to limit hardware mixing. One-call direct product is a bounded benchmark call-count reduction, not an asymptotic speedup claim.",
        "posthoc_decoder_note":"The scalar product decoder is affine and fit only after topology training on calibration operands <=3; held-out tests contain operand 4. It never feeds plasticity.",
    }
    (outdir/"arithmetic_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    return summary


def main() -> int:
    ap=argparse.ArgumentParser(description="V7.1 Sample1B literal arithmetic-compilation validation")
    ap.add_argument("--core",default=DEFAULT_CORE)
    ap.add_argument("--engine",default=DEFAULT_ENGINE)
    ap.add_argument("--project",default=DEFAULT_PROJECT)
    ap.add_argument("--evolution-config",default=DEFAULT_EVOLUTION)
    ap.add_argument("--task-config",default=DEFAULT_TASK)
    ap.add_argument("--mode",choices=["resource","surrogate","local","qpu"],default="surrogate")
    ap.add_argument("--output-dir",default=None)
    ap.add_argument("--agents",default="full",help="comma-separated: full,frozen_plasticity,no_tripartite_growth")
    ap.add_argument("--n-seeds",type=int,default=1)
    ap.add_argument("--seed-offset",type=int,default=0)
    args=ap.parse_args()

    here=Path(__file__).resolve().parent
    core=load_module("sample1b_core",resolve(here,args.core))
    engine=load_module("sample1b_engine",resolve(here,args.engine))
    bio,base_cfg,hw,_=core.project_from_json(resolve(here,args.project))
    core.validate_biological_input(bio); core.validate_network_config(base_cfg)
    gains=core.scores_to_layer_gains(core.compute_mechanistic_scores(bio),bio.hardware_gain)
    evo=engine.load_evolution_config(resolve(here,args.evolution_config)); engine.finalize_spatial_candidates(base_cfg,evo); engine.validate_evolution_config(base_cfg,evo)
    task=load_task(resolve(here,args.task_config)); validate_task(task)
    hw.shots=int(task.shots)

    if args.mode=="resource":
        resource_report(core,engine,base_cfg,evo); return 0

    agents=[x.strip() for x in str(args.agents).split(",") if x.strip()]
    allowed={"full","frozen_plasticity","no_tripartite_growth"}
    bad=[x for x in agents if x not in allowed]
    if bad: raise SystemExit(f"Unknown agents {bad}; allowed={sorted(allowed)}")

    root=Path(args.output_dir) if args.output_dir else here/f"sample1b_{args.mode}_output"
    root.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for si in range(int(args.n_seeds)):
        seed=int(task.seed)+int(args.seed_offset)+si
        for agent in agents:
            adir=root
            if int(args.n_seeds)>1:
                adir=root/f"seed_{seed}"
            print(f"\n=== Sample1B {args.mode}: agent={agent}, seed={seed} ===")
            summaries.append(run_agent(args.mode,core,engine,base_cfg,gains,hw,evo,task,adir,agent,seed))
    pd.DataFrame(summaries).to_csv(root/"arithmetic_agent_summary.csv",index=False)
    print("\nSaved:",root.resolve())
    for s in summaries:
        print(f"[{s['agent']}] ADD={s['addition_primitive_status']} topology={s['topology_compilation_status']} heldout_product={s['heldout_direct_product_status']} interaction={s['interaction_emergence_status']} reset={s['topology_reset_status']}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
