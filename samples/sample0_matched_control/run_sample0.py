"""Sample0: matched neuroglial fast-layer mechanistic experiment.

Example: python run_sample0.py --seeds 20 --shots 3000 --output-dir sample0_results
All execution is local. Original V7.1 slow updates are called without edits.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from classical_matched_fast_layer import simulate, total_variation
from v71_bridge import ROOT, ORIGINAL, core, evolution as ev, compile_tape

VERSION = "Sample0-V7.1-matched-1.0"


@dataclass(frozen=True)
class Condition:
    name: str
    backend: str
    mixer_gain: float = 0.0
    phase_scale: float = 1.0
    zz_scale: float = 1.0
    frozen: bool = False


def conditions(mixer_gain):
    return [
        Condition("q_original", "quantum"), Condition("c_original", "classical"),
        Condition("q_original_no_phase", "quantum", phase_scale=0),
        Condition("q_original_no_zz", "quantum", zz_scale=0),
        Condition("q_context", "quantum", mixer_gain), Condition("c_context", "classical", mixer_gain),
        Condition("q_context_no_phase", "quantum", mixer_gain, phase_scale=0),
        Condition("q_context_no_zz", "quantum", mixer_gain, zz_scale=0),
        Condition("q_original_frozen", "quantum", frozen=True),
        Condition("c_original_frozen", "classical", frozen=True),
    ]


def dump(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=256)
def cached_simulate(tape, n, backend, limit):
    return simulate(tape, n, backend, limit)


def evaluate(cfg, gains, condition, limit=20):
    tape = compile_tape(cfg, gains, condition.mixer_gain, condition.phase_scale, condition.zz_scale)
    return cached_simulate(tuple(tape), cfg.total_qubits, condition.backend, limit)


class CountLog:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def record(self, result, counts, stream, index, label):
        if counts is None:
            payload = {"mode": "exact", "probabilities": [[int(i), float(result.probability[i])]
                       for i in np.flatnonzero(result.probability > 1e-15)]}
        else:
            payload = {"mode": "sampled", "counts": [[int(i), int(counts[i])] for i in np.flatnonzero(counts)]}
        row = {"label": label, "stream": stream, "index": index,
               "circuit_sha256": result.circuit_sha256,
               "logical_width": result.logical_width, "active_qubits": result.active_qubits, **payload}
        self.rows.append(json.dumps(row, separators=(",", ":")) + "\n")

    def close(self):
        # Commit a complete, deterministic archive atomically, then verify the
        # actual bytes before allowing this seed/condition to be marked done.
        payload = "".join(self.rows).encode("utf-8")
        packed = gzip.compress(payload, mtime=0)
        temp = self.path.with_name(self.path.name + ".tmp")
        temp.write_bytes(packed)
        temp.replace(self.path)
        if gzip.decompress(self.path.read_bytes()) != payload:
            raise IOError(f"Raw count archive verification failed: {self.path}")


def grid_values(probe, grid_size):
    xs = list(map(float, probe["x"].get("values", [0, .25, .5, .75, 1])))
    ys = list(map(float, probe["y"].get("values", [0, .25, .5, .75, 1])))
    if grid_size:
        xs = np.linspace(min(xs), max(xs), grid_size).tolist()
        ys = np.linspace(min(ys), max(ys), grid_size).tolist()
    if len(set(xs)) < 3 or len(set(ys)) < 3:
        raise ValueError("At least three distinct input values per probe axis are required")
    xs, ys = sorted(set(xs)), sorted(set(ys))
    hx = ((np.array(xs[:-1]) + xs[1:]) / 2).tolist()
    hy = ((np.array(ys[:-1]) + ys[1:]) / 2).tolist()
    return xs, ys, hx, hy


def input_specs(probe, x, y):
    return [{"layer": int(spec.get("layer", 0)), "neuron": int(spec["neuron"]),
             "probability": value, "event_kind": "evidence"}
            for spec, value in [(probe["x"], x), (probe["y"], y)]]


def probe_surface(base, gains, evo, states, probe, cond, seed, args, stage, counts):
    xs, ys, hx, hy = grid_values(probe, args.grid_size)
    rows = []
    idx = 0
    for split, ax, ay in [("fit", xs, ys), ("holdout", hx, hy)]:
        for x in ax:
            for y in ay:
                events = ev.build_events(core, input_specs(probe, x, y), gains)
                cfg = ev.materialize_dynamic_config(core, base, evo.candidates, states, events)
                result = evaluate(cfg, gains, cond, args.max_active_qubits)
                exact = result.marginals()
                stream = 2 if stage == "pre" else 3
                sampled, cnt = result.sample(args.shots, seed, stream, idx)
                counts.record(result, cnt, stream, idx, f"{stage}_{split}_{x:g}_{y:g}")
                xn, yn = int(probe["x"]["neuron"]), int(probe["y"]["neuron"])
                gn, out = base.n_neurons + int(probe["context_glia"]), int(probe["output_neuron"])
                rows.append(dict(seed=seed, condition=cond.name, stage=stage, split=split, x=x, y=y,
                                 P_output=float(exact[out]), P_output_sampled=float(sampled[out]),
                                 P_x=float(exact[xn]), P_y_driver=float(exact[yn]),
                                 P_context_glia=float(exact[gn]), circuit_sha256=result.circuit_sha256))
                idx += 1
    return pd.DataFrame(rows)


def regression(surface):
    """Fit requested-input surfaces; holdout measures interpolation, not task success."""
    fit, hold = surface[surface.split == "fit"], surface[surface.split == "holdout"]
    out = []
    for name in ("additive", "interaction"):
        def design(df):
            cols = [np.ones(len(df)), df.x, df.y]
            if name == "interaction":
                cols.append(df.x * df.y)
            return np.column_stack(cols)
        x, h = design(fit), design(hold)
        y, yh = fit.P_output.to_numpy(), hold.P_output.to_numpy()
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        residual = y - x @ beta
        sst = np.sum((y-y.mean())**2)
        out.append(dict(model=name, R2=float(1 - residual @ residual/sst) if sst > 1e-12 else np.nan,
                        RMSE=float(np.sqrt(np.mean(residual**2))),
                        holdout_RMSE=float(np.sqrt(np.mean((yh-h@beta)**2))),
                        intercept=beta[0], beta_x=beta[1], beta_y=beta[2],
                        beta_xy=beta[3] if len(beta) == 4 else 0.0, rank=int(rank)))
    return pd.DataFrame(out)


def mixed_difference(df):
    fit = df[df.split == "fit"]
    xmin, xmax, ymin, ymax = fit.x.min(), fit.x.max(), fit.y.min(), fit.y.max()
    def at(x, y):
        return float(fit[(fit.x == x) & (fit.y == y)].P_output.iloc[0])
    return at(xmax, ymax) - at(xmax, ymin) - at(xmin, ymax) + at(xmin, ymin)


def kernel_replay(base, gains, evo, states, probe, reference, seed, args):
    """Lock topology + weights + input; isolate immediate fast-kernel effects."""
    rows, resources = [], []
    events = ev.build_events(core, evo.training_phases[-1]["neural_inputs"], gains)
    stimuli = [("training", events)] + [(f"probe_{x:g}_{y:g}", ev.build_events(core, input_specs(probe, x, y), gains))
                                       for x, y in [(0, 0), (.5, .5), (1, 1)]]
    for label, inputs in stimuli:
        cfg = ev.materialize_dynamic_config(core, base, evo.candidates, states, inputs)
        for gain in (0.0, args.mixer_gain):
            q = evaluate(cfg, gains, Condition("q", "quantum", gain), args.max_active_qubits)
            alternatives = [("matched_classical", Condition("c", "classical", gain)),
                            ("no_explicit_phase", Condition("q_no_phase", "quantum", gain, phase_scale=0)),
                            ("no_zz", Condition("q_no_zz", "quantum", gain, zz_scale=0))]
            for comparison, cond in alternatives:
                alt = evaluate(cfg, gains, cond, args.max_active_qubits)
                delta = alt.marginals() - q.marginals()
                for node in range(base.total_qubits):
                    rows.append(dict(seed=seed, topology_source=reference, input=label, mixer_gain=gain,
                                     comparison=comparison, node=f"N{node}" if node < base.n_neurons else f"G{node-base.n_neurons}",
                                     P_quantum=float(q.marginals()[node]), P_alternative=float(alt.marginals()[node]),
                                     delta_probability=float(delta[node]), joint_TV=total_variation(q, alt)))
            tape = compile_tape(cfg, gains, mixer_gain=gain)
            resources.append(dict(seed=seed, topology_source=reference, input=label, mixer_gain=gain,
                                  logical_qubits=base.total_qubits, active_simulation_qubits=len(q.active_qubits),
                                  semantic_gates=len(tape), CRY=sum(g.name == "CRY" for g in tape),
                                  CCRY=sum(g.name == "CCRY" for g in tape), XY=sum(g.name == "XY" for g in tape),
                                  ZZ=sum(g.name == "ZZ" for g in tape), context_mixer_gates=sum(g.tag == "context_mixer" for g in tape)))
    return pd.DataFrame(rows), pd.DataFrame(resources)


def run_one(base, gains, evo, probe, cond, seed, args, folder):
    folder.mkdir(parents=True, exist_ok=True)
    log = CountLog(folder / "raw_joint_counts.jsonl.gz")
    traces, states = ev.initialize_traces(base, evo), ev.initialize_states(evo)
    before = probe_surface(base, gains, evo, states, probe, cond, seed, args, "pre", log)
    nodes, edges, events_out, trial_rows = [], [], [], []
    trial = 0
    for phase in evo.training_phases:
        for rep in range(int(phase.get("repetitions", 1))):
            if args.max_training_trials and trial >= args.max_training_trials:
                break
            trial += 1
            events = ev.build_events(core, phase.get("neural_inputs", []), gains)
            cfg = ev.materialize_dynamic_config(core, base, evo.candidates, states, events)
            result = evaluate(cfg, gains, cond, args.max_active_qubits)
            marg, cnt = result.sample(args.shots, seed, 1, trial)
            log.record(result, cnt, 1, trial, f"training_{trial}")
            active_before = sum(st.active for st in states.values())
            plasticity = bool(phase.get("plasticity_update", True)) and not cond.frozen
            if plasticity:
                # These are the original, unedited V7.1 functions.
                ev.update_traces(traces, marg, cfg, evo.trace)
                formed = ev.update_candidate_states(evo.candidates, states, traces, evo.competition)
                events_out.extend(dict(seed=seed, condition=cond.name, trial=trial, phase=phase["name"], **x) for x in formed)
            nodes.extend(ev.trial_rows_from_marg(trial, phase["name"], rep+1, cfg, marg, traces, evo.candidates, states))
            edges.extend(ev.edge_rows(trial, phase["name"], evo.candidates, states, traces))
            trial_rows.append(dict(seed=seed, condition=cond.name, trial=trial, phase=phase["name"],
                                   plasticity_update=plasticity, active_edges_before=active_before,
                                   active_edges_after=sum(st.active for st in states.values()),
                                   circuit_sha256=result.circuit_sha256))
    after = probe_surface(base, gains, evo, states, probe, cond, seed, args, "post", log)
    log.close()
    surface = pd.concat([before, after], ignore_index=True)
    surface.to_csv(folder / "probe_surface.csv", index=False)
    regressions = pd.concat([regression(before).assign(stage="pre"), regression(after).assign(stage="post")], ignore_index=True)
    regressions.to_csv(folder / "surface_regression.csv", index=False)
    pd.DataFrame(nodes).to_csv(folder / "node_activity_and_slow_trace.csv", index=False)
    pd.DataFrame(edges).to_csv(folder / "edge_evolution.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(folder / "trial_summary.csv", index=False)
    pd.DataFrame(events_out, columns=["seed", "condition", "trial", "phase", "candidate_id", "edge_type", "event", "driver",
                                           "affinity", "competition_win", "strength_before", "strength_after", "generated_from"]).to_csv(folder / "emergence_events.csv", index=False)
    final = pd.DataFrame([dict(candidate_id=c.id, edge_type=c.type, **asdict(states[c.id])) for c in evo.candidates])
    final.to_csv(folder / "final_candidate_states.csv", index=False)
    replay, resource = kernel_replay(base, gains, evo, states, probe, cond.name, seed, args) if cond.name in ("q_original", "c_original", "q_context", "c_context") else (pd.DataFrame(), pd.DataFrame())
    if not replay.empty:
        replay.to_csv(folder / "kernel_replay.csv", index=False)
        resource.to_csv(folder / "circuit_resources.csv", index=False)
    summary = dict(seed=seed, condition=cond.name, backend=cond.backend, mixer_gain=cond.mixer_gain,
                   phase_scale=cond.phase_scale, zz_scale=cond.zz_scale, frozen=cond.frozen, training_trials=trial,
                   final_active_edges=int(final.active.sum()), final_strength_sum=float(final.strength.sum()),
                   pre_mixed_difference=mixed_difference(before), post_mixed_difference=mixed_difference(after),
                   final_output_mean=float(after[after.split == "fit"].P_output.mean()))
    summary["change_mixed_difference"] = summary["post_mixed_difference"] - summary["pre_mixed_difference"]
    reg = regressions[regressions.stage == "post"].set_index("model")
    summary["post_delta_R2"] = reg.loc["interaction", "R2"] - reg.loc["additive", "R2"]
    summary["interaction_holdout_RMSE"] = reg.loc["interaction", "holdout_RMSE"]
    for typ in ("glia_glia", "glia_to_neuron", "tripartite"):
        times = [e["trial"] for e in events_out if e["event"] == "formed" and e["edge_type"] == typ]
        summary[f"emerged_{typ}"] = bool(times)
        summary[f"T_emerge_{typ}"] = min(times) if times else np.nan
        summary[f"T_observed_{typ}"] = min(times) if times else trial
        summary[f"censored_{typ}"] = not bool(times)
    pd.DataFrame([summary]).to_csv(folder / "summary.csv", index=False)
    dump(folder / "complete.json", {"version": VERSION, "seed": seed, "condition": asdict(cond), "final_traces": traces,
                                    "raw_query_count": len(log.rows), "raw_counts_sha256": sha(log.path)})
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=Path, default=ORIGINAL / "V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json")
    ap.add_argument("--evolution-config", type=Path, default=ORIGINAL / "V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json")
    ap.add_argument("--output-dir", type=Path, default=Path("sample0_results"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=20260910)
    ap.add_argument("--shots", type=int, default=3000, help="Joint samples per trial/probe; 0=exact, requires --seeds 1")
    ap.add_argument("--mixer-gain", type=float, default=-.35, help="Prespecified optional extension coefficient; legacy gain remains zero")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--max-training-trials", type=int, default=None)
    ap.add_argument("--grid-size", type=int, default=None)
    ap.add_argument("--max-active-qubits", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-analyze", action="store_true")
    args = ap.parse_args()
    if args.seeds < 1 or args.seed_start < 0 or args.shots < 0 or (args.shots == 0 and args.seeds != 1):
        ap.error("Use positive seeds, nonnegative seed-start/shots; exact shots=0 requires seeds=1")
    if not -1 <= args.mixer_gain <= 1 or args.mixer_gain == 0:
        ap.error("Use a nonzero mixer gain in [-1,1] so the extension arms are identifiable")
    if args.grid_size is not None and args.grid_size < 3:
        ap.error("grid-size must be >=3")
    if args.max_training_trials is not None and args.max_training_trials < 1:
        ap.error("max-training-trials must be positive")
    cs = conditions(args.mixer_gain)
    if args.conditions:
        known = {c.name: c for c in cs}
        if set(args.conditions) - known.keys():
            ap.error(f"Unknown conditions; choose from {list(known)}")
        cs = [known[name] for name in dict.fromkeys(args.conditions)]
    bio, base, hw, _ = core.project_from_json(args.project)
    core.validate_biological_input(bio)
    gains = core.scores_to_layer_gains(core.compute_mechanistic_scores(bio), bio.hardware_gain)
    evo = ev.load_evolution_config(args.evolution_config)
    ev.finalize_spatial_candidates(base, evo)
    ev.validate_evolution_config(base, evo)
    probes = [p for p in evo.probes if p.get("type") == "product_grid"]
    if len(probes) != 1:
        ap.error("Sample0 requires exactly one product_grid probe in this evolution file")
    probe = probes[0]
    grid_values(probe, args.grid_size)
    full_trials = sum(int(p.get("repetitions", 1)) for p in evo.training_phases)
    spec = dict(version=VERSION, source_hashes={p.name: sha(p) for p in [Path(__file__), ROOT / "v71_bridge.py",
                ROOT / "v71_context_builder.py", ROOT / "classical_matched_fast_layer.py", args.project, args.evolution_config,
                ORIGINAL / "V7_1_neuroglial_gate_qpu.py", ORIGINAL / "V7_1_SPATIAL_EVOLVING_NETWORK.py"]},
                seeds=list(range(args.seed_start, args.seed_start + args.seeds)), shots=args.shots,
                conditions=[asdict(c) for c in cs], max_training_trials=args.max_training_trials,
                full_training_trials=full_trials, grid_size=args.grid_size, max_active_qubits=args.max_active_qubits,
                project=json.loads(args.project.read_text(encoding="utf-8")),
                evolution=json.loads(args.evolution_config.read_text(encoding="utf-8")))
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    out = args.output_dir.resolve()
    manifest = out / "run_manifest.json"
    if out.exists() and any(out.iterdir()):
        if not args.resume or not manifest.exists():
            ap.error("Output directory is nonempty; use a new directory, or --resume for the identical protocol")
        if json.loads(manifest.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
            ap.error("Resume rejected: source code/configuration/seed protocol changed")
    out.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        dump(manifest, dict(fingerprint=fingerprint, spec=spec, created_utc=datetime.now(timezone.utc).isoformat(),
                            python=sys.version, numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
                            experiment_scope="Finite matched model class; no physical quantum necessity or speedup test.",
                            seed_scope="Finite-shot Monte Carlo uncertainty; fixed anatomy and fixed input schedule.",
                            endpoint_readout="Exact post-training probe expectation; sampled probe values also exported.",
                            execution="Local CPU; no QPU jobs submitted"))
    ev.save_candidate_table(evo, out)
    dump(out / "layer_gains.json", asdict(gains))
    summaries = []
    started = time.monotonic()
    for seed in spec["seeds"]:
        for cond in cs:
            folder = out / f"seed_{seed}" / cond.name
            if args.resume and (folder / "complete.json").exists():
                complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
                archive = folder / "raw_joint_counts.jsonl.gz"
                if archive.exists() and sha(archive) == complete.get("raw_counts_sha256"):
                    summaries.append(pd.read_csv(folder / "summary.csv").iloc[0].to_dict())
                    print(f"RESUME {seed} {cond.name}", flush=True)
                    continue
                print(f"REPAIR missing/corrupt raw counts: {seed} {cond.name}", flush=True)
            summary = run_one(base, gains, evo, probe, cond, seed, args, folder)
            summaries.append(summary)
            print(f"{len(summaries)}/{len(cs)*args.seeds} seed={seed} {cond.name}: edges={summary['final_active_edges']}, "
                  f"D2={summary['post_mixed_difference']:.4f}; elapsed={time.monotonic()-started:.1f}s", flush=True)
    pd.DataFrame(summaries).to_csv(out / "seed_summary.csv", index=False)
    if not args.no_analyze:
        from analyze_sample0 import analyze
        analyze(out)
    print(f"DONE: {out}", flush=True)


if __name__ == "__main__":
    main()
