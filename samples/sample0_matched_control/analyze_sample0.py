"""Aggregate paired seeds; export every plot's data, SVG/PNG and paired statistics."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"q_original": "#0072B2", "c_original": "#D55E00", "q_context": "#009E73", "c_context": "#CC79A7"}
SHORT = {"q_original": "Original Q", "c_original": "Original C", "q_context": "Context Q", "c_context": "Context C"}
PAIRS = [("q_original", "c_original"), ("q_context", "c_context"),
         ("q_original", "q_original_no_phase"), ("q_original", "q_original_no_zz"),
         ("q_context", "q_context_no_phase"), ("q_context", "q_context_no_zz"),
         ("q_original", "q_original_frozen"), ("c_original", "c_original_frozen")]


def collect(out, filename):
    rows = []
    for p in sorted(out.glob(f"seed_*/*/{filename}")):
        df = pd.read_csv(p)
        if "seed" not in df:
            df["seed"] = int(p.parent.parent.name.split("_")[1])
        if "condition" not in df:
            df["condition"] = p.parent.name
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def mean_ci(values, rng):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan, np.nan, np.nan, 0
    mean = x.mean()
    if len(x) == 1:
        return mean, np.nan, mean, mean, 1
    bs = x[rng.integers(0, len(x), size=(5000, len(x)))].mean(axis=1)
    lo, hi = np.quantile(bs, [.025, .975])
    return mean, x.std(ddof=1), lo, hi, len(x)


def paired_statistics(summary):
    rng = np.random.default_rng(85021)
    metrics = ["final_active_edges", "final_strength_sum", "post_mixed_difference", "change_mixed_difference",
               "post_delta_R2", "interaction_holdout_RMSE", "final_output_mean"]
    for typ in ("glia_glia", "glia_to_neuron", "tripartite"):
        metrics.extend([f"emerged_{typ}", f"T_observed_{typ}"])
    rows = []
    for a, b in PAIRS:
        left, right = summary[summary.condition == a], summary[summary.condition == b]
        merged = left.merge(right, on="seed", suffixes=("_a", "_b"), validate="one_to_one")
        if merged.empty:
            continue
        for metric in metrics:
            xa, xb = merged[f"{metric}_a"].to_numpy(float), merged[f"{metric}_b"].to_numpy(float)
            good = np.isfinite(xa) & np.isfinite(xb)
            d = xa[good]-xb[good]
            mean, sd, lo, hi, n = mean_ci(d, rng)
            if n < 2:
                pv = np.nan
            elif np.all(d == 0):
                pv = 1.0
            else:
                signs = rng.choice([-1, 1], size=(19999, n))
                pv = (1 + np.sum(abs((signs*d).mean(axis=1)) >= abs(d.mean()) - 1e-14)) / 20000
            rows.append(dict(condition_a=a, condition_b=b, metric=metric, n_paired=n,
                             mean_a=float(xa[good].mean()) if n else np.nan,
                             mean_b=float(xb[good].mean()) if n else np.nan,
                             mean_difference_a_minus_b=mean, difference_sd=sd, CI95_low=lo, CI95_high=hi,
                             paired_dz=mean/sd if np.isfinite(sd) and sd > 1e-14 else np.nan,
                             permutation_p=pv, inference="paired finite-shot simulation seeds"))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # One declared family across all available endpoint/contrast tests.
    finite = result.permutation_p.dropna().sort_values()
    adjusted = np.maximum.accumulate(np.minimum(1, finite.to_numpy() * np.arange(len(finite), 0, -1)))
    result["holm_p"] = np.nan
    result.loc[finite.index, "holm_p"] = adjusted
    return result


def graph_comparison(states, surfaces):
    rows = []
    for qa, ca in [("q_original", "c_original"), ("q_context", "c_context")]:
        sa, sb = states[states.condition == qa], states[states.condition == ca]
        for seed in sorted(set(sa.seed) & set(sb.seed)):
            a, b = sa[sa.seed == seed], sb[sb.seed == seed]
            aset, bset = set(a[a.active].candidate_id), set(b[b.active].candidate_id)
            union = aset | bset
            merged = a.merge(b, on="candidate_id", suffixes=("_q", "_c"), validate="one_to_one")
            pa = surfaces[(surfaces.condition == qa) & (surfaces.seed == seed) & (surfaces.stage == "post")]
            pb = surfaces[(surfaces.condition == ca) & (surfaces.seed == seed) & (surfaces.stage == "post")]
            d = pa.merge(pb, on=["split", "x", "y"], suffixes=("_q", "_c"), validate="one_to_one")
            delta = d.P_output_q - d.P_output_c
            rows.append(dict(seed=seed, pair="original" if qa == "q_original" else "context",
                             graph_jaccard=len(aset&bset)/len(union) if union else 1.0,
                             strength_mean_absolute_difference=float(abs(merged.strength_q-merged.strength_c).mean()),
                             surface_RMSE=float(np.sqrt(np.mean(delta**2))),
                             surface_max_absolute_difference=float(abs(delta).max()),
                             surface_MAE=float(abs(delta).mean())))
    return pd.DataFrame(rows, columns=["seed", "pair", "graph_jaccard", "strength_mean_absolute_difference",
                                       "surface_RMSE", "surface_max_absolute_difference", "surface_MAE"])


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.dpi": 300,
                         "axes.linewidth": .7, "lines.linewidth": 1.7, "legend.frameon": False})


def save_figure(fig, folder, stem):
    fig.savefig(folder / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def summarize_curves(trials):
    rows, rng = [], np.random.default_rng(9184)
    for (condition, trial), group in trials.groupby(["condition", "trial"], sort=True):
        mean, sd, lo, hi, n = mean_ci(group.active_edges_after, rng)
        rows.append(dict(condition=condition, trial=trial, mean=mean, SD=sd, CI95_low=lo, CI95_high=hi, n=n))
    return pd.DataFrame(rows)


def dotpanel(ax, df, xcol, ycol, order, ylabel, labels=None):
    rng = np.random.default_rng(571)
    for i, cat in enumerate(order):
        vals = df[df[xcol] == cat][ycol].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        color = COLORS.get(cat, "#666666")
        if len(vals):
            ax.scatter(i+rng.uniform(-.10, .10, len(vals)), vals, s=16, c=color, alpha=.65, edgecolors="none")
            ax.plot([i-.19, i+.19], [vals.mean()]*2, color=color, linewidth=2.2)
        else:
            ax.text(i, .03, "NA", transform=ax.get_xaxis_transform(), ha="center", color="#777777")
    ax.set_xticks(range(len(order)), labels or [SHORT.get(x, x) for x in order], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-.5, len(order)-.5)


def make_figures(out, summary, curves, graphs, surfaces, replay, motif, dephase):
    folder = out / "figures"
    data = out / "figure_data"
    folder.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.6), constrained_layout=True)
    for idx, names in enumerate([("q_original", "c_original"), ("q_context", "c_context")]):
        ax = axes.flat[idx]
        sub = curves[curves.condition.isin(names)]
        sub.to_csv(data / f"Fig0_{'AB'[idx]}_edge_trajectories.csv", index=False)
        for name in names:
            d = sub[sub.condition == name]
            ax.plot(d.trial, d["mean"], color=COLORS[name], label=SHORT[name])
            ax.fill_between(d.trial, d.CI95_low, d.CI95_high, color=COLORS[name], alpha=.16, linewidth=0)
        ax.set(xlabel="Training trial", ylabel="Active candidate edges", title=["A  Original fast layer", "B  Optional context feedback"][idx])
        ax.legend(fontsize=8, loc="lower right")
    order = [x for x in SHORT if x in set(summary.condition)]
    d = summary[summary.condition.isin(order)][["seed", "condition", "post_mixed_difference"]]
    d.to_csv(data / "Fig0_C_mixed_difference.csv", index=False)
    dotpanel(axes[0, 2], d, "condition", "post_mixed_difference", order, "Post-training mixed difference")
    axes[0, 2].set_title("C  Input interaction")
    axes[0, 2].set_ylim(bottom=0)
    graphs.to_csv(data / "Fig0_D_topology_agreement.csv", index=False)
    dotpanel(axes[1, 0], graphs, "pair", "graph_jaccard", ["original", "context"], "Final graph Jaccard index", ["Original Q/C", "Context Q/C"])
    axes[1, 0].set(ylim=(-.04, 1.07), title="D  Paired topology agreement")
    sub = motif[motif.beta.isin([0.0, -.35])]
    sub.to_csv(data / "Fig0_E_local_phase_response.csv", index=False)
    for beta, color, label in [(0.0, "#0072B2", "No late feedback"), (-.35, "#009E73", "Late feedback Q")]:
        d = sub[sub.beta == beta]
        axes[1, 1].plot(d.phase_rad, d.quantum, color=color, label=label)
    d = sub[sub.beta == -.35]
    axes[1, 1].plot(d.phase_rad, d.primitive_classical, color="#D55E00", linestyle="--", label="Late feedback C")
    axes[1, 1].set(xlabel="Context phase (rad)", ylabel="P(output)", title="E  Local motif diagnostic")
    axes[1, 1].legend(fontsize=8)
    ax = axes[1, 2]
    if not replay.empty:
        d = replay[(replay.topology_source == "q_original") & (replay.input == "training") &
                   (replay.node.str.startswith("N")) & (replay.comparison != "matched_classical")]
        d = d.groupby(["seed", "mixer_gain", "comparison"]).delta_probability.apply(lambda x: abs(x).max()).reset_index(name="max_neuronal_change")
        d["group"] = np.where(d.mixer_gain == 0, "Old / ", "Context / ") + np.where(d.comparison == "no_zz", "ZZ", "phase")
        d.to_csv(data / "Fig0_F_phase_ablation.csv", index=False)
        order = [x for x in ["Old / phase", "Old / ZZ", "Context / phase", "Context / ZZ"] if x in set(d.group)]
        dotpanel(ax, d, "group", "max_neuronal_change", order, "Maximum neuronal probability change")
    ax.set_title("F  Fixed learned topology: phase ablation")
    ax.set_ylim(bottom=0)
    save_figure(fig, folder, "Fig0_matched_control")

    # All post-training input surfaces, including an explicit Q-C difference.
    post = surfaces[(surfaces.stage == "post") & (surfaces.split == "fit")]
    mean = post.groupby(["condition", "x", "y"], as_index=False).P_output.mean()
    probability_max = max(.01, float(mean.P_output.max()))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), constrained_layout=True)
    for row, (q, c) in enumerate([("q_original", "c_original"), ("q_context", "c_context")]):
        a = mean[mean.condition == q]
        b = mean[mean.condition == c]
        for col, df in enumerate([a, b]):
            if df.empty:
                continue
            grid = df.pivot(index="y", columns="x", values="P_output").sort_index()
            im = axes[row, col].imshow(grid, origin="lower", aspect="equal", vmin=0, vmax=probability_max, cmap="viridis")
            axes[row, col].set_xticks(range(len(grid.columns)), [f"{x:g}" for x in grid.columns])
            axes[row, col].set_yticks(range(len(grid.index)), [f"{y:g}" for y in grid.index])
            axes[row, col].set(title=SHORT[q if col == 0 else c], xlabel="Requested x", ylabel="Requested y")
            fig.colorbar(im, ax=axes[row, col], label="Mean P(output)", shrink=.8)
        d = a.merge(b, on=["x", "y"], suffixes=("_q", "_c"))
        if not d.empty:
            d["difference_q_minus_c"] = d.P_output_q-d.P_output_c
            grid = d.pivot(index="y", columns="x", values="difference_q_minus_c").sort_index()
            lim = max(.01, abs(grid.to_numpy()).max())
            im = axes[row, 2].imshow(grid, origin="lower", vmin=-lim, vmax=lim, cmap="RdBu_r")
            axes[row, 2].set_xticks(range(len(grid.columns)), [f"{x:g}" for x in grid.columns])
            axes[row, 2].set_yticks(range(len(grid.index)), [f"{y:g}" for y in grid.index])
            axes[row, 2].set(title="Q - C", xlabel="Requested x", ylabel="Requested y")
            fig.colorbar(im, ax=axes[row, 2], label="Probability difference", shrink=.8)
            d.to_csv(data / f"FigS0_1_row{row+1}_surfaces.csv", index=False)
    save_figure(fig, folder, "FigS0_1_response_surfaces")

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8), constrained_layout=True)
    # Administrative censoring: never-emerged runs remain in denominator.
    eventrows = []
    horizon = int(summary.training_trials.max())
    for name in SHORT:
        d = summary[summary.condition == name]
        if d.empty:
            continue
        for t in range(horizon+1):
            eventrows.append(dict(condition=name, trial=t, cumulative_emergence=float((d.T_emerge_tripartite <= t).mean()), n=len(d)))
    eventdf = pd.DataFrame(eventrows)
    for name, d in eventdf.groupby("condition"):
        axes[0, 0].step(d.trial, d.cumulative_emergence, where="post", color=COLORS[name], label=SHORT[name])
    axes[0, 0].set(title="A  Tripartite recruitment", xlabel="Training trial", ylabel="Fraction with first formation", ylim=(-.03, 1.05))
    axes[0, 0].legend(fontsize=8)
    eventdf.to_csv(data / "FigS0_2_A_emergence.csv", index=False)
    allorder = list(summary.condition.drop_duplicates())
    dotpanel(axes[0, 1], summary, "condition", "post_mixed_difference", allorder, "Post-training mixed difference",
             [x.replace("q_original", "Q old").replace("c_original", "C old").replace("q_context", "Q ctx").replace("c_context", "C ctx").replace("_", " ") for x in allorder])
    axes[0, 1].tick_params(axis="x", labelsize=7)
    axes[0, 1].set_title("B  Phase and frozen-plasticity controls")
    summary[["seed", "condition", "post_mixed_difference"]].to_csv(data / "FigS0_2_B_all_controls.csv", index=False)
    if not replay.empty:
        d = replay[(replay.comparison == "matched_classical") & (replay.topology_source == "q_original") &
                   (replay.input.isin(["training", "probe_1_1"]))].drop_duplicates(["seed", "input", "mixer_gain"]).copy()
        d["group"] = d.input.replace({"training": "Training", "probe_1_1": "Probe (1,1)"}) + np.where(d.mixer_gain == 0, " / old", " / ctx")
        order = list(d.group.drop_duplicates())
        dotpanel(axes[1, 0], d, "group", "joint_TV", order, "Joint total variation")
        axes[1, 0].tick_params(axis="x", labelsize=7, rotation=35)
        d.to_csv(data / "FigS0_2_C_fixed_kernel_TV.csv", index=False)
    axes[1, 0].set_title("C  Q/C replay on identical topology")
    axes[1, 0].set_ylim(0, 1)
    d = dephase.groupby("dephase_strength").P_output.agg(["min", "max"]).reset_index()
    d["phase_response_range"] = d["max"]-d["min"]
    axes[1, 1].plot(d.dephase_strength, d.phase_response_range, "o-", color="#009E73", markersize=4)
    axes[1, 1].set(title="D  Motif coherence ablation", xlabel="Dephasing strength", ylabel="Phase response range")
    d.to_csv(data / "FigS0_2_D_dephasing.csv", index=False)
    save_figure(fig, folder, "FigS0_2_mechanistic_diagnostics")


def analyze(out):
    out = Path(out).resolve()
    from verify_result_files import verify_files
    verify_files(out)
    summary = pd.read_csv(out / "seed_summary.csv")
    if summary.duplicated(["seed", "condition"]).any():
        raise ValueError("Duplicate seed/condition observations")
    spec = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))["spec"]
    expected = {(s, c["name"]) for s in spec["seeds"] for c in spec["conditions"]}
    found = set(zip(summary.seed, summary.condition))
    if expected != found:
        raise ValueError(f"Incomplete result matrix: expected {len(expected)}, found {len(found)}")
    paired_statistics(summary).to_csv(out / "paired_statistics.csv", index=False)
    trials = collect(out, "trial_summary.csv")
    surfaces = collect(out, "probe_surface.csv")
    states = collect(out, "final_candidate_states.csv")
    replay = collect(out, "kernel_replay.csv")
    resources = collect(out, "circuit_resources.csv")
    curves = summarize_curves(trials)
    graphs = graph_comparison(states, surfaces)
    graphs.to_csv(out / "paired_graph_and_surface.csv", index=False)
    curves.to_csv(out / "trajectory_summary.csv", index=False)
    replay.to_csv(out / "kernel_replay_all.csv", index=False)
    resources.to_csv(out / "circuit_resources_all.csv", index=False)
    surfaces.to_csv(out / "probe_surfaces_all.csv", index=False)
    states.to_csv(out / "final_candidate_states_all.csv", index=False)
    from audit_phase import audit
    auditdir = out / "phase_audit"
    project_snapshot = out / "protocol_project.json"
    evolution_snapshot = out / "protocol_evolution.json"
    project_snapshot.write_text(json.dumps(spec["project"], ensure_ascii=False, indent=2), encoding="utf-8")
    evolution_snapshot.write_text(json.dumps(spec["evolution"], ensure_ascii=False, indent=2), encoding="utf-8")
    audit(auditdir, project_snapshot, evolution_snapshot)
    motif, dephase = pd.read_csv(auditdir / "phase_motif.csv"), pd.read_csv(auditdir / "motif_dephasing.csv")
    make_figures(out, summary, curves, graphs, surfaces, replay, motif, dephase)
    note = ["# Sample0 generated analysis", "", f"Paired runs: {len(spec['seeds'])} seeds; {len(spec['conditions'])} conditions; {spec['shots']} shots per query.",
            "", "Seeds quantify finite-shot simulation uncertainty on a fixed anatomy and input schedule. They are not biological replicates.",
            "Primary surfaces use exact expectations of each seed's trained topology. Sampled probe values and raw joint counts are also retained.",
            "Confidence intervals use 5,000 paired bootstrap resamples; tests use 19,999 paired random sign flips and Holm correction across all reported finite tests.",
            "A non-significant difference does not establish equivalence. Model differences do not establish biological quantum dynamics or quantum computational advantage.",
            "The held-out metric evaluates interpolation of each model's own response surface; it is not externally validated task accuracy.",
            "", "Missing R2 for a constant output is intentionally NA. Non-emergence is right-censored; T_observed is capped at the common observation horizon.",
            "", "See seed_summary.csv, paired_statistics.csv and paired_graph_and_surface.csv for quantitative results; every panel has a CSV in figure_data/."]
    (out / "ANALYSIS_NOTES.md").write_text("\n".join(note)+"\n", encoding="utf-8")
    print(f"Analysis, SVG/PNG figures and source data: {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    args = ap.parse_args()
    analyze(args.input_dir)
