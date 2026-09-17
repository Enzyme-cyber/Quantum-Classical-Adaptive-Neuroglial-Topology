"""Exact phase-to-population diagnostics with SVG-ready source data.

Tests a transparent local motif and the actual uploaded V7.1 circuit.
The local motif is a mathematical validation, not biological evidence.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import math
import numpy as np
import pandas as pd
from classical_matched_fast_layer import Gate, simulate, evolve, unitary, total_variation
from v71_bridge import core, evolution as ev, ORIGINAL, compile_tape


def apply_density(rho, gate, n):
    u = np.column_stack([evolve([gate], n, initial=np.eye(1 << n)[i]) for i in range(1 << n)])
    return u @ rho @ u.conj().T


def motif_prob(phi, beta, dephase=0.0):
    """N0 -> G0(context) -> N1; insert dephasing before late N0->G0 mixer."""
    # Logical layout for this 3-variable diagnostic: N0=0, G0=1, N1=2.
    alpha, theta = math.pi/2, 1.1
    prefix = [Gate("RY", (0,), math.pi), Gate("CRY", (0, 1), alpha), Gate("RZ", (1,), phi)]
    suffix = ([] if beta == 0 else [Gate("CRY", (0, 1), beta, "context_mixer")]) + [Gate("CRY", (1, 2), theta)]
    tape = prefix + suffix
    psi = evolve(prefix, 3)
    rho = np.outer(psi, psi.conj())
    # Dephasing strength lambda: rho -> (1-lambda)*rho + lambda*diag(rho).
    rho = (1-dephase)*rho + dephase*np.diag(np.diag(rho))
    for g in suffix:
        rho = apply_density(rho, g, 3)
    pout = np.real(np.diag(rho))[4:].sum()
    classical = simulate(tape, 3, "classical").marginals()[2]
    expected = (1 - math.cos(alpha)*math.cos(beta) + math.sin(alpha)*math.sin(beta)*math.cos(phi)) / 2 * math.sin(theta/2)**2
    # Whole-block classical reset transition is exactly |U_block[:,0]|^2.
    block = np.abs(evolve(tape, 3))**2
    return float(pout), float(classical), float(block[4:].sum()), expected


def audit(out, project=None, evolution_config=None):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for phi in np.linspace(-math.pi, math.pi, 49):
        for beta in [0.0, -.15, -.35, -.6, .35]:
            q, c, block, formula = motif_prob(phi, beta)
            if abs(q-formula) > 1e-10 or abs(q-block) > 1e-10:
                raise AssertionError("Local motif disagrees with analytic formula/block-classical control")
            rows.append(dict(phase_rad=phi, beta=beta, quantum=q, primitive_classical=c,
                             block_classical=block, analytic=formula))
    pd.DataFrame(rows).to_csv(out / "phase_motif.csv", index=False)
    noise = []
    for level in np.linspace(0, 1, 11):
        for phi in np.linspace(-math.pi, math.pi, 49):
            q, c, _, _ = motif_prob(phi, -.35, level)
            noise.append(dict(dephase_strength=level, phase_rad=phi, P_output=q, primitive_classical=c))
    pd.DataFrame(noise).to_csv(out / "motif_dephasing.csv", index=False)

    project = Path(project) if project is not None else ORIGINAL / "V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json"
    evolution_config = Path(evolution_config) if evolution_config is not None else ORIGINAL / "V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json"
    bio, cfg, _, _ = core.project_from_json(project)
    gains = core.scores_to_layer_gains(core.compute_mechanistic_scores(bio), bio.hardware_gain)
    evo = ev.load_evolution_config(evolution_config)
    ev.finalize_spatial_candidates(cfg, evo)
    # All-candidates-active is an explicit diagnostic stress topology, not a
    # claimed training outcome. Learned topology audits are in kernel_replay.csv.
    cases = [("initial", ev.initialize_states(evo)),
             ("diagnostic_candidates_active", {c.id: ev.CandidateState(.55, True) for c in evo.candidates})]
    network_rows = []
    for topology, states in cases:
        for layers in sorted({1, 2, cfg.layers}):
            base = replace(cfg, layers=layers)
            inputs = ev.build_events(core, evo.training_phases[0]["neural_inputs"], gains)
            inputs = [event for event in inputs if event.layer < layers]
            conf = ev.materialize_dynamic_config(core, base, evo.candidates, states, inputs)
            for mixer in [0, -.35]:
                for readout in ["population", "phase"]:
                    baseline = simulate(compile_tape(conf, gains, mixer_gain=mixer, readout=readout), conf.total_qubits)
                    for what, ps, zs in [("no_zz", 1, 0), ("no_explicit_phase", 0, 1)]:
                        alt = simulate(compile_tape(conf, gains, mixer_gain=mixer, phase_scale=ps, zz_scale=zs, readout=readout), conf.total_qubits)
                        delta = alt.marginals() - baseline.marginals()
                        network_rows.append(dict(topology=topology, layers=layers, mixer_gain=mixer, readout=readout,
                                                 ablation=what, maximum_marginal_change=float(abs(delta).max()),
                                                 neuronal_max_change=float(abs(delta[:cfg.n_neurons]).max()),
                                                 glial_max_change=float(abs(delta[cfg.n_neurons:]).max()),
                                                 joint_TV=total_variation(baseline, alt)))
    pd.DataFrame(network_rows).to_csv(out / "v71_phase_audit.csv", index=False)
    # Local RZ on glia controlling a feedback CRY commutes with that feedback.
    # Same-edge ZZ and XY commute too: neither alone guarantees phase readout.
    return pd.DataFrame(network_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("phase_audit"))
    ap.add_argument("--project", type=Path, default=None)
    ap.add_argument("--evolution-config", type=Path, default=None)
    args = ap.parse_args()
    result = audit(args.output_dir, args.project, args.evolution_config)
    print(result.to_string(index=False))
