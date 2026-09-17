"""Numerical gates, quantum/dephased equivalence, V7.1 parity and locality checks."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import json
import math
import numpy as np
from classical_matched_fast_layer import Gate, unitary, evolve, simulate, total_variation
from v71_bridge import core, evolution as ev, ORIGINAL, Recorder, QP, compile_tape


def verify():
    results = []
    def check(name, error, tolerance=2e-10):
        results.append(dict(check=name, max_error=float(error), tolerance=tolerance, passed=bool(error <= tolerance)))
        if error > tolerance:
            raise AssertionError(f"{name}: {error} > {tolerance}")

    rng = np.random.default_rng(5701)
    specs = [("H", 1, 0), ("RY", 1, -.83), ("RZ", 1, .7), ("CRY", 2, -1.1),
             ("CCRY", 3, 1.8), ("ZZ", 2, -.9), ("XY", 2, .61), ("CNOT", 2, 0),
             ("CSWAP", 3, 0), ("TOFFOLI", 3, 0), ("CZ", 2, 0), ("DAMP_DILATION", 2, .31)]
    funcs = {"CRY": core.append_cry, "CCRY": core.append_ccry, "ZZ": core.append_zz,
             "XY": core.append_xy_exchange, "CSWAP": core.append_cswap,
             "TOFFOLI": core.append_toffoli, "CZ": core.append_cz,
             "DAMP_DILATION": core.append_amplitude_damping_dilation}
    for name, n, angle in specs:
        u = unitary(name, angle)
        check(f"{name}:unitarity", np.max(abs(u.conj().T @ u - np.eye(1 << n))))
        for basis in range(1 << n):
            initial = np.eye(1 << n)[basis]
            got = evolve([Gate(name, tuple(range(n)), angle)], n, "classical", initial)
            check(f"{name}:basis{basis}:probability_match", np.max(abs(got - abs(u[:, basis])**2)))
        if name in funcs:
            p = Recorder()
            args = (*range(n), angle, QP) if name not in ("CSWAP", "TOFFOLI", "CZ") else (*range(n), QP)
            funcs[name](p, *args)
            actual = np.column_stack([evolve(p.gates, n, initial=np.eye(1 << n)[i]) for i in range(1 << n)])
            # Exact unitary equivalence, modulo a single irrelevant global phase.
            a, b = np.unravel_index(np.argmax(abs(u)), u.shape)
            phase = actual[a, b] / u[a, b]
            check(f"{name}:V71_hardware_decomposition", np.max(abs(actual - phase * u)))

    # Independently propagate a mixed density matrix and fully dephase after
    # each semantic primitive. This must match the classical JOINT model.
    n = 3
    tape = [Gate("RY", (0,), .9), Gate("CRY", (0, 1), 1.2), Gate("ZZ", (1, 2), .7),
            Gate("RY", (2,), .3), Gate("XY", (1, 2), .6), Gate("CCRY", (0, 2, 1), -.8)]
    p = rng.dirichlet(np.ones(1 << n))
    rho = np.diag(p).astype(complex)
    for g in tape:
        u = np.column_stack([evolve([g], n, initial=np.eye(1 << n)[i]) for i in range(1 << n)])
        rho = u @ rho @ u.conj().T
        rho = np.diag(np.diag(rho))
    check("joint_classical_equals_primitive_dephased_density", np.max(abs(evolve(tape, n, "classical", p) - np.diag(rho))))

    # Positive and negative interference diagnostics, including a block-level
    # classical table reproducing the coherent reset-to-output distribution.
    for phi in np.linspace(-math.pi, math.pi, 9):
        t = [Gate("RY", (0,), math.pi/2), Gate("RZ", (0,), phi), Gate("RY", (0,), -math.pi/2)]
        check("Ramsey_quantum", abs(simulate(t, 1).marginals()[0] - (1-np.cos(phi))/2))
        check("Ramsey_primitive_classical", abs(simulate(t, 1, "classical").marginals()[0] - .5))
    uzz, uxy = unitary("ZZ", .61), unitary("XY", .79)
    check("same_edge_ZZ_XY_commute", np.max(abs(uzz @ uxy - uxy @ uzz)))

    bio, cfg, hw, _ = core.project_from_json(ORIGINAL / "V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_project.json")
    gains = core.scores_to_layer_gains(core.compute_mechanistic_scores(bio), bio.hardware_gain)
    evo = ev.load_evolution_config(ORIGINAL / "V7_1_SPATIAL_ADDITION_TO_MULTIPLICATION_evolution.json")
    ev.finalize_spatial_candidates(cfg, evo)
    events = ev.build_events(core, evo.training_phases[0]["neural_inputs"], gains)
    states = {c.id: ev.CandidateState(.55, True) for c in evo.candidates}
    cfgs = [replace(cfg, neural_inputs=events), ev.materialize_dynamic_config(core, cfg, evo.candidates, states, events)]
    for idx, conf in enumerate(cfgs):
        for readout in ("population", "phase"):
            orig = compile_tape(conf, gains, original=True, readout=readout)
            extension_zero = compile_tape(conf, gains, mixer_gain=0, readout=readout)
            check(f"V71_builder_exact_tape_parity_{idx}_{readout}", 0 if orig == extension_zero else 1)
            raw = compile_tape(conf, gains, original=True, semantic=False, readout=readout)
            check(f"V71_full_circuit_decomposition_{idx}_{readout}", total_variation(simulate(orig, cfg.total_qubits), simulate(raw, cfg.total_qubits)))

        quiet = replace(conf, neural_inputs=[], initialize_neural_h=False, initialize_glial_baseline=False)
        for backend in ("quantum", "classical"):
            t = compile_tape(quiet, gains, mixer_gain=-.35)
            check(f"extension_quiet_baseline_{idx}_{backend}", simulate(t, cfg.total_qubits, backend).marginals().max())
        t = compile_tape(conf, gains, mixer_gain=-.35)
        allowed = {(int(e.source), conf.n_neurons+int(e.target)) for e in conf.neuron_to_glia}
        check(f"extension_no_new_anatomy_{idx}", sum(g.qubits not in allowed for g in t if g.tag == "context_mixer"))

    # Exact compactification vs full 16-qubit propagation on a real circuit.
    t = compile_tape(cfgs[1], gains)
    for backend in ("quantum", "classical"):
        full = evolve(t, cfg.total_qubits, backend)
        pfull = full if backend == "classical" else abs(full)**2
        sparse = simulate(t, cfg.total_qubits, backend).sparse_logical()
        expanded = np.array([sparse.get(i, 0) for i in range(1 << cfg.total_qubits)])
        check(f"inactive_qubit_compaction_{backend}", np.max(abs(expanded-pfull)))
    r = simulate(compile_tape(cfgs[0], gains), cfg.total_qubits)
    m1, c1 = r.sample(3000, 17, 1, 2)
    m2, c2 = r.sample(3000, 17, 1, 2)
    check("paired_sampling_reproducible", np.max(abs(c1-c2)))
    check("joint_sample_count_conservation", abs(c1.sum()-3000))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("verification.json"))
    args = ap.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"PASS: {len(result)} numerical checks; maximum error={max(x['max_error'] for x in result):.3g}")
