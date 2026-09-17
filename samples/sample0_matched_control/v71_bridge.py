"""Reuse V7.1 construction and plasticity, with a replaceable fast kernel.

No pyqpanda installation is needed for Sample0. Semantic recording binds a
function to a private copy of its globals; it never changes the original module.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import FunctionType
from dataclasses import replace
from classical_matched_fast_layer import Gate
import v71_context_builder

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "v71_original"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


core = load_module("sample0_v71_core", ORIGINAL / "V7_1_neuroglial_gate_qpu.py")
evolution = load_module("sample0_v71_evolution", ORIGINAL / "V7_1_SPATIAL_EVOLVING_NETWORK.py")


class Recorder:
    def __init__(self):
        self.gates = []

    def __lshift__(self, gate):
        if gate is not None:
            self.gates.append(gate)
        return self


QP = {
    "QProg": Recorder,
    "H": lambda q: Gate("H", (q,)),
    "RY": lambda q, a: Gate("RY", (q,), a),
    "RZ": lambda q, a: Gate("RZ", (q,), a),
    "CNOT": lambda a, b: Gate("CNOT", (a, b)),
    "measure": lambda q, c: None,
}


def _append(name, arity):
    def append(prog, *args):
        qubits = tuple(int(x) for x in args[:arity])
        rest = args[arity:-1]  # Last positional argument is qp.
        angle = float(rest[0]) if rest else 0.0
        if name in ("CRY", "CCRY", "ZZ", "XY") and abs(angle) < 1e-12:
            return
        prog << Gate(name, qubits, angle)
    return append


SEMANTIC = {name: _append(op, arity) for name, op, arity in [
    ("append_cry", "CRY", 2), ("append_ccry", "CCRY", 3),
    ("append_zz", "ZZ", 2), ("append_xy_exchange", "XY", 2),
    ("append_cswap", "CSWAP", 3), ("append_cz", "CZ", 2),
    ("append_toffoli", "TOFFOLI", 3),
    ("append_amplitude_damping_dilation", "DAMP_DILATION", 2),
]}


def bound_builder(original=False, semantic=True):
    f = core.build_layered_program if original else v71_context_builder.build_layered_program
    namespace = dict(vars(core))
    if semantic:
        namespace.update(SEMANTIC)
    new = FunctionType(f.__code__, namespace, f.__name__, f.__defaults__, f.__closure__)
    new.__kwdefaults__ = f.__kwdefaults__
    return new


def context_hook(gain):
    """A late, local, activity-controlled re-evaluation of existing N->G input.

    beta_ij = gain * theta_NG,ij / layers. No new anatomical edge, no Hadamard
    readout, and no activation of the all-zero state. Negative gain represents
    an effective recovery/re-evaluation convention, NOT an identified ion gate.
    The entire extension and its strength are unvalidated model assumptions.
    """
    if not -1 <= gain <= 1:
        raise ValueError("context mixer gain must be in [-1,1]")

    def hook(prog, cfg, gains, qp, layer, evidence):
        for edge in cfg.neuron_to_glia or []:
            if not core.glia_microdomain_is_active(cfg, layer, int(edge.target), evidence):
                continue
            theta = core.clip(edge.weight * gains.neuron_to_glia_gain, -1.5, 1.5)
            beta = gain * theta / cfg.layers
            if abs(beta) > 1e-12:
                prog << Gate("CRY", (int(edge.source), cfg.n_neurons + int(edge.target)), beta, "context_mixer")
    return hook


def compile_tape(cfg, gains, mixer_gain=0.0, phase_scale=1.0, zz_scale=1.0,
                 readout="population", original=False, semantic=True):
    kwargs = dict(qp=QP, measure_glia=True, neural_readout_mode=readout)
    if original and mixer_gain:
        raise ValueError("Original builder has no mixer")
    if not original:
        kwargs["context_hook"] = context_hook(mixer_gain) if mixer_gain else None
    tape = bound_builder(original, semantic)(cfg, gains, **kwargs).gates
    if not semantic and (phase_scale != 1 or zz_scale != 1 or mixer_gain):
        raise ValueError("Ablations/mixers apply at semantic boundaries only")
    return [replace(g, angle=g.angle * (phase_scale * zz_scale if g.name == "ZZ" else phase_scale))
            if g.name in ("ZZ", "RZ") else g for g in tape]


def to_qpanda(tape, logical_width, measure_glia=True):
    """Optional circuit export only. Does not submit jobs or require an API key."""
    qp = core.import_qpanda_core()
    prog = qp["QProg"]()
    names = {"CRY": "append_cry", "CCRY": "append_ccry", "ZZ": "append_zz",
             "XY": "append_xy_exchange", "CSWAP": "append_cswap", "CZ": "append_cz",
             "TOFFOLI": "append_toffoli", "DAMP_DILATION": "append_amplitude_damping_dilation"}
    angled = {"CRY", "CCRY", "ZZ", "XY", "DAMP_DILATION"}
    for g in tape:
        if g.name in names:
            args = (*g.qubits, g.angle, qp) if g.name in angled else (*g.qubits, qp)
            getattr(core, names[g.name])(prog, *args)
        elif g.name in ("RY", "RZ"):
            prog << qp[g.name](g.qubits[0], g.angle)
        else:
            prog << qp[g.name](*g.qubits)
    qs = list(range(logical_width if measure_glia else logical_width // 2))
    prog << qp["measure"](qs, qs)
    return prog
