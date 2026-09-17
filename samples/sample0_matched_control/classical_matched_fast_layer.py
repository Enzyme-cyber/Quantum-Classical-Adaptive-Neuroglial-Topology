"""Exact joint-probability control at BIOLOGICAL PRIMITIVE boundaries.

For each declared unitary U, T[y,x] = abs(U[y,x])**2. This matches every
computational-basis conditional transition, retains arbitrary classical joint
correlations, and removes interference BETWEEN primitives. It is NOT obtained
by independently stochasticizing CNOT/RY hardware decompositions of a CRY.

Qubit 0 is the least significant bit, also within each Gate.qubits tuple.
Both backends are classical CPU computations with exponential state storage.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import numpy as np


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[int, ...]
    angle: float = 0.0
    tag: str = "legacy"


def _rotation_block(n, a, b, theta):
    u = np.eye(n, dtype=complex)
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    u[a, a] = u[b, b] = c
    u[a, b], u[b, a] = -s, s
    return u


@lru_cache(maxsize=4096)
def unitary(name, angle=0.0):
    """Small explicit matrices; used by both coherent and classical engines."""
    if not np.isfinite(angle):
        raise ValueError("Non-finite gate angle")
    if name == "H":
        return np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    if name == "RY":
        return _rotation_block(2, 0, 1, angle)
    if name == "RZ":
        return np.diag(np.exp(0.5j * angle * np.array([-1, 1])))
    if name in ("CRY", "CCRY"):
        return _rotation_block(4, 1, 3, angle) if name == "CRY" else _rotation_block(8, 3, 7, angle)
    if name == "ZZ":
        return np.diag(np.exp(-0.5j * angle * np.array([1, -1, -1, 1])))
    if name == "XY":
        u = np.eye(4, dtype=complex)
        u[1, 1] = u[2, 2] = np.cos(angle)
        u[1, 2] = u[2, 1] = -1j * np.sin(angle)
        return u
    if name in ("CNOT", "TOFFOLI", "CSWAP"):
        size, a, b = {"CNOT": (4, 1, 3), "TOFFOLI": (8, 3, 7), "CSWAP": (8, 3, 5)}[name]
        u = np.eye(size, dtype=complex)
        u[:, [a, b]] = u[:, [b, a]]
        return u
    if name == "CZ":
        return np.diag([1, 1, 1, -1]).astype(complex)
    if name == "DAMP_DILATION":
        # angle is gamma. The original glial environment is NOT reset here.
        if not 0 <= angle <= 1:
            raise ValueError("Damping gamma must be in [0,1]")
        cry = unitary("CRY", 2 * math.asin(math.sqrt(angle)))
        cx_reversed = np.eye(4, dtype=complex)
        cx_reversed[:, [2, 3]] = cx_reversed[:, [3, 2]]
        return cx_reversed @ cry
    raise ValueError(f"Unsupported semantic primitive: {name}")


@lru_cache(maxsize=512)
def blocks(n, qubits):
    """Gather disjoint local basis blocks without forming a 2**n matrix."""
    if len(set(qubits)) != len(qubits) or any(q < 0 or q >= n for q in qubits):
        raise ValueError(f"Invalid qubits {qubits} for width {n}")
    ids = np.arange(1 << n, dtype=np.int64)
    mask = sum(1 << q for q in qubits)
    base = ids[(ids & mask) == 0]
    shifts = np.array([sum(((x >> j) & 1) << q for j, q in enumerate(qubits))
                       for x in range(1 << len(qubits))])
    return base[None, :] + shifts[:, None]


def apply_gate(state, gate, n, classical=False):
    u = unitary(gate.name, float(gate.angle))
    if classical:
        if gate.name in ("RZ", "ZZ", "CZ"):
            return
        u = np.abs(u) ** 2
    idx = blocks(n, gate.qubits)
    values = state[idx]
    state[idx] = np.einsum("ab,bc->ac", u, values, optimize=False)


def compact_tape(tape):
    """Exactly remove qubits that remain |0> and see only diagonal operations.

    The logical width is retained separately. This is not a mean-field or
    entanglement truncation. ZZ with one fixed-zero endpoint becomes RZ.
    """
    active = sorted({q for g in tape if g.name not in ("RZ", "ZZ", "CZ") for q in g.qubits})
    lookup = {q: i for i, q in enumerate(active)}
    out = []
    for g in tape:
        kept = tuple(q for q in g.qubits if q in lookup)
        if g.name in ("RZ", "ZZ", "CZ"):
            if not kept:
                continue  # Global phase only.
            if g.name == "CZ" and len(kept) < 2:
                continue
            if g.name == "ZZ" and len(kept) == 1:
                g = replace(g, name="RZ", qubits=kept)
        out.append(replace(g, qubits=tuple(lookup[q] for q in g.qubits)))
    return out, tuple(active)


def evolve(tape, n, backend="quantum", initial=None):
    if backend not in ("quantum", "classical"):
        raise ValueError(backend)
    classical = backend == "classical"
    state = np.zeros(1 << n, dtype=float if classical else complex)
    state[0] = 1
    if initial is not None:
        state[:] = initial
    for gate in tape:
        apply_gate(state, gate, n, classical)
    return state


@dataclass
class FastResult:
    probability: np.ndarray
    active_qubits: tuple[int, ...]
    logical_width: int
    circuit_sha256: str

    def marginals(self, probability=None):
        p = self.probability if probability is None else probability
        ids = np.arange(len(p))
        out = np.zeros(self.logical_width)
        for k, q in enumerate(self.active_qubits):
            out[q] = p[((ids >> k) & 1) != 0].sum()
        return out

    def sample(self, shots, seed, stream, index):
        """Paired common uniforms, independent streams for trial/probe types."""
        if shots < 0:
            raise ValueError("shots cannot be negative")
        if shots == 0:
            return self.marginals(), None
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream), int(index)]))
        cdf = self.probability.cumsum()
        cdf[-1] = 1.0
        counts = np.bincount(np.searchsorted(cdf, rng.random(shots), side="right"), minlength=len(cdf))
        return self.marginals(counts / shots), counts

    def sparse_logical(self):
        return {sum(((i >> k) & 1) << q for k, q in enumerate(self.active_qubits)): float(p)
                for i, p in enumerate(self.probability) if p > 1e-15}


def simulate(tape, logical_width, backend="quantum", max_active_qubits=20):
    compact, active = compact_tape(tape)
    if len(active) > max_active_qubits:
        raise ValueError(f"Exact {backend} backend needs 2**{len(active)} states; limit={max_active_qubits}")
    state = evolve(compact, len(active), backend)
    p = state if backend == "classical" else np.abs(state) ** 2
    total = p.sum()
    if not np.all(np.isfinite(p)) or p.min() < -1e-12 or abs(total - 1) > 1e-8:
        raise ArithmeticError(f"Invalid probability distribution: total={total}")
    p = np.maximum(p, 0) / total
    digest = hashlib.sha256(json.dumps([(g.name, g.qubits, g.angle, g.tag) for g in tape],
                                     separators=(",", ":")).encode()).hexdigest()
    return FastResult(p, active, logical_width, digest)


def classical_matched_fast_layer(tape, logical_width, max_active_qubits=20):
    return simulate(tape, logical_width, "classical", max_active_qubits)


def total_variation(a, b):
    da, db = a.sparse_logical(), b.sparse_logical()
    return 0.5 * sum(abs(da.get(i, 0) - db.get(i, 0)) for i in da.keys() | db.keys())
