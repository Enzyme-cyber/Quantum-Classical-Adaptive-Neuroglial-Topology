#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V7.0 Neuroglial Gate QPU
==================================

Default architecture
--------------------
16 total qubits = 8 neuronal qubits + 8 glial qubits.

The same program can be configured for 4..128 total qubits (even numbers):
    N layer -> G layer -> G-G network -> N feedback

For each discrete layer l:
    1) selected neuronal inputs are encoded on N_i
    2) editable N_i -> G_j directed couplings are applied
    3) editable G_i <-> G_j network couplings are applied
    4) editable G_j -> N_k feedback couplings are applied

The 9 biological inputs from V6.1 are retained:
    KCNJ10, KCNJ16, ATP1A2, AQP4,
    SLC1A2, SLC1A3, GJA1, GJB6, ITPR2
plus pH, extracellular K+, astrocyte Vm and a global neuronal-drive parameter.

The biological variables are compressed classically into mechanistic scores:
    Kir buffering/noise, glutamate clearance, Ca signaling,
    gap-junction coupling, slow context, neural excitability, glial-network state.

Those scores then parameterize the quantum circuit:
    - neuronal input amplitude / phase
    - directed N->G controlled rotations
    - G-G hybrid ZZ phase coupling + optional XY/partial-iSWAP state exchange
    - directed G->N controlled rotations
    - glial local phase/context rotations

V6.6 retains the validated V6.3 neuroglial gate layer and the V6.4 hybrid
astrocyte-astrocyte operator, but changes the recruitment rule for NET state
transfer:
    - each G-G anatomical edge keeps its signed ZZ phase/context interaction
      continuously
    - an optional XY/partial-iSWAP exchange channel represents a net-flux/state-
      redistribution analogue across connexin-coupled astrocytes
    - in the primary V6.6 mode, XY is source-activity gated: it is applied only
      when a shared association glial endpoint receives coincident external
      neuronal evidence on that layer
    - this gate represents the presence of a local source/gradient for net flux,
      NOT instantaneous opening/closing of Cx43/Cx30 channels
    - ZZ phase gain and XY exchange capacity remain biologically parameterized separately
    - MD/literature-informed KCNJ10:KCNJ16 -> Kir4.1/Kir4.1-Kir5.1 composition proxy
    - Kir composition/pH -> effective conductance proxy -> damping gamma
    - shared-astrocyte two-input threshold -> parameterized CC-RY (Toffoli limit)
    - optional glial-controlled network exchange -> CSWAP/Fredkin idealized operator
    - built-in falsifiable gate validation for CRY/CNOT/CZ/ZZ/XY/CCRY/Toffoli/CSWAP/damping/reset

V6.8 added one biologically motivated topology type without adding a new gate
vocabulary: an active tripartite synapse uses the existing CC-RY primitive with
a sensory neuronal control, an astrocytic-context control, and a postsynaptic
neuronal target.  The primary mode requires phasic evidence on the sensory
control, so a tonic carrier state is not treated as a new afferent event.


V6.9 added a classical-neuronal directed topology primitive without changing the
neuroglial or tripartite mechanisms:
    - editable N_i -> N_j directed synapses
    - positive weight is an excitatory feed-forward synaptic analogue
    - implemented with the already-validated controlled-RY primitive
    - its gain is parameterized from neural excitability / global neuronal drive,
      not from connexin or astrocyte-Ca-specific scores
    - this is a circuit encoding of directed neuronal activation, not a claim that
      a biological chemical synapse literally implements CRY.


V7.0 preserves all V6.9 editable fast-circuit topology and adds two clean-state
switches used by the companion evolving-network runner. Across-trial slow traces
and structural plasticity live in V7_0_EVOLVING_NETWORK.py, which rebuilds this
core NetworkConfig between trials. Thus the fast circuit remains reusable and
GUI/JSON-editable, while candidate biological topology can emerge or disappear
without hard-coding a specific task into this core.

Scientific scope
----------------
This is a hybrid quantum-classical neuroglial emulator. Protein-to-angle weights
are mechanistically constrained, dimensionless encoding weights; they are not
measured biochemical kinetic constants. The program does not claim that
astrocytes are physical qubits or that biological brain tissue sustains quantum
entanglement.

Scalability
-----------
- 16 qubits is the recommended first real-QPU configuration.
- 32/64/128 qubits are supported for circuit construction / real-QPU submission
  when the selected backend has enough physical qubits.
- Local CPUQVM is intentionally blocked above a configurable safe limit because
  a statevector simulator scales exponentially. Use "Build / resource check"
  or a real QPU for larger circuits.
- Sparse topologies are strongly recommended for 64/128 qubits because circuit
  depth is driven by the number of N->G, G-G and G->N edges per layer.

OriginQ cloud
-------------
API is read ONLY from:
    QPANDA_QCLOUD_API_KEY
Optional backend preference:
    QPANDA_QCLOUD_BACKEND=auto

The cloud runner follows the previously validated pattern:
    submit a wave of cloud jobs -> keep jobs in flight -> wait concurrently ->
    save job IDs / physical blocks / timing to job_ids.json.

Dependencies
------------
pip install -U pyqpanda3 numpy pandas matplotlib
"""

from __future__ import annotations

import json
import math
import os
import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


VERSION = "V7.1"

PROTEIN_INFO = {
    "KCNJ10": "Kir4.1 — K+ buffering / passive leak",
    "KCNJ16": "Kir5.1 — Kir4.1/Kir5.1 composition & pH sensitivity",
    "ATP1A2": "Na+/K+-ATPase alpha2 — ionic recovery/homeostasis",
    "AQP4": "Aquaporin-4 — water/K+ spatial buffering & slow context",
    "SLC1A2": "EAAT2 — glutamate clearance",
    "SLC1A3": "EAAT1 — glutamate clearance",
    "GJA1": "Connexin-43 — astrocyte gap-junction coupling",
    "GJB6": "Connexin-30 — astrocyte coupling/context",
    "ITPR2": "IP3R2 — astrocytic Ca2+ signaling",
}

REFERENCE_PROTEINS = {
    "KCNJ10": 1.0,
    "KCNJ16": 0.325,
    "ATP1A2": 1.0,
    "AQP4": 1.0,
    "SLC1A2": 1.0,
    "SLC1A3": 1.0,
    "GJA1": 1.0,
    "GJB6": 1.0,
    "ITPR2": 1.0,
}

REFERENCE_ENV = {
    "pH": 7.40,
    "K_out_mM": 3.50,
    "Vm_mV": -80.0,
    "neuronal_drive": 1.0,
    "hardware_gain": 1.0,
}

PRESETS = {
    "Reference": dict(REFERENCE_PROTEINS),
    "Kir4.1 low": {**REFERENCE_PROTEINS, "KCNJ10": 0.30},
    "Kir5.1 high": {**REFERENCE_PROTEINS, "KCNJ16": 0.80},
    "Gap junction high": {**REFERENCE_PROTEINS, "GJA1": 1.8, "GJB6": 1.5},
    "Ca2+ high": {**REFERENCE_PROTEINS, "ITPR2": 1.8},
    "Glutamate clearance low": {**REFERENCE_PROTEINS, "SLC1A2": 0.35, "SLC1A3": 0.55},
    "Epilepsy-like stress": {
        **REFERENCE_PROTEINS,
        "KCNJ10": 0.35,
        "ATP1A2": 0.55,
        "AQP4": 0.65,
        "SLC1A2": 0.40,
        "SLC1A3": 0.65,
        "ITPR2": 1.35,
    },
}


# =============================================================================
# 1. Utilities / biological compression
# =============================================================================

def clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), lo, hi))


def safe_log2_ratio(value: float, reference: float, eps: float = 1e-9) -> float:
    return float(np.log2((max(value, 0.0) + eps) / (max(reference, 0.0) + eps)))


def bounded_fold_score(value: float, reference: float) -> float:
    return float(np.tanh(safe_log2_ratio(value, reference) / 2.0))


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class V3NoiseMap:
    def __init__(self, path: str | Path):
        path = Path(path)
        df = pd.read_csv(path)
        required = {"ratio", "mean_pre_voltage_SD_mV"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"V3 summary missing columns {sorted(missing)}; required={sorted(required)}"
            )
        d = df[["ratio", "mean_pre_voltage_SD_mV"]].copy()
        d["ratio"] = pd.to_numeric(d["ratio"], errors="coerce")
        d["mean_pre_voltage_SD_mV"] = pd.to_numeric(
            d["mean_pre_voltage_SD_mV"], errors="coerce"
        )
        d = d.dropna().groupby("ratio", as_index=False)["mean_pre_voltage_SD_mV"].mean()
        d = d.sort_values("ratio")
        if len(d) < 2:
            raise ValueError("V3 summary needs at least two valid ratio points.")
        self.x = d["ratio"].to_numpy(float)
        self.y = d["mean_pre_voltage_SD_mV"].to_numpy(float)
        try:
            from scipy.interpolate import PchipInterpolator
            self.f = PchipInterpolator(self.x, self.y, extrapolate=False)
        except Exception:
            self.f = None

    def __call__(self, ratio: float) -> float:
        r = float(np.clip(ratio, self.x.min(), self.x.max()))
        if self.f is None:
            y = float(np.interp(r, self.x, self.y))
        else:
            y = float(self.f(r))
        return max(y, 0.0)



# =============================================================================
# V6.6 KCNJ10/KCNJ16 -> Kir composition / open-system parameterization
# =============================================================================

# Literature-informed single-channel values used in the earlier KCNJ10:KCNJ16
# validation chain.  They constrain the encoding direction and relative scale;
# V6.6 still does NOT treat the final quantum angles/gamma as measured kinetic
# constants.
KIR_LITERATURE = {
    "kir44_single_g_ps": 22.2,
    "kir45_single_g_ps": 59.2,
    "kir44_po_max": 0.890,
    "kir45_po_max": 0.934,
    "kir44_pka": 5.99,
    "kir44_hill": 2.0,
    "kir45_pka": 7.45,
    "kir45_hill": 2.3,
}


def hill_open_fraction(ph: float, pka: float, hill: float) -> float:
    """Fraction of maximal activity remaining at intracellular pH."""
    return float(1.0 / (1.0 + 10.0 ** (float(hill) * (float(pka) - float(ph)))))


def kir_channel_po(ph: float, heteromer: bool) -> float:
    if heteromer:
        return float(
            KIR_LITERATURE["kir45_po_max"]
            * hill_open_fraction(ph, KIR_LITERATURE["kir45_pka"], KIR_LITERATURE["kir45_hill"])
        )
    return float(
        KIR_LITERATURE["kir44_po_max"]
        * hill_open_fraction(ph, KIR_LITERATURE["kir44_pka"], KIR_LITERATURE["kir44_hill"])
    )


def assembly_from_ratio(ratio_kcnj16_over_kcnj10: float, eta: float = 0.75):
    """
    Coarse expression -> tetramer assembly map inherited from the KCNJ10:16
    validation workflow.  Kir4.1 availability is normalized to A=1, Kir5.1 to
    B=ratio.  Heteromer is represented as Kir4.1_2/Kir5.1_2 and homomer as
    Kir4.1_4.  Returns channel-number fractions plus unnormalized channel counts.
    """
    A = 1.0
    B = max(float(ratio_kcnj16_over_kcnj10), 0.0)
    eta = clip(float(eta), 0.0, 1.0)
    n45 = eta * min(A, B) / 2.0
    a_remaining = max(A - 2.0 * n45, 0.0)
    n44 = a_remaining / 4.0
    total = n44 + n45
    if total <= 0:
        return 1.0, 0.0, n44, n45
    return n44 / total, n45 / total, n44, n45


def kir_composition_metrics(kcnj10: float, kcnj16: float, ph: float, eta: float = 0.75):
    """
    Produce an MD/literature-informed *proxy* linking expression composition to
    channel composition and effective open conductance.  This deliberately avoids
    claiming direct RNA->membrane stoichiometry or an absolute whole-cell current.
    """
    eps = 1e-12
    k10 = max(float(kcnj10), 0.0)
    k16 = max(float(kcnj16), 0.0)
    ratio = (k16 + eps) / (k10 + eps)
    f44, f45, n44, n45 = assembly_from_ratio(ratio, eta)
    po44 = kir_channel_po(ph, False)
    po45 = kir_channel_po(ph, True)
    # assembly_from_ratio is per unit KCNJ10 availability; multiply by KCNJ10
    # abundance to preserve the loss-of-function direction when KCNJ10 falls.
    g_proxy = k10 * (
        n44 * KIR_LITERATURE["kir44_single_g_ps"] * po44
        + n45 * KIR_LITERATURE["kir45_single_g_ps"] * po45
    )
    return {
        "ratio": float(ratio),
        "f44": float(f44),
        "f45": float(f45),
        "n44": float(n44),
        "n45": float(n45),
        "po44": float(po44),
        "po45": float(po45),
        "g_proxy": float(max(g_proxy, 0.0)),
    }


def reference_kir_g_proxy(eta: float = 0.75) -> float:
    m = kir_composition_metrics(
        REFERENCE_PROTEINS["KCNJ10"],
        REFERENCE_PROTEINS["KCNJ16"],
        REFERENCE_ENV["pH"],
        eta,
    )
    return max(float(m["g_proxy"]), 1e-12)


@dataclass
class BiologicalInput:
    proteins: dict[str, float]
    pH: float = 7.40
    K_out_mM: float = 3.50
    Vm_mV: float = -80.0
    neuronal_drive: float = 1.0
    hardware_gain: float = 1.0
    scale_mode: str = "relative_linear"
    v3_summary: str = ""
    kir_assembly_eta: float = 0.75


@dataclass
class MechanisticScores:
    ratio_KCNJ16_over_KCNJ10: float
    kir_f44: float
    kir_f45: float
    kir_po44: float
    kir_po45: float
    kir_effective_g_proxy: float
    kir_effective_g_ratio_to_reference: float
    kir_conductance_score: float
    kir_noise_SD_mV: float | None
    kir_noise_score: float
    kir_buffer_score: float
    kir_heteromer_pH_context: float
    ionic_homeostasis_score: float
    glutamate_clearance_score: float
    ca_signal_score: float
    gap_junction_score: float
    slow_context_score: float
    neural_excitability_score: float
    glial_network_score: float


@dataclass
class LayerGains:
    neuronal_input_gain: float
    neuronal_phase_gain: float
    neuron_to_neuron_gain: float
    glial_local_bias: float
    glial_phase_gain: float
    neuron_to_glia_gain: float
    # V6.6 separates phase coupling from population/state exchange and time-scales XY across the context window.
    glia_glia_phase_gain: float
    glia_glia_exchange_gain: float
    glia_to_neuron_gain: float
    readout_phase_gain: float
    # V6.6 operator parameters.  These are bounded encoding coefficients.
    kir_damping_gamma: float
    threshold_gate_gain: float
    exchange_gate_gain: float


def convert_scale(proteins: dict[str, float], scale_mode: str) -> dict[str, float]:
    out = {}
    for gene in PROTEIN_INFO:
        value = float(proteins[gene])
        if value < 0:
            raise ValueError(f"{gene} must be >= 0.")
        if scale_mode == "relative_linear":
            out[gene] = value
        elif scale_mode == "log1p_expression":
            out[gene] = float(np.expm1(value))
        else:
            raise ValueError(f"Unknown scale mode: {scale_mode}")
    return out


def validate_biological_input(inp: BiologicalInput) -> None:
    missing = set(PROTEIN_INFO) - set(inp.proteins)
    if missing:
        raise ValueError(f"Missing protein inputs: {sorted(missing)}")
    if not (5.5 <= inp.pH <= 8.5):
        raise ValueError("pH should be 5.5..8.5 for this screening model.")
    if inp.K_out_mM <= 0:
        raise ValueError("Extracellular K+ must be > 0 mM.")
    if not (-140 <= inp.Vm_mV <= 40):
        raise ValueError("Vm must be -140..+40 mV.")
    if inp.neuronal_drive < 0:
        raise ValueError("Neuronal drive must be >= 0.")
    if not (0.1 <= inp.hardware_gain <= 5.0):
        raise ValueError("Hardware gain must be 0.1..5.0.")
    if not (0.0 <= inp.kir_assembly_eta <= 1.0):
        raise ValueError("Kir assembly eta must be 0..1.")


def compute_mechanistic_scores(inp: BiologicalInput) -> MechanisticScores:
    validate_biological_input(inp)
    p = convert_scale(inp.proteins, inp.scale_mode)
    z = {
        g: bounded_fold_score(p[g], REFERENCE_PROTEINS[g])
        for g in PROTEIN_INFO
    }

    eps = 1e-9
    km = kir_composition_metrics(
        p["KCNJ10"], p["KCNJ16"], inp.pH, inp.kir_assembly_eta
    )
    ratio = km["ratio"]
    ratio_ref = REFERENCE_PROTEINS["KCNJ16"] / REFERENCE_PROTEINS["KCNJ10"]
    ratio_score = float(np.tanh(np.log2((ratio + eps) / (ratio_ref + eps)) / 1.5))
    g_ref = reference_kir_g_proxy(inp.kir_assembly_eta)
    g_ratio = float((km["g_proxy"] + eps) / (g_ref + eps))
    g_score = float(np.tanh(np.log2(g_ratio + eps) / 1.5))

    v3_noise = None
    noise_score = ratio_score
    if inp.v3_summary and Path(inp.v3_summary).exists():
        noise_map = V3NoiseMap(inp.v3_summary)
        v3_noise = noise_map(ratio)
        ref_noise = noise_map(ratio_ref)
        noise_score = float(
            np.tanh(np.log2((v3_noise + eps) / (ref_noise + eps)) / 1.5)
        )

    ph_acid = clip((7.40 - inp.pH) / 0.60, -1, 1)
    k_stress = float(np.tanh(np.log2(max(inp.K_out_mM, eps) / 3.50)))
    vm_depol = float(np.tanh((inp.Vm_mV + 80.0) / 20.0))
    drive = float(np.tanh((inp.neuronal_drive - 1.0) / 0.75))

    # V6.6: Kir buffering retains both absolute KCNJ10 availability and the
    # MD/literature-informed composition/conductance proxy.  The old terms are
    # retained so V6.2.1 behavior is not discarded.
    kir_buffer = clip(
        0.35*z["KCNJ10"] + 0.25*g_score
        + 0.20*z["ATP1A2"] + 0.20*z["AQP4"]
        - 0.15*noise_score,
        -1, 1
    )
    heteromer_context = clip(0.50*ratio_score + 0.30*ph_acid + 0.20*g_score, -1, 1)
    glutamate_clearance = clip(0.70*z["SLC1A2"] + 0.30*z["SLC1A3"], -1, 1)
    gap_junction = clip(0.70*z["GJA1"] + 0.30*z["GJB6"], -1, 1)
    slow_context = clip(0.55*z["ATP1A2"] + 0.45*z["AQP4"], -1, 1)
    ca_signal = clip(
        0.60*z["ITPR2"] + 0.25*drive + 0.15*ph_acid
        + 0.10*noise_score - 0.15*glutamate_clearance,
        -1, 1
    )
    ionic_homeostasis = clip(
        0.45*kir_buffer + 0.20*g_score + 0.25*slow_context
        - 0.25*k_stress - 0.10*vm_depol,
        -1, 1
    )
    excitability = clip(
        0.55*drive + 0.35*k_stress + 0.20*vm_depol
        - 0.35*glutamate_clearance - 0.20*kir_buffer,
        -1, 1
    )
    glial_network = clip(
        0.65*gap_junction + 0.20*ca_signal + 0.15*slow_context,
        -1, 1
    )

    return MechanisticScores(
        ratio_KCNJ16_over_KCNJ10=float(ratio),
        kir_f44=float(km["f44"]),
        kir_f45=float(km["f45"]),
        kir_po44=float(km["po44"]),
        kir_po45=float(km["po45"]),
        kir_effective_g_proxy=float(km["g_proxy"]),
        kir_effective_g_ratio_to_reference=float(g_ratio),
        kir_conductance_score=float(g_score),
        kir_noise_SD_mV=None if v3_noise is None else float(v3_noise),
        kir_noise_score=float(noise_score),
        kir_buffer_score=float(kir_buffer),
        kir_heteromer_pH_context=float(heteromer_context),
        ionic_homeostasis_score=float(ionic_homeostasis),
        glutamate_clearance_score=float(glutamate_clearance),
        ca_signal_score=float(ca_signal),
        gap_junction_score=float(gap_junction),
        slow_context_score=float(slow_context),
        neural_excitability_score=float(excitability),
        glial_network_score=float(glial_network),
    )


def scores_to_layer_gains(scores: MechanisticScores, hardware_gain: float) -> LayerGains:
    """
    Convert biological mechanism scores to bounded quantum-layer gains.

    V6.6 keeps the validated N->G / shared-glia / G->N mappings, and retains
    astrocyte-astrocyte coupling into two bounded operator gains:
      - glia_glia_phase_gain: ZZ-like correlated/context phase coupling
      - glia_glia_exchange_gain: XY/partial-iSWAP-like state exchange
    Kir-like damping, shared-glia threshold gain, and the optional glial-
    controlled neuronal CSWAP gain are retained.
    """
    g = float(hardware_gain)
    homeo01 = 0.5*(scores.ionic_homeostasis_score + 1.0)
    kir01 = 0.5*(scores.kir_conductance_score + 1.0)
    ca01 = 0.5*(scores.ca_signal_score + 1.0)
    exc01 = 0.5*(scores.neural_excitability_score + 1.0)
    net01 = 0.5*(scores.glial_network_score + 1.0)
    return LayerGains(
        neuronal_input_gain=clip(
            g*(0.65 + 0.30*scores.neural_excitability_score
               - 0.10*scores.ionic_homeostasis_score),
            0.10, 1.40
        ),
        neuronal_phase_gain=clip(
            g*(0.25 + 0.25*scores.neural_excitability_score),
            0.05, 0.80
        ),
        # V6.9 classical excitatory N->N synaptic propagation analogue.
        # Deliberately depends on neuronal excitability rather than astrocyte-
        # specific gap-junction/Ca2+ scores. This is a bounded encoding gain,
        # not a measured synaptic conductance.
        neuron_to_neuron_gain=clip(
            g*(0.40 + 0.35*scores.neural_excitability_score),
            0.05, 1.20
        ),
        glial_local_bias=clip(
            g*(0.12 + 0.18*scores.ca_signal_score
               + 0.08*scores.kir_noise_score),
            -0.45, 0.45
        ),
        glial_phase_gain=clip(
            g*(0.30 + 0.35*scores.ca_signal_score
               + 0.15*scores.kir_heteromer_pH_context),
            -1.20, 1.20
        ),
        neuron_to_glia_gain=clip(
            g*(0.30 + 0.30*scores.ca_signal_score
               + 0.20*scores.neural_excitability_score
               + 0.10*scores.kir_noise_score),
            0.03, 1.20
        ),
        # Phase/context channel is deliberately IDENTICAL to V6.3's
        # glia_glia_gain so ZZ_only_glia_glia is a clean phase-only control.
        glia_glia_phase_gain=clip(
            g*(0.32 + 0.42*scores.glial_network_score
               + 0.12*scores.gap_junction_score),
            0.02, 1.20
        ),
        # State-exchange channel: weighted primarily toward Cx43/Cx30 coupling
        # and slower spatial/homeostatic context. This is an operator analogue,
        # not a measured microscopic quantum rate.
        glia_glia_exchange_gain=clip(
            g*(0.42 + 0.38*scores.gap_junction_score
               + 0.20*scores.slow_context_score),
            0.02, 1.20
        ),
        glia_to_neuron_gain=clip(
            g*(0.28 + 0.22*scores.ca_signal_score
               + 0.20*(1.0 - scores.glutamate_clearance_score)/2.0
               + 0.12*scores.neural_excitability_score),
            0.03, 1.20
        ),
        readout_phase_gain=clip(
            g*(0.18 + 0.18*scores.ca_signal_score),
            0.02, 0.70
        ),
        # Reference-like biology is deliberately around gamma~0.3-0.4 rather than
        # hard reset. KCNJ10 loss / impaired conductance lowers this coefficient.
        kir_damping_gamma=clip(
            g*(0.08 + 0.28*kir01 + 0.16*homeo01 - 0.12*exc01),
            0.01, 0.90
        ),
        # Ca/IP3R2 and excitability set how strongly coincident neuronal controls
        # recruit a shared astrocyte.  Multiplied by pi in the CC-RY operator.
        threshold_gate_gain=clip(
            g*(0.12 + 0.55*ca01 + 0.18*exc01),
            0.03, 1.00
        ),
        # Gap-junction/glial-network context gates an idealized controlled exchange.
        exchange_gate_gain=clip(
            g*(0.10 + 0.72*net01),
            0.02, 1.00
        ),
    )


# =============================================================================
# 2. Network model
# =============================================================================

@dataclass
class NeuralInputEvent:
    layer: int
    neuron: int
    amplitude: float
    phase_rad: float = 0.0
    # V6.8 distinguishes phasic evidence from tonic/context-carrier drive.
    # This tag is used only by activity-gating logic; it does not alter the
    # RY/RZ encoding of the event itself. Old projects default to evidence.
    event_kind: str = "evidence"


@dataclass
class DirectedEdge:
    source: int
    target: int
    weight: float = 1.0


@dataclass
class UndirectedEdge:
    a: int
    b: int
    # Signed phase-coupling weight used by ZZ.
    weight: float = 1.0
    # Non-negative state-exchange weight used by XY. None preserves backward
    # compatibility with V6.3 projects by defaulting to abs(weight).
    exchange_weight: float | None = None


def effective_glia_exchange_weight(edge: UndirectedEdge) -> float:
    if edge.exchange_weight is None:
        return abs(float(edge.weight))
    return float(edge.exchange_weight)


@dataclass
class TripartiteSynapse:
    """
    V6.8 mixed neuron-astrocyte-neuron higher-order interaction.

    Biological topology: active presynaptic/sensory neuron + astrocytic context
    jointly modulate a postsynaptic/decision neuron.  The implementation reuses
    the existing CC-RY primitive; this is a quantum-operator analogue, not a
    claim that a biological tripartite synapse literally implements CC-RY.
    """
    sensory_neuron: int
    context_glia: int
    target_neuron: int
    weight: float = 1.0
    # If true, the operator is recruited only when the sensory neuron receives
    # a phasic evidence event on this layer.  This prevents tonic |+> carriers
    # from masquerading as a new afferent event.
    require_phasic_sensory: bool = True


@dataclass
class NetworkConfig:
    total_qubits: int = 16
    layers: int = 3
    layer_dt_ms: float = 10.0
    neural_inputs: list[NeuralInputEvent] | None = None
    # V6.9 editable directed neuronal synapses. Positive weights are interpreted
    # as excitatory activation strengths in the primary biological use case.
    neuron_to_neuron: list[DirectedEdge] | None = None
    neuron_to_glia: list[DirectedEdge] | None = None
    glia_glia: list[UndirectedEdge] | None = None
    glia_to_neuron: list[DirectedEdge] | None = None
    # V6.6 optional higher-order/open-system operators. Threshold detection is
    # topology-driven: >=2 N inputs sharing one G create a parameterized CC-RY.
    enable_shared_glia_threshold: bool = True
    max_threshold_pairs_per_glia: int = 1
    # V6.8: selected shared-glia microdomains can require a phasic evidence
    # event on at least one convergent neuronal source before the higher-order
    # CC-RY is applied on that layer. This is used for current sensory routing
    # microdomains so tonic/internal context alone does not recruit an inactive
    # sensory pathway. None/[] preserves V6.6 behavior for all microdomains.
    activity_gate_shared_glia_threshold_targets: list[int] | None = None
    # V6.8 active-tripartite-microdomain gate. For selected glial targets, the
    # local N->G inputs, shared-glia CC-RY, and G->N outputs are executed only
    # on layers with phasic evidence arriving on at least one convergent source.
    # Tonic/internal context can modulate an active pathway but cannot by itself
    # create a new sensory-routing event. None/[] preserves legacy behavior.
    activity_gate_glia_microdomain_targets: list[int] | None = None
    # V6.8: direct mixed neuron+astrocyte -> neuron tripartite interactions.
    # These reuse the existing CC-RY primitive with heterogeneous controls.
    tripartite_synapses: list[TripartiteSynapse] | None = None
    enable_tripartite_synapses: bool = True
    enable_controlled_exchange: bool = False
    exchange_activation_threshold: float = 0.75
    # V6.6: if True, each G-G edge applies XY exchange in addition to ZZ phase.
    # Set False to recover the V6.3-like phase-only G-G operator.
    enable_glia_exchange: bool = True
    # Time-discretization correction introduced in V6.6. The biology-derived
    # XY gain is interpreted as the exchange budget for the complete layered
    # context window. If True, each layer receives gain/layers. Setting False
    # exactly recovers the V6.6 per-layer unscaled XY dynamics.
    scale_glia_exchange_by_layers: bool = False
    # V6.6 biology-grounded source-activity gate for net G-G state transfer.
    # Gap-junction topology/ZZ coupling remains present continuously; XY exchange
    # is applied only when at least one endpoint glial microdomain is being
    # recruited by coincident external neuronal inputs on that layer. This
    # models activity/gradient-driven net inter-astrocytic flux, not literal
    # instantaneous opening/closing of connexin channels.
    activity_gate_glia_exchange: bool = True
    enable_kir_damping: bool = False
    # V7.0: clean-state switches. Defaults preserve all V6.9 behavior.
    # Turning these off is useful for proof-of-principle probes where a true
    # |0...0> baseline is required instead of the legacy carrier/background.
    initialize_neural_h: bool = True
    initialize_glial_baseline: bool = True
    # Sample2B: optional history-dependent astrocytic carry-over state.
    # Each value is a computational-basis population P(G_i=1) used to
    # initialize the glial qubit before the layered N->G->G->N circuit.
    # None preserves the original V7.1 initialization exactly.
    glial_initial_probabilities: list[float] | None = None
    # V7.1 separates population/activity readout from the legacy phase-sensitive
    # behavioral readout. `phase` preserves V6.9 behavior; `population` measures
    # the computational basis directly and is the only mode that should feed
    # slow plasticity traces in the evolving-network runner.
    neural_readout_mode: str = "phase"

    @property
    def n_neurons(self) -> int:
        return self.total_qubits // 2

    @property
    def n_glia(self) -> int:
        return self.total_qubits // 2

    @property
    def neuron_qubits(self) -> list[int]:
        return list(range(self.n_neurons))

    @property
    def glia_qubits(self) -> list[int]:
        return list(range(self.n_neurons, self.total_qubits))


def validate_network_config(cfg: NetworkConfig):
    if cfg.total_qubits < 4 or cfg.total_qubits > 128 or cfg.total_qubits % 2:
        raise ValueError("Total qubits must be an even number from 4 to 128.")
    if cfg.layers < 1 or cfg.layers > 64:
        raise ValueError("Layers must be 1..64.")
    nN, nG = cfg.n_neurons, cfg.n_glia
    if cfg.max_threshold_pairs_per_glia < 0 or cfg.max_threshold_pairs_per_glia > 16:
        raise ValueError("max_threshold_pairs_per_glia must be 0..16.")
    if not (0.0 <= cfg.exchange_activation_threshold <= 1.0):
        raise ValueError("exchange_activation_threshold must be 0..1.")
    if str(getattr(cfg, "neural_readout_mode", "phase")).lower() not in {"phase", "population"}:
        raise ValueError("neural_readout_mode must be 'phase' or 'population'.")

    for ev in cfg.neural_inputs or []:
        if not (0 <= ev.layer < cfg.layers):
            raise ValueError(f"Input layer out of range: {ev}")
        if not (0 <= ev.neuron < nN):
            raise ValueError(f"Neuron index out of range: {ev}")
        if not (-4.0 <= ev.amplitude <= 4.0):
            raise ValueError(f"Input amplitude should be -4..4: {ev}")
        if str(getattr(ev, "event_kind", "evidence")) not in {"evidence", "tonic"}:
            raise ValueError(f"event_kind must be evidence or tonic: {ev}")

    for gidx in cfg.activity_gate_shared_glia_threshold_targets or []:
        if not (0 <= int(gidx) < nG):
            raise ValueError(f"Activity-gated shared-glia target out of range: {gidx}")
    for gidx in cfg.activity_gate_glia_microdomain_targets or []:
        if not (0 <= int(gidx) < nG):
            raise ValueError(f"Activity-gated glial microdomain target out of range: {gidx}")

    for t in cfg.tripartite_synapses or []:
        if not (0 <= int(t.sensory_neuron) < nN):
            raise ValueError(f"Tripartite sensory neuron out of range: {t}")
        if not (0 <= int(t.context_glia) < nG):
            raise ValueError(f"Tripartite context glia out of range: {t}")
        if not (0 <= int(t.target_neuron) < nN):
            raise ValueError(f"Tripartite target neuron out of range: {t}")
        if int(t.sensory_neuron) == int(t.target_neuron):
            raise ValueError(f"Tripartite sensory and target neuron must differ: {t}")
        if abs(float(t.weight)) > 4.0:
            raise ValueError(f"Tripartite weight should be -4..4: {t}")

    for e in cfg.neuron_to_neuron or []:
        if not (0 <= e.source < nN and 0 <= e.target < nN):
            raise ValueError(f"N->N edge out of range: {e}")
        if e.source == e.target:
            raise ValueError(f"N->N self-edge is not allowed: {e}")
        if float(e.weight) < 0:
            raise ValueError(
                f"V7.0 primary N->N primitive is excitatory; weight must be >= 0: {e}"
            )

    for e in cfg.neuron_to_glia or []:
        if not (0 <= e.source < nN and 0 <= e.target < nG):
            raise ValueError(f"N->G edge out of range: {e}")
        if abs(e.weight) > 4:
            raise ValueError(f"N->G edge weight should be -4..4: {e}")

    for e in cfg.glia_glia or []:
        if not (0 <= e.a < nG and 0 <= e.b < nG):
            raise ValueError(f"G-G edge out of range: {e}")
        if e.a == e.b:
            raise ValueError(f"G-G self edge is not allowed: {e}")
        if abs(e.weight) > 4:
            raise ValueError(f"G-G phase weight should be -4..4: {e}")
        if e.exchange_weight is not None and not (0.0 <= float(e.exchange_weight) <= 4.0):
            raise ValueError(f"G-G exchange_weight should be 0..4 or null: {e}")

    for e in cfg.glia_to_neuron or []:
        if not (0 <= e.source < nG and 0 <= e.target < nN):
            raise ValueError(f"G->N edge out of range: {e}")
        if abs(e.weight) > 4:
            raise ValueError(f"G->N edge weight should be -4..4: {e}")


def default_network(total_qubits: int = 16, layers: int = 3) -> NetworkConfig:
    n = total_qubits // 2
    inputs = [
        NeuralInputEvent(layer=0, neuron=0, amplitude=1.0, phase_rad=0.0)
    ]
    # One-to-one neural -> glial gates.
    ng = [DirectedEdge(i, i, 1.0) for i in range(n)]
    # Ring astrocyte network.
    gg = []
    for i in range(n):
        j = (i + 1) % n
        if i < j or (i == n - 1 and j == 0):
            gg.append(UndirectedEdge(i, j, 1.0))
    # Glial feedback to corresponding neural node.
    gn = [DirectedEdge(i, i, 0.75) for i in range(n)]
    return NetworkConfig(
        total_qubits=total_qubits,
        layers=layers,
        neural_inputs=inputs,
        neuron_to_glia=ng,
        glia_glia=gg,
        glia_to_neuron=gn,
        enable_shared_glia_threshold=True,
        max_threshold_pairs_per_glia=1,
        activity_gate_shared_glia_threshold_targets=None,
        activity_gate_glia_microdomain_targets=None,
        tripartite_synapses=None,
        enable_tripartite_synapses=True,
        enable_controlled_exchange=False,
        exchange_activation_threshold=0.75,
        enable_glia_exchange=True,
        scale_glia_exchange_by_layers=False,
        activity_gate_glia_exchange=True,
        enable_kir_damping=False,
    )


def build_condition_variants(cfg: NetworkConfig) -> dict[str, NetworkConfig]:
    """Matched architecture ablations."""
    def clone(**updates):
        d = {
            "total_qubits": cfg.total_qubits,
            "layers": cfg.layers,
            "layer_dt_ms": cfg.layer_dt_ms,
            "neural_inputs": list(cfg.neural_inputs or []),
            "neuron_to_neuron": list(cfg.neuron_to_neuron or []),
            "neuron_to_glia": list(cfg.neuron_to_glia or []),
            "glia_glia": list(cfg.glia_glia or []),
            "glia_to_neuron": list(cfg.glia_to_neuron or []),
            "enable_shared_glia_threshold": cfg.enable_shared_glia_threshold,
            "max_threshold_pairs_per_glia": cfg.max_threshold_pairs_per_glia,
            "activity_gate_shared_glia_threshold_targets": list(cfg.activity_gate_shared_glia_threshold_targets or []),
            "activity_gate_glia_microdomain_targets": list(cfg.activity_gate_glia_microdomain_targets or []),
            "tripartite_synapses": list(cfg.tripartite_synapses or []),
            "enable_tripartite_synapses": cfg.enable_tripartite_synapses,
            "enable_controlled_exchange": cfg.enable_controlled_exchange,
            "exchange_activation_threshold": cfg.exchange_activation_threshold,
            "enable_glia_exchange": cfg.enable_glia_exchange,
            "scale_glia_exchange_by_layers": cfg.scale_glia_exchange_by_layers,
            "activity_gate_glia_exchange": cfg.activity_gate_glia_exchange,
            "enable_kir_damping": cfg.enable_kir_damping,
            "initialize_neural_h": getattr(cfg, "initialize_neural_h", True),
            "initialize_glial_baseline": getattr(cfg, "initialize_glial_baseline", True),
            "neural_readout_mode": getattr(cfg, "neural_readout_mode", "phase"),
        }
        d.update(updates)
        return NetworkConfig(**d)

    return {
        # V6.6 primary model: same biology-derived XY angle as V6.4, but net
        # state transfer is recruited only on layers where an association
        # microdomain receives coincident external neuronal evidence.
        "full_N_G_G_N": clone(
            scale_glia_exchange_by_layers=False,
            activity_gate_glia_exchange=True,
        ),
        # Exact V6.4 dynamical control: full XY angle on every G-G exchange edge
        # on every layer, independent of source activity.
        "unscaled_XY_V6_4": clone(
            scale_glia_exchange_by_layers=False,
            activity_gate_glia_exchange=False,
        ),
        # Exact V6.5 dynamical control: whole-window division across all layers.
        "divided_XY_V6_5": clone(
            scale_glia_exchange_by_layers=True,
            activity_gate_glia_exchange=False,
        ),
        # Same G-G topology and ZZ phase, but no XY population/state exchange.
        "ZZ_only_glia_glia": clone(enable_glia_exchange=False),
        "no_neuron_to_neuron": clone(neuron_to_neuron=[]),
        "no_neuron_to_glia": clone(neuron_to_glia=[]),
        "no_glia_glia_network": clone(glia_glia=[]),
        "no_glia_to_neuron": clone(glia_to_neuron=[], enable_tripartite_synapses=False),
        "no_tripartite_synapse": clone(enable_tripartite_synapses=False),
        "neural_only": clone(neuron_to_glia=[], glia_glia=[], glia_to_neuron=[], enable_tripartite_synapses=False),
    }


def shared_glia_input_pairs(cfg: NetworkConfig):
    """Return bounded pairs of N inputs that converge on the same glial node."""
    by_g: dict[int, list[DirectedEdge]] = {}
    for e in cfg.neuron_to_glia or []:
        by_g.setdefault(int(e.target), []).append(e)
    out = []
    limit = max(0, int(cfg.max_threshold_pairs_per_glia))
    for gidx, edges in by_g.items():
        edges = sorted(edges, key=lambda e: abs(float(e.weight)), reverse=True)
        count = 0
        for i in range(len(edges)):
            for j in range(i+1, len(edges)):
                if edges[i].source == edges[j].source:
                    continue
                out.append((gidx, edges[i], edges[j]))
                count += 1
                if count >= limit:
                    break
            if count >= limit:
                break
    return out


def shared_glia_output_pairs(cfg: NetworkConfig):
    """Return first pair of downstream neurons controlled by each shared glial node."""
    by_g: dict[int, list[DirectedEdge]] = {}
    for e in cfg.glia_to_neuron or []:
        by_g.setdefault(int(e.source), []).append(e)
    out = []
    for gidx, edges in by_g.items():
        uniq = []
        for e in sorted(edges, key=lambda e: abs(float(e.weight)), reverse=True):
            if e.target not in [x.target for x in uniq]:
                uniq.append(e)
        if len(uniq) >= 2:
            out.append((gidx, uniq[0], uniq[1]))
    return out


def circuit_resource_estimate(cfg: NetworkConfig) -> dict[str, int | float]:
    n_inputs = len(cfg.neural_inputs or [])
    nn = len(cfg.neuron_to_neuron or [])
    ng = len(cfg.neuron_to_glia or [])
    gg = len(cfg.glia_glia or [])
    gg_exchange = (
        sum(1 for e in (cfg.glia_glia or []) if effective_glia_exchange_weight(e) > 1e-12)
        if cfg.enable_glia_exchange else 0
    )
    gn = len(cfg.glia_to_neuron or [])
    tripartite = len(cfg.tripartite_synapses or []) if cfg.enable_tripartite_synapses else 0
    evidence_by_layer = evidence_neurons_by_layer(cfg)
    tripartite_applications = int(sum(
        1 for layer in range(cfg.layers) for t in (cfg.tripartite_synapses or [])
        if cfg.enable_tripartite_synapses
        and ((not bool(t.require_phasic_sensory))
             or int(t.sensory_neuron) in evidence_by_layer.get(layer, set()))
    ))
    threshold_pair_list = shared_glia_input_pairs(cfg) if cfg.enable_shared_glia_threshold else []
    threshold_pairs = len(threshold_pair_list)
    threshold_applications = int(sum(
        1 for layer in range(cfg.layers) for gidx,e0,e1 in threshold_pair_list
        if glia_microdomain_is_active(cfg, layer, int(gidx), evidence_by_layer)
        and shared_glia_threshold_is_active(cfg, layer, gidx, e0, e1, evidence_by_layer)
    ))
    nn_applications = int(cfg.layers * nn)
    ng_applications = int(sum(
        1 for layer in range(cfg.layers) for e in (cfg.neuron_to_glia or [])
        if glia_microdomain_is_active(cfg, layer, int(e.target), evidence_by_layer)
    ))
    gn_applications = int(sum(
        1 for layer in range(cfg.layers) for e in (cfg.glia_to_neuron or [])
        if glia_microdomain_is_active(cfg, layer, int(e.source), evidence_by_layer)
    ))
    exchange_pairs = len(shared_glia_output_pairs(cfg)) if cfg.enable_controlled_exchange else 0
    damping_pairs = min(cfg.n_neurons, cfg.n_glia) if cfg.enable_kir_damping else 0

    recruited_by_layer = externally_recruited_shared_glia_by_layer(cfg)
    xy_applications = int(
        sum(
            1
            for layer, recruited in recruited_by_layer.items()
            for e in (cfg.glia_glia or [])
            if cfg.enable_glia_exchange
            and effective_glia_exchange_weight(e) > 1e-12
            and (
                (not cfg.activity_gate_glia_exchange)
                or glia_exchange_edge_is_driven(e, recruited)
            )
        )
    )

    # Decompositions: CRY=4 primitive/2 CX; ZZ=3/2 CX.
    # V6.6 XY exchange = RXX + RYY = 18 primitive / 4 CX with the portable
    # H/RZ/CNOT decomposition used here. CCRY = 14 primitive / 8 CX.
    # CSWAP ~=17 primitive / 8 CX; damping dilation = 5 primitive / 3 CX.
    input_gates = 2*n_inputs
    total_primitive = (
        input_gates + 4*ng_applications + cfg.layers*(3*gg + 2*cfg.n_glia + 17*exchange_pairs)
        + 4*gn_applications + 14*threshold_applications + 14*tripartite_applications
        + 18*xy_applications + 2*cfg.n_neurons + 5*damping_pairs
    )
    approx_cnot = (
        2*ng_applications + cfg.layers*(2*gg + 8*exchange_pairs)
        + 2*gn_applications + 8*threshold_applications + 8*tripartite_applications
        + 4*xy_applications + 3*damping_pairs
    )
    return {
        "total_qubits": cfg.total_qubits,
        "neuronal_qubits": cfg.n_neurons,
        "glial_qubits": cfg.n_glia,
        "layers": cfg.layers,
        "input_events": n_inputs,
        "N_to_N_edges": nn,
        "N_to_N_applications": nn_applications,
        "N_to_G_edges": ng,
        "N_to_G_active_edge_layer_applications": ng_applications,
        "activity_gated_glial_microdomain_targets": list(cfg.activity_gate_glia_microdomain_targets or []),
        "G_to_G_edges": gg,
        "G_to_G_XY_exchange_edges": gg_exchange,
        "XY_exchange_time_scaled": bool(cfg.scale_glia_exchange_by_layers),
        "XY_exchange_layer_divisor": int(cfg.layers if cfg.scale_glia_exchange_by_layers else 1),
        "XY_exchange_activity_gated": bool(cfg.activity_gate_glia_exchange),
        "XY_exchange_active_edge_layer_applications": xy_applications,
        "G_to_N_edges": gn,
        "G_to_N_active_edge_layer_applications": gn_applications,
        "shared_glia_threshold_pairs": threshold_pairs,
        "shared_glia_threshold_activity_gated_targets": list(cfg.activity_gate_shared_glia_threshold_targets or []),
        "shared_glia_threshold_active_applications": threshold_applications,
        "tripartite_synapse_edges": tripartite,
        "tripartite_synapse_active_applications": tripartite_applications,
        "controlled_exchange_pairs": exchange_pairs,
        "kir_damping_pairs": damping_pairs,
        "approx_primitive_gates": int(total_primitive),
        "approx_CNOT_count": int(approx_cnot),
    }


# =============================================================================
# 3. QPanda circuit building
# =============================================================================

def import_qpanda_core():
    try:
        import pyqpanda3
        from pyqpanda3.core import QProg, H, RY, RZ, CNOT, measure, CPUQVM
    except ImportError as exc:
        raise RuntimeError(
            "pyqpanda3 is not installed. Run: pip install -U pyqpanda3"
        ) from exc
    return {
        "pyqpanda3": pyqpanda3,
        "QProg": QProg,
        "H": H,
        "RY": RY,
        "RZ": RZ,
        "CNOT": CNOT,
        "measure": measure,
        "CPUQVM": CPUQVM,
    }


def append_cry(program, control: int, target: int, theta: float, qp):
    """
    Controlled-RY decomposition using only RY and CNOT:
        RY(theta/2) - CX - RY(-theta/2) - CX
    Direction is control -> target.
    """
    if abs(theta) < 1e-12:
        return
    program << qp["RY"](target, theta/2.0)
    program << qp["CNOT"](control, target)
    program << qp["RY"](target, -theta/2.0)
    program << qp["CNOT"](control, target)


def append_zz(program, q0: int, q1: int, theta: float, qp):
    """RZZ(theta) = exp[-i theta Z⊗Z / 2] using CNOT-RZ-CNOT."""
    if abs(theta) < 1e-12:
        return
    program << qp["CNOT"](q0, q1)
    program << qp["RZ"](q1, theta)
    program << qp["CNOT"](q0, q1)


def append_xx(program, q0: int, q1: int, theta: float, qp):
    """RXX(theta) = exp[-i theta X⊗X / 2] via H⊗H basis change."""
    if abs(theta) < 1e-12:
        return
    program << qp["H"](q0)
    program << qp["H"](q1)
    append_zz(program, q0, q1, theta, qp)
    program << qp["H"](q0)
    program << qp["H"](q1)


def append_yy(program, q0: int, q1: int, theta: float, qp):
    """RYY(theta) = exp[-i theta Y⊗Y / 2] using S-conjugated RXX."""
    if abs(theta) < 1e-12:
        return
    # The implemented sequence is S† -> RXX -> S; the total unitary is
    # S RXX S†, and S X S† = Y.
    program << qp["RZ"](q0, -math.pi/2.0)
    program << qp["RZ"](q1, -math.pi/2.0)
    append_xx(program, q0, q1, theta, qp)
    program << qp["RZ"](q0, +math.pi/2.0)
    program << qp["RZ"](q1, +math.pi/2.0)


def append_xy_exchange(program, q0: int, q1: int, theta: float, qp):
    """
    XY / partial-iSWAP-like exchange:
        U_XY(theta) = exp[-i theta (X⊗X + Y⊗Y)/2].

    Since XX and YY commute, this is exactly RXX(theta) RYY(theta).
    In the {|01>,|10>} subspace it gives:
        |01> -> cos(theta)|01> - i sin(theta)|10>
        |10> -> cos(theta)|10> - i sin(theta)|01>
    so theta=pi/2 is a full iSWAP-like exchange up to phase convention.
    """
    if abs(theta) < 1e-12:
        return
    append_xx(program, q0, q1, theta, qp)
    append_yy(program, q0, q1, theta, qp)


def append_cz(program, control: int, target: int, qp):
    """CZ using H-CNOT-H."""
    program << qp["H"](target)
    program << qp["CNOT"](control, target)
    program << qp["H"](target)


def append_ccry(program, c0: int, c1: int, target: int, theta: float, qp):
    """
    Parameterized two-control RY.  Using V^2=RY(theta), a controlled-controlled-U
    decomposition is CV(c1)-CX(c0,c1)-CV†(c1)-CX(c0,c1)-CV(c0), V=RY(theta/2).
    At theta=pi the computational-basis truth table reaches the Toffoli limit
    (up to phase on the target branch).
    """
    if abs(theta) < 1e-12:
        return
    append_cry(program, c1, target, theta/2.0, qp)
    program << qp["CNOT"](c0, c1)
    append_cry(program, c1, target, -theta/2.0, qp)
    program << qp["CNOT"](c0, c1)
    append_cry(program, c0, target, theta/2.0, qp)


def append_toffoli(program, c0: int, c1: int, target: int, qp):
    """Exact CCNOT (global phase irrelevant) using H/RZ/CNOT only."""
    H, RZ, CNOT = qp["H"], qp["RZ"], qp["CNOT"]
    t = math.pi / 4.0
    program << H(target)
    program << CNOT(c1, target)
    program << RZ(target, -t)
    program << CNOT(c0, target)
    program << RZ(target, +t)
    program << CNOT(c1, target)
    program << RZ(target, -t)
    program << CNOT(c0, target)
    program << RZ(c1, +t)
    program << RZ(target, +t)
    program << H(target)
    program << CNOT(c0, c1)
    program << RZ(c0, +t)
    program << RZ(c1, -t)
    program << CNOT(c0, c1)


def append_cswap(program, control: int, a: int, b: int, qp):
    """Fredkin/CSWAP = CX(a,b) -> Toffoli(control,b,a) -> CX(a,b)."""
    program << qp["CNOT"](a, b)
    append_toffoli(program, control, b, a, qp)
    program << qp["CNOT"](a, b)


def append_amplitude_damping_dilation(program, system: int, env: int, gamma: float, qp):
    """
    Stinespring dilation of amplitude damping for an environment initialized |0>.
    Exact channel validation uses a clean ancilla.  In full network mode the same
    construction is optional and explicitly interpreted as a coherent Kir-like
    restoration proxy because the glial qubit may already carry contextual state.
    """
    gamma = clip(gamma, 0.0, 1.0)
    theta = 2.0 * math.asin(math.sqrt(gamma))
    append_cry(program, system, env, theta, qp)
    program << qp["CNOT"](env, system)


def input_events_by_layer(cfg: NetworkConfig) -> dict[int, list[NeuralInputEvent]]:
    out = {l: [] for l in range(cfg.layers)}
    for ev in cfg.neural_inputs or []:
        out.setdefault(ev.layer, []).append(ev)
    return out


def evidence_neurons_by_layer(cfg: NetworkConfig) -> dict[int, set[int]]:
    """Phasic evidence events only; tonic carrier drive is excluded."""
    by_layer = input_events_by_layer(cfg)
    out: dict[int, set[int]] = {}
    for layer in range(cfg.layers):
        out[layer] = {
            int(ev.neuron) for ev in by_layer.get(layer, [])
            if abs(float(ev.amplitude)) > 1e-12
            and str(getattr(ev, "event_kind", "evidence")) == "evidence"
        }
    return out


def externally_recruited_shared_glia_by_layer(cfg: NetworkConfig) -> dict[int, set[int]]:
    """
    Identify shared-glia microdomains receiving coincident PHASIC evidence on
    each layer. Tonic/context-carrier events do not count as new source drive.

    Biological interpretation: connexin channels can remain structurally
    available while net intercellular flux depends on a local activity/chemical
    source. Paired sensory + action evidence recruits an association astrocyte.
    """
    evidence = evidence_neurons_by_layer(cfg)
    pairs = shared_glia_input_pairs(cfg)
    out: dict[int, set[int]] = {}
    for layer in range(cfg.layers):
        active_neurons = evidence.get(layer, set())
        recruited = set()
        for gidx, e0, e1 in pairs:
            if int(e0.source) in active_neurons and int(e1.source) in active_neurons:
                recruited.add(int(gidx))
        out[layer] = recruited
    return out


def shared_glia_threshold_is_active(
    cfg: NetworkConfig, layer: int, gidx: int, e0: DirectedEdge, e1: DirectedEdge,
    evidence_by_layer: dict[int, set[int]],
) -> bool:
    """
    V6.8 sensory-event gate for selected higher-order microdomains.

    For targets listed in activity_gate_shared_glia_threshold_targets, at least
    one convergent source must carry a phasic evidence event on this layer. The
    other control may be an internally maintained context representation. This
    models astrocytic modulation of an ACTIVE sensory/synaptic pathway rather
    than treating an inactive |+> carrier as equivalent to a new afferent event.
    Unlisted targets exactly preserve V6.6 shared-glia behavior.
    """
    targets = set(int(x) for x in (cfg.activity_gate_shared_glia_threshold_targets or []))
    if int(gidx) not in targets:
        return True
    active = evidence_by_layer.get(int(layer), set())
    return int(e0.source) in active or int(e1.source) in active


def glia_microdomain_is_active(
    cfg: NetworkConfig, layer: int, gidx: int,
    evidence_by_layer: dict[int, set[int]],
) -> bool:
    """Return whether a selected tripartite microdomain is phasically recruited."""
    targets = set(int(x) for x in (cfg.activity_gate_glia_microdomain_targets or []))
    if int(gidx) not in targets:
        return True
    evidence = evidence_by_layer.get(int(layer), set())
    sources = {int(e.source) for e in (cfg.neuron_to_glia or []) if int(e.target) == int(gidx)}
    return bool(sources & evidence)


def glia_exchange_edge_is_driven(
    edge: UndirectedEdge, recruited_glia: set[int]
) -> bool:
    """True when an XY edge touches a currently recruited source microdomain."""
    return int(edge.a) in recruited_glia or int(edge.b) in recruited_glia


def build_layered_program(
    cfg: NetworkConfig,
    gains: LayerGains,
    qp=None,
    upto_layer: int | None = None,
    measure_glia: bool = False,
    neural_readout_mode: str | None = None,
):
    """
    Build N -> G -> G-network -> N circuit.

    Qubit layout:
        neuronal qubits: 0 .. nN-1
        glial qubits:    nN .. nN+nG-1

    Measurement:
        default = neuronal output only
        optional = all neuronal + glial qubits
    """
    validate_network_config(cfg)
    if qp is None:
        qp = import_qpanda_core()

    prog = qp["QProg"]()
    nN = cfg.n_neurons

    # V7.1 keeps the V6.9 carrier/background as the default, but makes both
    # initializations explicitly switchable for clean operator probes.
    if bool(getattr(cfg, "initialize_neural_h", True)):
        for nq in cfg.neuron_qubits:
            prog << qp["H"](nq)

    glial_init = getattr(cfg, "glial_initial_probabilities", None)
    if glial_init is not None:
        if len(glial_init) != cfg.n_glia:
            raise ValueError(
                f"glial_initial_probabilities must have length {cfg.n_glia}; got {len(glial_init)}"
            )
        for gi, (gq, p1) in enumerate(zip(cfg.glia_qubits, glial_init)):
            p1 = clip(float(p1), 0.0, 1.0)
            theta = 2.0 * math.asin(math.sqrt(p1))
            prog << qp["RY"](gq, theta)
    elif bool(getattr(cfg, "initialize_glial_baseline", True)):
        for gi, gq in enumerate(cfg.glia_qubits):
            local_scale = 1.0 + 0.05*math.sin(2*math.pi*gi/max(1, cfg.n_glia))
            prog << qp["RY"](gq, gains.glial_local_bias*local_scale)
            prog << qp["RZ"](gq, 0.25*gains.glial_phase_gain*local_scale)

    by_layer = input_events_by_layer(cfg)
    evidence_by_layer = evidence_neurons_by_layer(cfg)
    recruited_glia_by_layer = externally_recruited_shared_glia_by_layer(cfg)
    max_layer = cfg.layers if upto_layer is None else min(cfg.layers, int(upto_layer))

    for layer in range(max_layer):
        # ------------------------------------------------------------------
        # A. Neural input layer
        # ------------------------------------------------------------------
        for ev in by_layer.get(layer, []):
            nq = cfg.neuron_qubits[ev.neuron]
            prog << qp["RY"](
                nq,
                clip(ev.amplitude*gains.neuronal_input_gain, -math.pi, math.pi),
            )
            if abs(ev.phase_rad) > 1e-12:
                prog << qp["RZ"](
                    nq,
                    clip(ev.phase_rad*gains.neuronal_phase_gain, -math.pi, math.pi),
                )

        # ------------------------------------------------------------------
        # A2. V6.9 directed neuron -> neuron excitatory synapses
        # ------------------------------------------------------------------
        # Biological analogue: activity in an upstream neuronal population
        # promotes activation of a postsynaptic population through a directed
        # excitatory connection. CRY is an encoding primitive, not a literal
        # microscopic synaptic gate.
        for e in cfg.neuron_to_neuron or []:
            src_q = cfg.neuron_qubits[int(e.source)]
            dst_q = cfg.neuron_qubits[int(e.target)]
            theta_nn = clip(
                float(e.weight)*gains.neuron_to_neuron_gain, 0.0, 1.5
            )
            append_cry(prog, src_q, dst_q, theta_nn, qp)

        # ------------------------------------------------------------------
        # B. Directed neuron -> glia input
        # ------------------------------------------------------------------
        for e in cfg.neuron_to_glia or []:
            if not glia_microdomain_is_active(
                cfg, layer, int(e.target), evidence_by_layer
            ):
                continue
            nq = cfg.neuron_qubits[e.source]
            gq = cfg.glia_qubits[e.target]
            theta = clip(e.weight*gains.neuron_to_glia_gain, -1.5, 1.5)
            append_cry(prog, nq, gq, theta, qp)

        # V6.8 B2. Shared-astrocyte coincidence / threshold operator.
        # If two distinct neurons converge on the same glial node, their joint
        # activity conditionally rotates that glial state.
        if cfg.enable_shared_glia_threshold:
            for gidx, e0, e1 in shared_glia_input_pairs(cfg):
                if not glia_microdomain_is_active(
                    cfg, layer, int(gidx), evidence_by_layer
                ):
                    continue
                if not shared_glia_threshold_is_active(
                    cfg, layer, gidx, e0, e1, evidence_by_layer
                ):
                    continue
                c0 = cfg.neuron_qubits[e0.source]
                c1 = cfg.neuron_qubits[e1.source]
                gq = cfg.glia_qubits[gidx]
                w = min(1.0, math.sqrt(abs(float(e0.weight)*float(e1.weight))))
                theta_thr = clip(math.pi*gains.threshold_gate_gain*w, -math.pi, math.pi)
                append_ccry(prog, c0, c1, gq, theta_thr, qp)

        # ------------------------------------------------------------------
        # C. Glia -> glia network
        # ------------------------------------------------------------------
        # V6.8 hybrid operator:
        #   signed ZZ phase/context coupling
        #   + optional non-negative XY / partial-iSWAP state exchange.
        # The connexin-linked XY capacity is biology-derived. In the V6.6 primary
        # mode, net exchange is SOURCE-ACTIVITY gated: an edge exchanges state
        # only when a shared association glial endpoint receives coincident
        # external neuronal evidence on that layer. ZZ remains continuous.
        # Controls recover V6.4 (ungated/unscaled) and V6.5 (ungated/1-L).
        for e in cfg.glia_glia or []:
            g0 = cfg.glia_qubits[e.a]
            g1 = cfg.glia_qubits[e.b]

            theta_phase = clip(
                float(e.weight)*gains.glia_glia_phase_gain, -1.5, 1.5
            )
            append_zz(prog, g0, g1, theta_phase, qp)

            if cfg.enable_glia_exchange:
                driven = (
                    (not cfg.activity_gate_glia_exchange)
                    or glia_exchange_edge_is_driven(
                        e, recruited_glia_by_layer.get(layer, set())
                    )
                )
                if driven:
                    xw = effective_glia_exchange_weight(e)
                    xy_divisor = (
                        float(cfg.layers)
                        if cfg.scale_glia_exchange_by_layers else 1.0
                    )
                    theta_exchange = clip(
                        xw*gains.glia_glia_exchange_gain/xy_divisor, 0.0, 1.5
                    )
                    append_xy_exchange(prog, g0, g1, theta_exchange, qp)

        # A biology/context-dependent glial phase each layer.
        for gi, gq in enumerate(cfg.glia_qubits):
            local_scale = 1.0 + 0.03*math.cos(
                2*math.pi*(gi + layer)/max(1, cfg.n_glia)
            )
            prog << qp["RZ"](
                gq,
                clip(gains.glial_phase_gain*local_scale/cfg.layers, -1.0, 1.0),
            )

        # ------------------------------------------------------------------
        # D. Directed glia -> neuron feedback
        # ------------------------------------------------------------------
        for e in cfg.glia_to_neuron or []:
            if not glia_microdomain_is_active(
                cfg, layer, int(e.source), evidence_by_layer
            ):
                continue
            gq = cfg.glia_qubits[e.source]
            nq = cfg.neuron_qubits[e.target]
            theta = clip(e.weight*gains.glia_to_neuron_gain, -1.5, 1.5)
            append_cry(prog, gq, nq, theta, qp)

        # ------------------------------------------------------------------
        # D2. V6.8 active tripartite synapse: sensory neuron + astrocytic
        # context jointly modulate a postsynaptic decision neuron.
        # ------------------------------------------------------------------
        # This intentionally reuses the SAME CC-RY primitive already validated
        # for shared-glia coincidence.  Biology changes topology/controls, not
        # the operator vocabulary.  Recruitment can require a phasic sensory
        # event so a tonic carrier state is not mistaken for a new afferent.
        if cfg.enable_tripartite_synapses:
            active_evidence = evidence_by_layer.get(layer, set())
            for t in cfg.tripartite_synapses or []:
                if bool(t.require_phasic_sensory) and int(t.sensory_neuron) not in active_evidence:
                    continue
                sn = cfg.neuron_qubits[int(t.sensory_neuron)]
                cg = cfg.glia_qubits[int(t.context_glia)]
                tn = cfg.neuron_qubits[int(t.target_neuron)]
                theta_tri = clip(
                    math.pi * gains.threshold_gate_gain * float(t.weight),
                    -math.pi, math.pi
                )
                append_ccry(prog, sn, cg, tn, theta_tri, qp)

        # V6.6 D3. Optional idealized glial-controlled exchange.  Kept OFF by
        # default; it only engages when topology supplies two G->N targets and
        # the biology-derived exchange gain crosses the configured threshold.
        if (cfg.enable_controlled_exchange and
                gains.exchange_gate_gain >= cfg.exchange_activation_threshold):
            for gidx, ea, eb in shared_glia_output_pairs(cfg):
                gq = cfg.glia_qubits[gidx]
                na = cfg.neuron_qubits[ea.target]
                nb = cfg.neuron_qubits[eb.target]
                append_cswap(prog, gq, na, nb, qp)

    # Optional Kir-like restoration dilation once before readout.  This is an
    # exact amplitude-damping dilation only when the glial/environment qubit is
    # initialized |0>; because full-network glia may carry state, network mode
    # reports it as a coherent open-system proxy. Exact CPTP behavior is checked
    # independently by the built-in gate validator.
    if cfg.enable_kir_damping:
        for i in range(min(cfg.n_neurons, cfg.n_glia)):
            append_amplitude_damping_dilation(
                prog, cfg.neuron_qubits[i], cfg.glia_qubits[i],
                gains.kir_damping_gamma, qp
            )

    # V7.1 dual neural readout.
    #   phase      = legacy V6.9 RZ+H behavioral / interference-sensitive readout
    #   population = direct Z-basis population proxy; no terminal rotation.
    # Plasticity/slow traces must use population mode so |0> remains P_active=0.
    readout_mode = str(
        neural_readout_mode if neural_readout_mode is not None
        else getattr(cfg, "neural_readout_mode", "phase")
    ).lower()
    if readout_mode not in {"phase", "population"}:
        raise ValueError(f"Unknown neural_readout_mode: {readout_mode}")
    if readout_mode == "phase":
        for nq in cfg.neuron_qubits:
            prog << qp["RZ"](nq, gains.readout_phase_gain)
            prog << qp["H"](nq)

    if measure_glia:
        qubits = list(range(cfg.total_qubits))
        cbits = list(range(cfg.total_qubits))
    else:
        qubits = cfg.neuron_qubits
        cbits = list(range(cfg.n_neurons))
    prog << qp["measure"](qubits, cbits)
    return prog


# =============================================================================
# 4. Result parsing / metrics
# =============================================================================

def parse_basis_key(key: Any) -> int:
    text = str(key).strip().lower()
    if text.startswith("0x"):
        return int(text, 16)
    if text.startswith("0b"):
        return int(text, 2)
    if text and all(c in "01" for c in text):
        return int(text, 2)
    return int(text)


def normalized_sparse_distribution(raw: dict[Any, Any], measured_bits: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for key, value in dict(raw).items():
        idx = parse_basis_key(key)
        if idx >= (1 << measured_bits):
            # Keep only measured classical bits if a backend returns a wider key.
            idx &= (1 << measured_bits) - 1
        out[idx] = out.get(idx, 0.0) + float(value)
    total = sum(out.values())
    if total <= 0:
        raise RuntimeError(f"Cannot normalize QPU result: {raw}")
    return {k: v/total for k, v in out.items()}


def sparse_js_bits(a: dict[int, float], b: dict[int, float]) -> float:
    keys = set(a) | set(b)
    eps = 1e-15
    result = 0.0
    for k in keys:
        p = float(a.get(k, 0.0))
        q = float(b.get(k, 0.0))
        m = 0.5*(p+q)
        if p > 0:
            result += 0.5*p*math.log2((p+eps)/(m+eps))
        if q > 0:
            result += 0.5*q*math.log2((q+eps)/(m+eps))
    return float(result)


def sparse_tv(a: dict[int, float], b: dict[int, float]) -> float:
    keys = set(a) | set(b)
    return float(0.5*sum(abs(a.get(k, 0)-b.get(k, 0)) for k in keys))


def neuron_marginals(dist: dict[int, float], n_neurons: int) -> np.ndarray:
    p = np.zeros(n_neurons, dtype=float)
    for idx, prob in dist.items():
        for q in range(n_neurons):
            if (idx >> q) & 1:
                p[q] += prob
    return p


def activation_count_distribution(
    dist: dict[int, float], n_neurons: int
) -> np.ndarray:
    out = np.zeros(n_neurons+1, dtype=float)
    for idx, prob in dist.items():
        count = int(idx).bit_count()
        if count <= n_neurons:
            out[count] += prob
    return out


def sparse_entropy(dist: dict[int, float]) -> float:
    return float(-sum(p*math.log2(p) for p in dist.values() if p > 0))


def dist_to_rows(
    condition: str,
    repeat: int,
    dist: dict[int, float],
    n_neurons: int,
):
    return [
        {
            "condition": condition,
            "repeat": repeat,
            "state_int": int(idx),
            "state_binary": f"{idx:0{n_neurons}b}",
            "probability": float(prob),
        }
        for idx, prob in sorted(dist.items())
    ]


def summarize_distributions(
    mean_dists: dict[str, dict[int, float]],
    n_neurons: int,
    output_dir: Path,
    prefix: str,
):
    full = mean_dists["full_N_G_G_N"]
    rows = []
    marginal_rows = []
    for name, dist in mean_dists.items():
        marg = neuron_marginals(dist, n_neurons)
        act = activation_count_distribution(dist, n_neurons)
        rows.append({
            "condition": name,
            "entropy_bits": sparse_entropy(dist),
            "JS_vs_full_bits": sparse_js_bits(dist, full),
            "TV_vs_full": sparse_tv(dist, full),
            "mean_active_neurons": float(
                sum(i*act[i] for i in range(len(act)))
            ),
            "P_zero_active": float(act[0]),
            "P_half_or_more_active": float(
                act[math.ceil(n_neurons/2):].sum()
            ),
        })
        for i, p in enumerate(marg):
            marginal_rows.append({
                "condition": name,
                "neuron": i,
                "P_active": float(p),
            })

    summary = pd.DataFrame(rows)
    marg_df = pd.DataFrame(marginal_rows)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    marg_df.to_csv(output_dir / f"{prefix}_neuron_marginals.csv", index=False)

    # Marginal probability plot.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(n_neurons)
    for name in mean_dists:
        d = marg_df[marg_df["condition"] == name]
        ax.plot(x, d["P_active"].to_numpy(float), marker="o", label=name)
    ax.set_xlabel("Neuronal output qubit")
    ax.set_ylabel("Marginal P(N_i = 1)")
    ax.set_title(f"{VERSION} neuronal-output marginals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_neuronal_marginals.png", dpi=240)
    plt.close(fig)

    # Matched ablation bars.
    ablations = [
        "unscaled_XY_V6_4",
        "ZZ_only_glia_glia",
        "no_neuron_to_glia",
        "no_glia_glia_network",
        "no_glia_to_neuron",
        "neural_only",
    ]
    vals = [sparse_js_bits(full, mean_dists[n]) for n in ablations]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(np.arange(len(ablations)), vals)
    ax.set_xticks(np.arange(len(ablations)), ablations, rotation=24, ha="right")
    ax.set_ylabel("JS divergence from full N→G→G→N (bits)")
    ax.set_title(f"{VERSION} matched architecture ablations")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_matched_ablations.png", dpi=240)
    plt.close(fig)

    return summary, marg_df


# =============================================================================
# 5. Local simulation
# =============================================================================

def run_local(
    variants: dict[str, NetworkConfig],
    gains: LayerGains,
    shots: int,
    max_qubits: int,
    output_dir: Path,
    log: Callable[[str], None],
    measure_glia: bool = False,
    neural_readout_mode: str | None = None,
):
    total_q = next(iter(variants.values())).total_qubits
    if total_q > max_qubits:
        raise RuntimeError(
            f"Local CPUQVM blocked at {total_q} qubits. Safe limit is {max_qubits}. "
            "Use Build/resource check or the real QPU for larger circuits."
        )

    qp = import_qpanda_core()
    qvm = qp["CPUQVM"]()
    nN = total_q // 2
    measured_bits = total_q if measure_glia else nN
    mean_dists = {}
    raw_rows = []

    for i, (name, cfg) in enumerate(variants.items(), start=1):
        log(f"[CPUQVM] {i}/{len(variants)} {name}")
        prog = build_layered_program(cfg, gains, qp=qp, measure_glia=measure_glia, neural_readout_mode=neural_readout_mode)
        qvm.run(prog, shots=int(shots))
        counts = qvm.result().get_counts()
        dist = normalized_sparse_distribution(counts, measured_bits)
        mean_dists[name] = dist
        raw_rows.extend(dist_to_rows(name, 0, dist, measured_bits))

    pd.DataFrame(raw_rows).to_csv(
        output_dir / "V6_4_simulator_sparse_probabilities.csv", index=False
    )
    return mean_dists


# =============================================================================
# 6. OriginQ real-QPU runner
# =============================================================================

@dataclass
class HardwareSettings:
    backend: str = "auto"
    shots: int = 1000
    repeats: int = 3
    # V6.2.1 defaults are deliberately conservative for >=16-qubit real-QPU work.
    # Five matched architecture conditions are batched together by default, so one
    # repeat normally occupies one cloud job / one scheduler allocation.
    batch_size: int = 5
    parallel_jobs: int = 1
    mapping: bool = True
    optimization: bool = True
    amend: bool = True
    # Specified physical blocks worked well for small circuits, but large blocks
    # can be rejected by the cloud scheduler with "No available aio ... logic
    # qubit-block".  Therefore automatic mapping is the safe default.
    use_specified_blocks: bool = False
    prefer_disjoint_blocks: bool = False
    # Scheduler-capacity failures can be transient.  V6.2.1 retries a failed
    # batch, first without any specified block and then with bounded backoff.
    scheduler_retries: int = 3
    scheduler_retry_wait_s: int = 30
    # For >=16 qubits, keep one large job in flight unless explicitly overridden.
    force_parallel_large_qpu: bool = False
    local_safe_qubits: int = 20


def import_qcloud():
    try:
        from pyqpanda3.qcloud import (
            QCloudService, QCloudOptions, JobStatus, DataBase
        )
    except ImportError as exc:
        raise RuntimeError(
            "pyqpanda3/qcloud unavailable. Run: pip install -U pyqpanda3"
        ) from exc
    return {
        "QCloudService": QCloudService,
        "QCloudOptions": QCloudOptions,
        "JobStatus": JobStatus,
        "DataBase": DataBase,
    }


def get_api_key() -> str:
    key = os.environ.get("QPANDA_QCLOUD_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "QPANDA_QCLOUD_API_KEY is not set.\n"
            "PowerShell: $env:QPANDA_QCLOUD_API_KEY=\"YOUR_API_KEY\"\n"
            "macOS/Linux: export QPANDA_QCLOUD_API_KEY=\"YOUR_API_KEY\""
        )
    return key


def is_simulator_backend(name: str) -> bool:
    t = str(name).lower()
    return any(
        token in t
        for token in ("amplitude", "simulator", "noise", "density", "stabilizer")
    )


def choose_real_backend(service, requested: str, required_qubits: int, log):
    visible = service.backends()
    log("Visible OriginQ backends:")
    for name, available in visible.items():
        log(f"  {name}: {'available' if available else 'unavailable'}")

    if requested and requested != "auto":
        if requested not in visible:
            raise RuntimeError(f"Backend {requested!r} is not visible.")
        if not visible[requested]:
            raise RuntimeError(f"Backend {requested!r} is unavailable.")
        backend = service.backend(requested)
        try:
            info = backend.chip_info()
            if len(list(info.available_qubits())) < required_qubits:
                raise RuntimeError(
                    f"{requested} currently has fewer than {required_qubits} available qubits."
                )
        except AttributeError:
            pass
        return requested, backend

    preferred = ["WK_C180", "WK_C180_2", "WK_C102_400", "72"]
    for name in preferred:
        if visible.get(name, False) and not is_simulator_backend(name):
            try:
                b = service.backend(name)
                info = b.chip_info()
                if len(list(info.available_qubits())) >= required_qubits:
                    return name, b
            except Exception:
                continue

    candidates = []
    for name, available in visible.items():
        if not available or is_simulator_backend(name):
            continue
        try:
            b = service.backend(name)
            info = b.chip_info()
            n = len(list(info.available_qubits()))
            if n >= required_qubits:
                candidates.append((n, name, b))
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            f"No available physical backend with >= {required_qubits} qubits."
        )
    candidates.sort(reverse=True, key=lambda x: x[0])
    _, name, backend = candidates[0]
    return name, backend


def make_cloud_options(settings: HardwareSettings, block, qcloud):
    options = qcloud["QCloudOptions"]()
    options.set_mapping(bool(settings.mapping))
    options.set_optimization(bool(settings.optimization))
    options.set_amend(bool(settings.amend))
    if block and hasattr(options, "set_specified_block"):
        options.set_specified_block(block)
    return options


def choose_physical_block_pool(
    backend,
    total_qubits: int,
    settings: HardwareSettings,
    log,
):
    """Return physical blocks only when explicitly requested.

    For V6.2.1 the default is automatic mapping.  Large specified blocks can
    conflict with scheduler-side logic-block allocation even when the chip has
    enough nominally available qubits.
    """
    if not settings.use_specified_blocks:
        log(
            "Specified physical blocks disabled (V6.2.1 robust mode); "
            "cloud automatic mapping will be used."
        )
        return [None]

    requested = max(1, settings.parallel_jobs)
    if not hasattr(backend, "best_qubit_blocks"):
        log("best_qubit_blocks() unavailable; automatic mapping will be used.")
        return [None]

    try:
        blocks = backend.best_qubit_blocks(
            qubit_num=total_qubits,
            label=1,
            qubit_block_num=requested,
        )
        parsed = [list(map(int, b)) for b in blocks if b]
        if not parsed:
            log(
                f"No {total_qubits}-qubit physical block returned; "
                "falling back to automatic mapping."
            )
            return [None]
        if settings.prefer_disjoint_blocks:
            disjoint = []
            used = set()
            for block in parsed:
                bs = set(block)
                if not (used & bs):
                    disjoint.append(block)
                    used.update(bs)
            if disjoint:
                parsed = disjoint
        return parsed
    except Exception as exc:
        log(f"best_qubit_blocks() failed: {exc}; using automatic mapping.")
        return [None]


def scheduler_capacity_error(exc: Exception) -> bool:
    text = str(exc).lower()
    tokens = (
        "no available aio",
        "logic qubit-block",
        "logic qubit block",
        "no available qubit-block",
        "no available qubit block",
    )
    return any(t in text for t in tokens)

def parse_cloud_batch_result(result, expected_count: int, qcloud, measured_bits: int):
    if hasattr(result, "job_status"):
        try:
            if result.job_status() == qcloud["JobStatus"].FAILED:
                msg = result.error_message() if hasattr(result, "error_message") else ""
                raise RuntimeError(f"OriginQ cloud task failed: {msg}")
        except TypeError:
            pass

    raw_list = None
    last_exc = None
    attempts = [
        lambda: result.get_probs_list(base=qcloud["DataBase"].Binary),
        lambda: result.get_probs_list(),
        lambda: result.get_counts_list(base=qcloud["DataBase"].Binary),
        lambda: result.get_counts_list(),
    ]
    for attempt in attempts:
        try:
            raw_list = attempt()
            break
        except Exception as exc:
            last_exc = exc
    if raw_list is None:
        raise RuntimeError(f"Cannot parse cloud result: {last_exc}")
    if len(raw_list) != expected_count:
        raise RuntimeError(
            f"Cloud returned {len(raw_list)} results for {expected_count} programs."
        )
    return [
        normalized_sparse_distribution(item, measured_bits)
        for item in raw_list
    ]


def chunked(items, size):
    for start in range(0, len(items), size):
        yield items[start:start+size]


def atomic_json_write(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8"
    )
    tmp.replace(path)


def build_qpu_records(variants, gains, repeats: int, qp, measure_glia: bool = False, neural_readout_mode: str | None = None):
    records = []
    # Interleave matched ablations within every repeat to reduce calibration drift.
    for rep in range(int(repeats)):
        for name, cfg in variants.items():
            records.append({
                "repeat": rep,
                "condition": name,
                "program": build_layered_program(cfg, gains, qp=qp, measure_glia=measure_glia, neural_readout_mode=neural_readout_mode),
                "measured_bits": cfg.total_qubits if measure_glia else cfg.n_neurons,
                "logical_qubits": cfg.total_qubits,
            })
    return records


def submit_programs_in_waves(
    backend,
    records,
    block_pool,
    settings: HardwareSettings,
    qcloud,
    output_dir: Path,
    log,
):
    batches = list(chunked(records, settings.batch_size))
    total_batches = len(batches)
    measured_bits_global = int(records[0]["measured_bits"]) if records else 0
    total_logic_qubits = int(records[0].get("logical_qubits", 2 * measured_bits_global)) if records else 0

    workers = min(settings.parallel_jobs, max(1, total_batches))
    if (
        total_logic_qubits >= 16
        and not settings.force_parallel_large_qpu
        and workers > 1
    ):
        log(
            f"V6.2.1 robust scheduler: {total_logic_qubits}-qubit task -> "
            "forcing max_in_flight=1 to avoid competing large logic-block allocations."
        )
        workers = 1

    checkpoint = output_dir / "job_ids.json"
    results_by_batch = {}
    job_records = []

    log(
        f"Cloud: circuits={len(records)}, batches={total_batches}, "
        f"batch_size={settings.batch_size}, max_in_flight={workers}, "
        f"specified_blocks={settings.use_specified_blocks}"
    )

    def write_record(rec):
        job_records.append(rec)
        atomic_json_write(checkpoint, job_records)

    def mark_record(rec, **updates):
        rec.update(updates)
        atomic_json_write(checkpoint, job_records)

    def submit_once(batch_index, batch_records, block, attempt, retry_of=None):
        programs = [r["program"] for r in batch_records]
        measured_bits = int(batch_records[0]["measured_bits"])
        if any(int(r["measured_bits"]) != measured_bits for r in batch_records):
            raise RuntimeError("Mixed measured-bit widths inside one cloud batch.")

        options = make_cloud_options(settings, block, qcloud)
        log(
            f"Submitting batch {batch_index}/{total_batches}, attempt={attempt}: "
            f"programs={len(programs)}, block={block if block else 'auto'}"
        )
        job = backend.run(programs, int(settings.shots), options)
        rec = {
            "batch": batch_index,
            "attempt": attempt,
            "retry_of_job_id": retry_of,
            "job_id": job.job_id(),
            "program_count": len(programs),
            "physical_block": block,
            "status": "submitted",
            "submitted_at": datetime.now().isoformat(timespec="seconds"),
            "circuits": [
                {
                    "repeat": r["repeat"],
                    "condition": r["condition"],
                    "measured_bits": r["measured_bits"],
                }
                for r in batch_records
            ],
        }
        write_record(rec)
        log(f"  job_id={job.job_id()}")
        return job, rec, measured_bits

    def wait_and_parse(job, rec, batch_records, measured_bits):
        try:
            result = job.result()
            parsed = parse_cloud_batch_result(
                result, len(batch_records), qcloud, measured_bits
            )
            updates = {
                "status": "finished",
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
            if hasattr(result, "timing_info"):
                try:
                    updates["timing_info"] = result.timing_info()
                except Exception:
                    pass
            mark_record(rec, **updates)
            return parsed
        except Exception as exc:
            mark_record(
                rec,
                status="failed",
                failed_at=datetime.now().isoformat(timespec="seconds"),
                error=str(exc),
            )
            raise

    for wave_start in range(0, total_batches, workers):
        wave_indices = list(
            range(wave_start, min(wave_start + workers, total_batches))
        )
        submitted = []

        # Submit all jobs in the wave before blocking on results.
        for zidx in wave_indices:
            bidx = zidx + 1
            batch_records = batches[zidx]
            block = block_pool[zidx % len(block_pool)] if block_pool else None
            try:
                job, rec, measured_bits = submit_once(
                    bidx, batch_records, block, attempt=1
                )
            except Exception as exc:
                if not scheduler_capacity_error(exc):
                    raise
                # Submission itself can occasionally be rejected before a job ID exists.
                log(
                    "Scheduler rejected the initial submission before a job was created: "
                    f"{exc}"
                )
                job = rec = measured_bits = None
            submitted.append((bidx, batch_records, block, job, rec, measured_bits))

        # Preserve parallel waiting for small-QPU mode. Large-QPU robust mode is
        # normally workers=1, which makes retry behavior deterministic.
        live = [x for x in submitted if x[3] is not None]
        completed_errors = {}
        if live:
            with ThreadPoolExecutor(max_workers=max(1, len(live))) as pool:
                futures = {
                    pool.submit(wait_and_parse, job, rec, batch_records, measured_bits):
                    (bidx, batch_records, block, job, rec, measured_bits)
                    for bidx, batch_records, block, job, rec, measured_bits in live
                }
                for future in as_completed(futures):
                    bidx, batch_records, block, job, rec, measured_bits = futures[future]
                    try:
                        results_by_batch[bidx] = future.result()
                        log(f"Batch {bidx}/{total_batches} finished.")
                    except Exception as exc:
                        completed_errors[bidx] = (
                            exc, batch_records, block, job, rec, measured_bits
                        )

        # Jobs rejected before creation also enter the retry path.
        for bidx, batch_records, block, job, rec, measured_bits in submitted:
            if job is None:
                completed_errors[bidx] = (
                    RuntimeError(
                        "Cloud submission was rejected before job creation; "
                        "treating as scheduler-capacity failure."
                    ),
                    batch_records,
                    block,
                    job,
                    rec,
                    int(batch_records[0]["measured_bits"]),
                )

        # Retry only scheduler-capacity / logic-block failures. All retries use
        # automatic mapping and run sequentially, avoiding another block conflict.
        for bidx in sorted(completed_errors):
            exc, batch_records, original_block, original_job, original_rec, measured_bits = (
                completed_errors[bidx]
            )
            if not scheduler_capacity_error(exc):
                raise exc

            log(
                f"Batch {bidx}: OriginQ scheduler reports unavailable logic-qubit block/AIO."
            )
            log(
                "V6.2.1 will retry with automatic mapping, one large cloud job at a time."
            )

            last_exc = exc
            retry_of = original_job.job_id() if original_job is not None else None
            success = False
            for retry in range(1, int(settings.scheduler_retries) + 1):
                # If the failed task used a specified block, the first retry drops
                # that block immediately. Otherwise use bounded backoff because the
                # cloud resource may simply be temporarily occupied.
                if original_block is not None and retry == 1:
                    wait_s = 0
                else:
                    wait_s = max(0, int(settings.scheduler_retry_wait_s)) * retry
                if wait_s:
                    log(
                        f"  scheduler retry {retry}/{settings.scheduler_retries} "
                        f"after {wait_s}s..."
                    )
                    time.sleep(wait_s)
                else:
                    log(
                        f"  scheduler retry {retry}/{settings.scheduler_retries} "
                        "immediately with automatic mapping..."
                    )

                try:
                    retry_job, retry_rec, measured_bits = submit_once(
                        bidx,
                        batch_records,
                        block=None,
                        attempt=retry + 1,
                        retry_of=retry_of,
                    )
                    parsed = wait_and_parse(
                        retry_job, retry_rec, batch_records, measured_bits
                    )
                    results_by_batch[bidx] = parsed
                    log(
                        f"Batch {bidx}/{total_batches} finished on scheduler retry {retry}."
                    )
                    success = True
                    break
                except Exception as retry_exc:
                    last_exc = retry_exc
                    retry_of = (
                        retry_job.job_id()
                        if 'retry_job' in locals() and retry_job is not None
                        else retry_of
                    )
                    if not scheduler_capacity_error(retry_exc):
                        raise
                    log(f"  retry {retry} still unavailable: {retry_exc}")

            if not success:
                raise RuntimeError(
                    "OriginQ currently cannot allocate the required "
                    f"{total_logic_qubits}-qubit logic block after "
                    f"{settings.scheduler_retries} retries. This is a cloud/hardware "
                    "scheduler-capacity condition, not a biological-model error. "
                    "Try the same project later, reduce total qubits, or choose another "
                    "currently available backend. Last error: " + str(last_exc)
                )

    rows = []
    repeat_dists = {}
    for bidx in range(1, total_batches + 1):
        metas = batches[bidx - 1]
        vals = results_by_batch[bidx]
        for meta, dist in zip(metas, vals):
            key = (int(meta["repeat"]), str(meta["condition"]))
            repeat_dists[key] = dist
            rows.extend(
                dist_to_rows(
                    meta["condition"], meta["repeat"], dist, meta["measured_bits"]
                )
            )
    return repeat_dists, rows

def mean_sparse_distributions(repeat_dists, condition_names):
    out = {}
    for cond in condition_names:
        ds = [d for (rep, c), d in repeat_dists.items() if c == cond]
        keys = set()
        for d in ds:
            keys |= set(d)
        mean = {
            k: float(np.mean([d.get(k, 0.0) for d in ds]))
            for k in keys
        }
        total = sum(mean.values())
        if total > 0:
            mean = {k: v/total for k, v in mean.items()}
        out[cond] = mean
    return out


def run_qpu(
    variants,
    gains,
    settings: HardwareSettings,
    output_dir: Path,
    log,
    measure_glia: bool = False,
    neural_readout_mode: str | None = None,
):
    qp = import_qpanda_core()
    qcloud = import_qcloud()
    key = get_api_key()
    QCloudService = qcloud["QCloudService"]
    try:
        service = QCloudService(api_key=key)
    except TypeError:
        service = QCloudService(key)

    total_q = next(iter(variants.values())).total_qubits

    # Robust large-QPU scheduling: keep all matched architecture conditions from
    # one repeat inside the same cloud batch, and avoid multiple simultaneous
    # large logic-block allocations.  This directly addresses the scheduler
    # failure seen in V6.2: "No available aio for task needed logic qubit-block".
    effective_settings = settings
    if total_q >= 16 and not settings.force_parallel_large_qpu:
        matched_conditions = max(1, len(variants))
        effective_settings = replace(
            settings,
            batch_size=matched_conditions,
            parallel_jobs=1,
            use_specified_blocks=False,
            prefer_disjoint_blocks=False,
        )
        log(
            f"V6.2.1 robust large-QPU mode: batch_size={matched_conditions}, "
            "max_in_flight=1, specified_blocks=OFF."
        )

    request = os.environ.get(
        "QPANDA_QCLOUD_BACKEND", effective_settings.backend or "auto"
    ).strip() or "auto"
    backend_name, backend = choose_real_backend(
        service, request, total_q, log
    )
    log(f"Selected real QPU: {backend_name}")

    cloud_report = {
        "backend": backend_name,
        "total_qubits_requested": total_q,
        "requested_settings": asdict(settings),
        "effective_settings": asdict(effective_settings),
    }
    try:
        info = backend.chip_info()
        cloud_report.update({
            "chip_qubits": int(info.qubits_num()),
            "available_qubits": [int(x) for x in list(info.available_qubits())],
            "basic_gates": [str(x) for x in info.get_basic_gates()],
        })
        log(
            f"Chip total={cloud_report['chip_qubits']}; "
            f"available={len(cloud_report['available_qubits'])}"
        )
    except Exception as exc:
        cloud_report["chip_info_error"] = str(exc)

    blocks = choose_physical_block_pool(backend, total_q, effective_settings, log)
    cloud_report["physical_blocks"] = blocks
    atomic_json_write(output_dir / "cloud_report.json", cloud_report)

    records = build_qpu_records(variants, gains, effective_settings.repeats, qp, measure_glia=measure_glia, neural_readout_mode=neural_readout_mode)
    log(
        f"QPU circuits={len(records)}; shots/circuit={effective_settings.shots}; "
        f"repeats={effective_settings.repeats}"
    )
    repeat_dists, raw_rows = submit_programs_in_waves(
        backend, records, blocks, effective_settings, qcloud, output_dir, log
    )
    pd.DataFrame(raw_rows).to_csv(
        output_dir / "V6_4_QPU_sparse_probabilities.csv", index=False
    )
    mean_dists = mean_sparse_distributions(repeat_dists, list(variants))
    return mean_dists, repeat_dists, cloud_report


# =============================================================================
# 7. GUI helpers
# =============================================================================

class EditableTree:
    """Small edge-list editor built on ttk.Treeview."""

    def __init__(self, parent, columns, headings, widths=None):
        from tkinter import ttk
        self.columns = list(columns)
        self.tree = ttk.Treeview(
            parent, columns=self.columns, show="headings", height=14
        )
        widths = widths or [90]*len(columns)
        for c, h, w in zip(self.columns, headings, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill="both", expand=True)

    def clear(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def add(self, values):
        self.tree.insert("", "end", values=list(values))

    def selected_values(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return list(self.tree.item(sel[0], "values"))

    def delete_selected(self):
        for item in self.tree.selection():
            self.tree.delete(item)

    def rows(self):
        return [list(self.tree.item(i, "values")) for i in self.tree.get_children()]


# =============================================================================
# 8. GUI application
# =============================================================================

class V64App:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = tk.Tk()
        self.root.title("V7.0 Neuroglial Gate QPU")
        self.root.geometry("1240x900")
        self.root.minsize(1050, 760)

        # Biological input
        self.protein_vars = {
            g: tk.StringVar(value=str(REFERENCE_PROTEINS[g]))
            for g in PROTEIN_INFO
        }
        self.scale_mode = tk.StringVar(value="relative_linear")
        self.ph_var = tk.StringVar(value="7.4")
        self.k_var = tk.StringVar(value="3.5")
        self.vm_var = tk.StringVar(value="-80")
        self.drive_var = tk.StringVar(value="1.0")
        self.gain_var = tk.StringVar(value="1.0")
        self.v3_path_var = tk.StringVar(value="")

        # Network / hardware
        self.total_qubits_var = tk.StringVar(value="16")
        self.layers_var = tk.StringVar(value="3")
        self.layer_dt_var = tk.StringVar(value="10.0")
        self.backend_var = tk.StringVar(
            value=os.environ.get("QPANDA_QCLOUD_BACKEND", "auto")
        )
        self.qpu_shots_var = tk.StringVar(value="1000")
        self.qpu_repeats_var = tk.StringVar(value="3")
        self.batch_size_var = tk.StringVar(value="5")
        self.parallel_jobs_var = tk.StringVar(value="1")
        self.local_shots_var = tk.StringVar(value="20000")
        self.local_safe_var = tk.StringVar(value="20")
        self.mapping_var = tk.BooleanVar(value=True)
        self.optimization_var = tk.BooleanVar(value=True)
        self.amend_var = tk.BooleanVar(value=True)
        self.use_specified_blocks_var = tk.BooleanVar(value=False)
        self.force_parallel_large_var = tk.BooleanVar(value=False)
        self.scheduler_retries_var = tk.StringVar(value="3")
        self.scheduler_retry_wait_var = tk.StringVar(value="30")

        # V6.6 operator switches
        self.enable_threshold_var = tk.BooleanVar(value=True)
        self.threshold_pairs_var = tk.StringVar(value="1")
        self.enable_exchange_var = tk.BooleanVar(value=False)
        self.exchange_threshold_var = tk.StringVar(value="0.75")
        self.enable_glia_exchange_var = tk.BooleanVar(value=True)
        self.enable_damping_var = tk.BooleanVar(value=False)
        # V7.0 clean-state switches. Defaults preserve V6.9 behavior.
        self.initialize_neural_h_var = tk.BooleanVar(value=True)
        self.initialize_glial_baseline_var = tk.BooleanVar(value=True)
        self.neural_readout_mode_var = tk.StringVar(value="phase")
        self.kir_eta_var = tk.StringVar(value="0.75")

        self.preview_vars = {}
        self._build()
        self.reset_network_defaults()
        self.update_preview()

    # ---------------------------------------------------------------------
    # UI construction
    # ---------------------------------------------------------------------

    def _build(self):
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="V7.1  Evolving Neuroglial Gate QPU",
            font=("Segoe UI", 17, "bold")
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text=(
                "默认 16 qubits = 8 neurons + 8 glia；可扩展至 128。"
                " 通过边列表直接编辑 N→N、N→G、G↔G、G→N；"
                "神经输入可按 layer / neuron / amplitude / phase 编排。"
            ),
            wraplength=1160,
        ).pack(anchor="w", pady=(2, 8))

        nb = ttk.Notebook(outer)
        nb.pack(fill="both", expand=True)

        self.bio_tab = ttk.Frame(nb, padding=10)
        self.input_tab = ttk.Frame(nb, padding=10)
        self.nn_tab = ttk.Frame(nb, padding=10)
        self.ng_tab = ttk.Frame(nb, padding=10)
        self.gg_tab = ttk.Frame(nb, padding=10)
        self.gn_tab = ttk.Frame(nb, padding=10)
        self.operator_tab = ttk.Frame(nb, padding=10)
        self.hw_tab = ttk.Frame(nb, padding=10)
        self.run_tab = ttk.Frame(nb, padding=10)

        nb.add(self.bio_tab, text="1. 生物参数")
        nb.add(self.input_tab, text="2. 神经输入")
        nb.add(self.nn_tab, text="3. N→N 神经连接")
        nb.add(self.ng_tab, text="4. N→G 输入映射")
        nb.add(self.gg_tab, text="5. G↔G 网络")
        nb.add(self.gn_tab, text="6. G→N 反馈")
        nb.add(self.operator_tab, text="7. 门 / 开放系统")
        nb.add(self.hw_tab, text="8. 真机设置")
        nb.add(self.run_tab, text="9. 运行 / 结果")

        self._build_bio_tab()
        self._build_input_tab()
        self._build_nn_tab()
        self._build_ng_tab()
        self._build_gg_tab()
        self._build_gn_tab()
        self._build_operator_tab()
        self._build_hw_tab()
        self._build_run_tab()

    def _build_bio_tab(self):
        ttk = self.ttk
        left = ttk.Frame(self.bio_tab)
        right = ttk.Frame(self.bio_tab)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right.grid(row=0, column=1, sticky="nsew")
        self.bio_tab.columnconfigure(0, weight=1)
        self.bio_tab.columnconfigure(1, weight=1)
        self.bio_tab.rowconfigure(0, weight=1)

        ttk.Label(left, text="九个核心蛋白", font=("Segoe UI", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )
        r = 1
        for g, desc in PROTEIN_INFO.items():
            ttk.Label(left, text=g, width=10).grid(row=r, column=0, sticky="w", pady=3)
            e = ttk.Entry(left, textvariable=self.protein_vars[g], width=12)
            e.grid(row=r, column=1, sticky="w", pady=3)
            e.bind("<FocusOut>", lambda event: self.update_preview())
            ttk.Label(left, text=desc, wraplength=420).grid(
                row=r, column=2, sticky="w", padx=(8,0), pady=3
            )
            r += 1

        ttk.Label(left, text="Input scale").grid(row=r, column=0, sticky="w", pady=(8,3))
        cb = ttk.Combobox(
            left,
            textvariable=self.scale_mode,
            values=["relative_linear", "log1p_expression"],
            state="readonly",
            width=18,
        )
        cb.grid(row=r, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", lambda event: self.update_preview())
        r += 1

        preset = ttk.LabelFrame(left, text="Presets", padding=6)
        preset.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10,0))
        for i, name in enumerate(PRESETS):
            ttk.Button(
                preset, text=name, command=lambda n=name: self.apply_preset(n)
            ).grid(row=i//3, column=i%3, padx=3, pady=3, sticky="ew")
        for c in range(3):
            preset.columnconfigure(c, weight=1)

        env = ttk.LabelFrame(right, text="Environment / state", padding=8)
        env.pack(fill="x")
        fields = [
            ("pH", self.ph_var),
            ("Extracellular K+ (mM)", self.k_var),
            ("Astrocyte Vm (mV)", self.vm_var),
            ("Global neuronal drive", self.drive_var),
            ("Hardware encoding gain", self.gain_var),
        ]
        for i, (lab, var) in enumerate(fields):
            ttk.Label(env, text=lab).grid(row=i, column=0, sticky="w", pady=4)
            e = ttk.Entry(env, textvariable=var, width=16)
            e.grid(row=i, column=1, sticky="w", pady=4)
            e.bind("<FocusOut>", lambda event: self.update_preview())

        v3 = ttk.LabelFrame(right, text="Optional V3 Kir noise calibration", padding=8)
        v3.pack(fill="x", pady=(10,0))
        ttk.Entry(v3, textvariable=self.v3_path_var, width=52).grid(
            row=0, column=0, sticky="ew", padx=(0,6)
        )
        ttk.Button(v3, text="Choose CSV", command=self.browse_v3).grid(row=0, column=1)
        ttk.Label(
            v3,
            text="Expected columns: ratio, mean_pre_voltage_SD_mV",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4,0))
        v3.columnconfigure(0, weight=1)

        preview = ttk.LabelFrame(right, text="Mechanistic / layer-gain preview", padding=8)
        preview.pack(fill="both", expand=True, pady=(10,0))
        labels = [
            ("KCNJ16/KCNJ10", "ratio"),
            ("Kir4.1/Kir5.1 f45", "f45"),
            ("Kir g/g_ref", "gkir"),
            ("Kir damping gamma", "gamma"),
            ("Threshold gain", "thr"),
            ("G→NN CSWAP gain", "xchg"),
            ("Kir buffer", "kir"),
            ("Glutamate clearance", "glu"),
            ("Ca signal", "ca"),
            ("Gap junction", "gj"),
            ("Neural excitability", "exc"),
            ("Glial network", "net"),
            ("N→N excitatory gain", "nn"),
            ("N→G gain", "ng"),
            ("G↔G ZZ phase gain", "gg_phase"),
            ("G↔G XY exchange gain", "gg_xy"),
            ("G→N gain", "gn"),
        ]
        for i, (lab, key) in enumerate(labels):
            ttk.Label(preview, text=lab).grid(row=i, column=0, sticky="w", pady=2)
            self.preview_vars[key] = self.tk.StringVar(value="-")
            ttk.Label(preview, textvariable=self.preview_vars[key]).grid(
                row=i, column=1, sticky="w", padx=(8,0)
            )
        ttk.Button(preview, text="Refresh", command=self.update_preview).grid(
            row=len(labels), column=0, columnspan=2, sticky="ew", pady=(8,0)
        )

    def _build_input_tab(self):
        ttk = self.ttk
        top = ttk.LabelFrame(self.input_tab, text="Network scale / layers", padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Total qubits").grid(row=0, column=0, sticky="w")
        qbox = ttk.Combobox(
            top,
            textvariable=self.total_qubits_var,
            values=["8", "16", "32", "64", "128"],
            width=10,
        )
        qbox.grid(row=0, column=1, padx=(6,16))
        ttk.Label(top, text="Layers").grid(row=0, column=2)
        ttk.Entry(top, textvariable=self.layers_var, width=8).grid(row=0, column=3, padx=(6,16))
        ttk.Label(top, text="Layer dt metadata (ms)").grid(row=0, column=4)
        ttk.Entry(top, textvariable=self.layer_dt_var, width=10).grid(row=0, column=5, padx=(6,16))
        ttk.Button(top, text="Apply size / reset topology", command=self.reset_network_defaults).grid(
            row=0, column=6
        )
        ttk.Label(
            top,
            text=(
                "每个 layer 表示一次离散 N→G→G→N interaction step；dt 目前仅作为时间标签，"
                "不是已拟合的真实生理时间常数。"
            ),
            wraplength=1080,
        ).grid(row=1, column=0, columnspan=7, sticky="w", pady=(6,0))

        body = ttk.Frame(self.input_tab)
        body.pack(fill="both", expand=True, pady=(10,0))
        left = ttk.Frame(body)
        right = ttk.LabelFrame(body, text="Add / edit neural input", padding=8)
        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="y", padx=(10,0))

        self.input_tree = EditableTree(
            left,
            ["layer", "neuron", "amplitude", "phase"],
            ["Layer", "Neuron", "Amplitude", "Phase (rad)"],
            [80, 90, 110, 110],
        )

        self.in_layer = self.tk.StringVar(value="0")
        self.in_neuron = self.tk.StringVar(value="0")
        self.in_amp = self.tk.StringVar(value="1.0")
        self.in_phase = self.tk.StringVar(value="0.0")
        for i, (lab, var) in enumerate([
            ("Layer", self.in_layer),
            ("Neuron", self.in_neuron),
            ("Amplitude", self.in_amp),
            ("Phase rad", self.in_phase),
        ]):
            ttk.Label(right, text=lab).grid(row=i, column=0, sticky="w", pady=4)
            ttk.Entry(right, textvariable=var, width=14).grid(row=i, column=1, pady=4)

        ttk.Button(right, text="Add input", command=self.add_input_event).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(8,3)
        )
        ttk.Button(right, text="Update selected", command=self.update_input_event).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=3
        )
        ttk.Button(right, text="Delete selected", command=self.input_tree.delete_selected).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=3
        )
        ttk.Separator(right).grid(row=8, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(right, text="Focal N0 pulse", command=self.preset_focal_input).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=3
        )
        ttk.Button(right, text="Sequential wave", command=self.preset_sequential_input).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=3
        )
        ttk.Button(right, text="Clear inputs", command=self.input_tree.clear).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=3
        )

    def _edge_tab_common(self, tab, kind):
        ttk = self.ttk
        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        right = ttk.LabelFrame(body, text="Edge editor", padding=8)
        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="y", padx=(10,0))

        if kind == "nn":
            tree = EditableTree(
                left, ["source", "target", "weight"],
                ["Neuron source", "Neuron target", "Weight"],
                [120,120,100],
            )
            labels = ("Neuron source", "Neuron target")
        elif kind == "ng":
            tree = EditableTree(
                left, ["source", "target", "weight"],
                ["Neuron source", "Glia target", "Weight"],
                [120,120,100],
            )
            labels = ("Neuron", "Glia")
        elif kind == "gg":
            tree = EditableTree(
                left, ["source", "target", "weight", "exchange_weight"],
                ["Glia A", "Glia B", "Phase weight", "XY exchange"],
                [110,110,110,110],
            )
            labels = ("Glia A", "Glia B")
        else:
            tree = EditableTree(
                left, ["source", "target", "weight"],
                ["Glia source", "Neuron target", "Weight"],
                [120,120,100],
            )
            labels = ("Glia", "Neuron")

        src = self.tk.StringVar(value="0")
        dst = self.tk.StringVar(value="0" if kind != "gg" else "1")
        w = self.tk.StringVar(value="1.0")
        xw = self.tk.StringVar(value="" if kind == "gg" else "")
        ttk.Label(right, text=labels[0]).grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(right, textvariable=src, width=12).grid(row=0, column=1, pady=4)
        ttk.Label(right, text=labels[1]).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(right, textvariable=dst, width=12).grid(row=1, column=1, pady=4)
        ttk.Label(right, text="Phase weight" if kind == "gg" else "Weight").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(right, textvariable=w, width=12).grid(row=2, column=1, pady=4)
        if kind == "gg":
            ttk.Label(right, text="XY exchange (blank=|phase|)").grid(row=3, column=0, sticky="w", pady=4)
            ttk.Entry(right, textvariable=xw, width=12).grid(row=3, column=1, pady=4)

        def _edge_values():
            vals = [int(src.get()), int(dst.get()), float(w.get())]
            if kind == "gg":
                vals.append(xw.get().strip())
            return vals

        def add_edge():
            tree.add(_edge_values())

        def update_edge():
            sel = tree.tree.selection()
            if not sel:
                return
            tree.tree.item(sel[0], values=_edge_values())

        def load_selected(event=None):
            vals = tree.selected_values()
            if vals:
                src.set(vals[0]); dst.set(vals[1]); w.set(vals[2])
                if kind == "gg":
                    xw.set(vals[3] if len(vals) > 3 else "")

        tree.tree.bind("<<TreeviewSelect>>", load_selected)

        ttk.Button(right, text="Add edge", command=add_edge).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(8,3)
        )
        ttk.Button(right, text="Update selected", command=update_edge).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=3
        )
        ttk.Button(right, text="Delete selected", command=tree.delete_selected).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=3
        )
        ttk.Button(right, text="Clear edges", command=tree.clear).grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=3
        )

        ttk.Separator(right).grid(row=8, column=0, columnspan=2, sticky="ew", pady=8)

        if kind == "nn":
            ttk.Button(right, text="Preset: feed-forward chain", command=self.preset_nn_chain).grid(
                row=9, column=0, columnspan=2, sticky="ew", pady=3
            )
            ttk.Button(right, text="Preset: clear", command=tree.clear).grid(
                row=10, column=0, columnspan=2, sticky="ew", pady=3
            )
            self.nn_tree = tree
        elif kind == "ng":
            ttk.Button(right, text="Preset: one-to-one", command=self.preset_ng_one_to_one).grid(
                row=9, column=0, columnspan=2, sticky="ew", pady=3
            )
            ttk.Button(right, text="Preset: local + neighbor", command=self.preset_ng_neighbor).grid(
                row=10, column=0, columnspan=2, sticky="ew", pady=3
            )
            self.ng_tree = tree
        elif kind == "gg":
            ttk.Button(right, text="Preset: ring", command=self.preset_gg_ring).grid(
                row=9, column=0, columnspan=2, sticky="ew", pady=3
            )
            ttk.Button(right, text="Preset: line", command=self.preset_gg_line).grid(
                row=10, column=0, columnspan=2, sticky="ew", pady=3
            )
            self.gg_tree = tree
        else:
            ttk.Button(right, text="Preset: one-to-one", command=self.preset_gn_one_to_one).grid(
                row=9, column=0, columnspan=2, sticky="ew", pady=3
            )
            ttk.Button(right, text="Mirror N→G", command=self.preset_gn_mirror_ng).grid(
                row=10, column=0, columnspan=2, sticky="ew", pady=3
            )
            self.gn_tree = tree

        ttk.Separator(right).grid(row=11, column=0, columnspan=2, sticky="ew", pady=8)
        ttk.Button(
            right,
            text="Import CSV",
            command=lambda: self.import_edge_csv(kind)
        ).grid(row=12, column=0, columnspan=2, sticky="ew", pady=3)
        ttk.Button(
            right,
            text="Export CSV",
            command=lambda: self.export_edge_csv(kind)
        ).grid(row=13, column=0, columnspan=2, sticky="ew", pady=3)

    def _build_nn_tab(self):
        self._edge_tab_common(self.nn_tab, "nn")
        self.ttk.Label(
            self.nn_tab,
            text=(
                "N→N 为经典有向兴奋性神经连接：source=N_i，target=N_j，weight>=0。"
                " 数学上用受 source 活动控制的 RY 激活 target；这是突触传播的量子编码类比。"
            )
        ).pack(anchor="w", pady=(8,0))

    def _build_ng_tab(self):
        self._edge_tab_common(self.ng_tab, "ng")
        self.ttk.Label(
            self.ng_tab,
            text=(
                "这里就是“哪些神经输入哪些胶质”的直接编辑端。"
                " source=N_i，target=G_j，weight 可正可负。"
            )
        ).pack(anchor="w", pady=(8,0))

    def _build_gg_tab(self):
        self._edge_tab_common(self.gg_tab, "gg")
        self.ttk.Label(
            self.gg_tab,
            text="G↔G 为无向胶质网络；默认 ring。大规模 64/128 qubits 建议保持稀疏。"
        ).pack(anchor="w", pady=(8,0))

    def _build_gn_tab(self):
        self._edge_tab_common(self.gn_tab, "gn")
        self.ttk.Label(
            self.gn_tab,
            text="G→N 为胶质反馈读出，可与 N→G 不对称；这使 N→G→G→N 路径可独立消融。"
        ).pack(anchor="w", pady=(8,0))

    def _build_operator_tab(self):
        ttk = self.ttk
        ttk.Label(
            self.operator_tab,
            text="V6.6 neuroglial operator layer",
            font=("Segoe UI", 12, "bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,10))

        ttk.Checkbutton(
            self.operator_tab,
            text="Enable shared-glia two-input CC-RY threshold (Toffoli limit)",
            variable=self.enable_threshold_var,
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(self.operator_tab, text="Max threshold pairs / glia").grid(row=2, column=0, sticky="w")
        ttk.Entry(self.operator_tab, textvariable=self.threshold_pairs_var, width=10).grid(row=2, column=1, sticky="w")

        ttk.Checkbutton(
            self.operator_tab,
            text="Enable glial-controlled CSWAP exchange (idealized; OFF by default)",
            variable=self.enable_exchange_var,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(10,4))
        ttk.Label(self.operator_tab, text="Exchange activation threshold").grid(row=4, column=0, sticky="w")
        ttk.Entry(self.operator_tab, textvariable=self.exchange_threshold_var, width=10).grid(row=4, column=1, sticky="w")

        ttk.Checkbutton(
            self.operator_tab,
            text="Enable V6.6 G↔G XY / partial-iSWAP state exchange (hybrid ZZ+XY)",
            variable=self.enable_glia_exchange_var,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(10,4))

        ttk.Checkbutton(
            self.operator_tab,
            text="Enable Kir-like final damping dilation in full network (experimental; OFF by default)",
            variable=self.enable_damping_var,
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(10,4))
        ttk.Label(self.operator_tab, text="Kir assembly efficiency eta").grid(row=7, column=0, sticky="w")
        e = ttk.Entry(self.operator_tab, textvariable=self.kir_eta_var, width=10)
        e.grid(row=7, column=1, sticky="w")
        e.bind("<FocusOut>", lambda event: self.update_preview())

        ttk.Checkbutton(
            self.operator_tab,
            text="V7.1 initialize neuronal H carrier (ON = preserve V6.9 behavior)",
            variable=self.initialize_neural_h_var,
        ).grid(row=8, column=0, columnspan=3, sticky="w", pady=(10,4))
        ttk.Checkbutton(
            self.operator_tab,
            text="V7.1 initialize biology-dependent glial baseline (ON = preserve V6.9 behavior)",
            variable=self.initialize_glial_baseline_var,
        ).grid(row=9, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(self.operator_tab, text="V7.1 neural readout mode").grid(row=10, column=0, sticky="w", pady=(8,4))
        ttk.Combobox(
            self.operator_tab, textvariable=self.neural_readout_mode_var,
            values=["phase", "population"], state="readonly", width=14
        ).grid(row=10, column=1, sticky="w", pady=(8,4))

        ttk.Separator(self.operator_tab).grid(row=10, column=0, columnspan=3, sticky="ew", pady=12)
        ttk.Label(
            self.operator_tab,
            text=(
                "Exact gate-validation mode starts from explicit computational states and clean ancillae. "
                "Use it to falsifiably test CRY/CNOT/CZ/ZZ/XY/CCRY/Toffoli/CSWAP/damping/reset. "
                "The full-network damping switch is not claimed to be a Markovian CPTP channel because the glial qubit may retain context."
            ), wraplength=980
        ).grid(row=9, column=0, columnspan=3, sticky="w")

        ttk.Button(
            self.operator_tab, text="Validate gates on CPUQVM",
            command=lambda: self.start_gate_validation("local")
        ).grid(row=9, column=0, sticky="w", pady=(14,4))
        ttk.Button(
            self.operator_tab, text="Validate gates on OriginQ QPU",
            command=lambda: self.start_gate_validation("qpu")
        ).grid(row=9, column=1, sticky="w", pady=(14,4), padx=(8,0))

    def _build_hw_tab(self):
        ttk = self.ttk
        self.api_status_var = self.tk.StringVar()
        self.refresh_api_status()
        ttk.Label(
            self.hw_tab, textvariable=self.api_status_var,
            font=("Segoe UI", 11, "bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,8))

        fields = [
            ("Backend", self.backend_var),
            ("QPU shots / circuit", self.qpu_shots_var),
            ("QPU repeats", self.qpu_repeats_var),
            ("Circuits per cloud batch", self.batch_size_var),
            ("Max in-flight cloud jobs", self.parallel_jobs_var),
            ("Scheduler retries", self.scheduler_retries_var),
            ("Retry wait (s)", self.scheduler_retry_wait_var),
            ("CPUQVM shots", self.local_shots_var),
            ("CPUQVM safe qubit limit", self.local_safe_var),
        ]
        for i, (lab, var) in enumerate(fields, start=1):
            ttk.Label(self.hw_tab, text=lab).grid(row=i, column=0, sticky="w", pady=5)
            ttk.Entry(self.hw_tab, textvariable=var, width=24).grid(
                row=i, column=1, sticky="w", pady=5
            )

        ttk.Checkbutton(self.hw_tab, text="Mapping", variable=self.mapping_var).grid(
            row=11, column=0, sticky="w", pady=5
        )
        ttk.Checkbutton(
            self.hw_tab, text="Optimization", variable=self.optimization_var
        ).grid(row=11, column=1, sticky="w", pady=5)
        ttk.Checkbutton(self.hw_tab, text="Amend", variable=self.amend_var).grid(
            row=11, column=2, sticky="w", pady=5
        )
        ttk.Checkbutton(
            self.hw_tab,
            text="Advanced: use specified physical blocks",
            variable=self.use_specified_blocks_var,
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Checkbutton(
            self.hw_tab,
            text="Advanced: force parallel large-QPU jobs",
            variable=self.force_parallel_large_var,
        ).grid(row=13, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Button(
            self.hw_tab, text="Recheck environment",
            command=self.refresh_api_status
        ).grid(row=14, column=0, sticky="w", pady=(10,0))

        ttk.Label(
            self.hw_tab,
            text=(
                "V6.6 retains V6.2.1 robust scheduler mode: for 16+ qubits, matched ablations are automatically batched "
                "into one cloud job per repeat and only one large job is kept in flight. Specified "
                "physical blocks are OFF by default to avoid scheduler logic-block allocation errors. "
                "128-qubit mode still depends on the current backend's connected/available hardware."
            ),
            wraplength=950
        ).grid(row=15, column=0, columnspan=3, sticky="w", pady=(12,0))

    def _build_run_tab(self):
        ttk = self.ttk
        bar = ttk.Frame(self.run_tab)
        bar.pack(fill="x")
        ttk.Button(bar, text="Build / resource check", command=self.resource_check).pack(
            side="left", padx=(0,5)
        )
        ttk.Button(bar, text="Save project JSON", command=self.save_project_dialog).pack(
            side="left", padx=5
        )
        ttk.Button(bar, text="Load project JSON", command=self.load_project_dialog).pack(
            side="left", padx=5
        )
        ttk.Button(bar, text="Run CPUQVM", command=lambda: self.start_run("local")).pack(
            side="left", padx=5
        )
        ttk.Button(bar, text="Run OriginQ QPU", command=lambda: self.start_run("qpu")).pack(
            side="left", padx=5
        )
        ttk.Button(bar, text="Gate test CPU", command=lambda: self.start_gate_validation("local")).pack(
            side="left", padx=5
        )
        ttk.Button(bar, text="Gate test QPU", command=lambda: self.start_gate_validation("qpu")).pack(
            side="left", padx=5
        )

        self.status_var = self.tk.StringVar(value="Ready")
        ttk.Label(self.run_tab, textvariable=self.status_var).pack(anchor="w", pady=(10,4))
        self.log_text = self.tk.Text(self.run_tab, height=34, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.insert(
            "end",
            "V6.6 ready.\n"
            "Recommended first experiment: 16 qubits, 3 layers, sparse one-to-one N→G, ring G↔G, "
            "one-to-one G→N, focal N0 input.\n\n"
        )

    # ---------------------------------------------------------------------
    # Data collection / preset actions
    # ---------------------------------------------------------------------

    def current_sizes(self):
        tq = int(self.total_qubits_var.get())
        layers = int(self.layers_var.get())
        if tq < 4 or tq > 128 or tq % 2:
            raise ValueError("Total qubits must be even, 4..128.")
        if layers < 1 or layers > 64:
            raise ValueError("Layers must be 1..64.")
        return tq, tq//2, layers

    def reset_network_defaults(self):
        tq, n, layers = self.current_sizes()
        self.input_tree.clear()
        self.nn_tree.clear()
        self.ng_tree.clear()
        self.gg_tree.clear()
        self.gn_tree.clear()
        self.preset_focal_input()
        self.preset_ng_one_to_one()
        self.preset_gg_ring()
        self.preset_gn_one_to_one()
        self.set_status(f"Topology reset: {n} neurons + {n} glia, {layers} layers")

    def add_input_event(self):
        self.input_tree.add([
            int(self.in_layer.get()),
            int(self.in_neuron.get()),
            float(self.in_amp.get()),
            float(self.in_phase.get()),
        ])

    def update_input_event(self):
        sel = self.input_tree.tree.selection()
        if not sel:
            return
        self.input_tree.tree.item(
            sel[0],
            values=[
                int(self.in_layer.get()),
                int(self.in_neuron.get()),
                float(self.in_amp.get()),
                float(self.in_phase.get()),
            ],
        )

    def preset_focal_input(self):
        self.input_tree.clear()
        self.input_tree.add([0, 0, 1.0, 0.0])

    def preset_sequential_input(self):
        self.input_tree.clear()
        tq, n, layers = self.current_sizes()
        for l in range(layers):
            self.input_tree.add([l, l % n, 1.0, 0.0])

    def preset_nn_chain(self):
        self.nn_tree.clear()
        _, n, _ = self.current_sizes()
        for i in range(max(0, n-1)):
            self.nn_tree.add([i, i+1, 1.0])

    def preset_ng_one_to_one(self):
        self.ng_tree.clear()
        _, n, _ = self.current_sizes()
        for i in range(n):
            self.ng_tree.add([i, i, 1.0])

    def preset_ng_neighbor(self):
        self.ng_tree.clear()
        _, n, _ = self.current_sizes()
        for i in range(n):
            self.ng_tree.add([i, i, 1.0])
            self.ng_tree.add([i, (i+1) % n, 0.45])

    def preset_gg_ring(self):
        self.gg_tree.clear()
        _, n, _ = self.current_sizes()
        seen = set()
        for i in range(n):
            j = (i+1) % n
            edge = tuple(sorted((i,j)))
            if edge not in seen:
                self.gg_tree.add([edge[0], edge[1], 1.0, ""])
                seen.add(edge)

    def preset_gg_line(self):
        self.gg_tree.clear()
        _, n, _ = self.current_sizes()
        for i in range(n-1):
            self.gg_tree.add([i, i+1, 1.0, ""])

    def preset_gn_one_to_one(self):
        self.gn_tree.clear()
        _, n, _ = self.current_sizes()
        for i in range(n):
            self.gn_tree.add([i, i, 0.75])

    def preset_gn_mirror_ng(self):
        self.gn_tree.clear()
        for src, dst, weight in self.ng_tree.rows():
            # N src -> G dst becomes G dst -> N src.
            self.gn_tree.add([int(dst), int(src), 0.75*float(weight)])

    def import_edge_csv(self, kind):
        from tkinter import filedialog, messagebox
        try:
            path = filedialog.askopenfilename(
                title="Import edge CSV",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            if not path:
                return
            df = pd.read_csv(path)
            required = {"source", "target", "weight"}
            if not required.issubset(df.columns):
                raise ValueError("CSV requires columns: source,target,weight")
            tree = {"nn": self.nn_tree, "ng": self.ng_tree, "gg": self.gg_tree, "gn": self.gn_tree}[kind]
            tree.clear()
            for _, r in df.iterrows():
                vals = [int(r["source"]), int(r["target"]), float(r["weight"])]
                if kind == "gg":
                    if "exchange_weight" in df.columns and not pd.isna(r["exchange_weight"]):
                        vals.append(float(r["exchange_weight"]))
                    else:
                        vals.append("")
                tree.add(vals)
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))

    def export_edge_csv(self, kind):
        from tkinter import filedialog
        tree = {"nn": self.nn_tree, "ng": self.ng_tree, "gg": self.gg_tree, "gn": self.gn_tree}[kind]
        path = filedialog.asksaveasfilename(
            title="Export edge CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if path:
            cols = ["source", "target", "weight"]
            if kind == "gg":
                cols.append("exchange_weight")
            pd.DataFrame(tree.rows(), columns=cols).to_csv(path, index=False)

    def apply_preset(self, name):
        p = PRESETS[name]
        for g, v in p.items():
            self.protein_vars[g].set(str(v))
        self.scale_mode.set("relative_linear")
        if name == "Epilepsy-like stress":
            self.k_var.set("6.5")
            self.drive_var.set("1.6")
            self.vm_var.set("-68")
        else:
            self.k_var.set("3.5")
            self.drive_var.set("1.0")
            self.vm_var.set("-80")
        self.update_preview()

    def browse_v3(self):
        from tkinter import filedialog
        p = filedialog.askopenfilename(
            title="Choose V3_ratio_wholecell_summary.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if p:
            self.v3_path_var.set(p)
            self.update_preview()

    # ---------------------------------------------------------------------
    # Collect / serialize project
    # ---------------------------------------------------------------------

    def collect_bio(self):
        inp = BiologicalInput(
            proteins={g: float(self.protein_vars[g].get()) for g in PROTEIN_INFO},
            pH=float(self.ph_var.get()),
            K_out_mM=float(self.k_var.get()),
            Vm_mV=float(self.vm_var.get()),
            neuronal_drive=float(self.drive_var.get()),
            hardware_gain=float(self.gain_var.get()),
            scale_mode=self.scale_mode.get(),
            v3_summary=self.v3_path_var.get().strip(),
            kir_assembly_eta=float(self.kir_eta_var.get()),
        )
        validate_biological_input(inp)
        return inp

    def collect_network(self):
        tq, n, layers = self.current_sizes()
        inputs = [
            NeuralInputEvent(
                int(r[0]), int(r[1]), float(r[2]), float(r[3])
            )
            for r in self.input_tree.rows()
        ]
        nn = [
            DirectedEdge(int(r[0]), int(r[1]), float(r[2]))
            for r in self.nn_tree.rows()
        ]
        ng = [
            DirectedEdge(int(r[0]), int(r[1]), float(r[2]))
            for r in self.ng_tree.rows()
        ]
        gg = [
            UndirectedEdge(
                int(r[0]), int(r[1]), float(r[2]),
                None if len(r) < 4 or str(r[3]).strip() == "" else float(r[3])
            )
            for r in self.gg_tree.rows()
        ]
        gn = [
            DirectedEdge(int(r[0]), int(r[1]), float(r[2]))
            for r in self.gn_tree.rows()
        ]
        cfg = NetworkConfig(
            total_qubits=tq,
            layers=layers,
            layer_dt_ms=float(self.layer_dt_var.get()),
            neural_inputs=inputs,
            neuron_to_neuron=nn,
            neuron_to_glia=ng,
            glia_glia=gg,
            glia_to_neuron=gn,
            enable_shared_glia_threshold=bool(self.enable_threshold_var.get()),
            max_threshold_pairs_per_glia=int(self.threshold_pairs_var.get()),
            enable_controlled_exchange=bool(self.enable_exchange_var.get()),
            exchange_activation_threshold=float(self.exchange_threshold_var.get()),
            enable_glia_exchange=bool(self.enable_glia_exchange_var.get()),
            enable_kir_damping=bool(self.enable_damping_var.get()),
            initialize_neural_h=bool(self.initialize_neural_h_var.get()),
            initialize_glial_baseline=bool(self.initialize_glial_baseline_var.get()),
            neural_readout_mode=str(self.neural_readout_mode_var.get()).lower(),
        )
        validate_network_config(cfg)
        return cfg

    def collect_hardware(self):
        h = HardwareSettings(
            backend=self.backend_var.get().strip() or "auto",
            shots=int(self.qpu_shots_var.get()),
            repeats=int(self.qpu_repeats_var.get()),
            batch_size=int(self.batch_size_var.get()),
            parallel_jobs=int(self.parallel_jobs_var.get()),
            mapping=bool(self.mapping_var.get()),
            optimization=bool(self.optimization_var.get()),
            amend=bool(self.amend_var.get()),
            use_specified_blocks=bool(self.use_specified_blocks_var.get()),
            prefer_disjoint_blocks=False,
            scheduler_retries=int(self.scheduler_retries_var.get()),
            scheduler_retry_wait_s=int(self.scheduler_retry_wait_var.get()),
            force_parallel_large_qpu=bool(self.force_parallel_large_var.get()),
            local_safe_qubits=int(self.local_safe_var.get()),
        )
        if min(h.shots, h.repeats, h.batch_size, h.parallel_jobs) < 1:
            raise ValueError("shots/repeats/batch_size/parallel_jobs must be >= 1.")
        return h

    def project_payload(self):
        bio = self.collect_bio()
        cfg = self.collect_network()
        hw = self.collect_hardware()
        return {
            "version": VERSION,
            "biological_input": asdict(bio),
            "network": {
                "total_qubits": cfg.total_qubits,
                "layers": cfg.layers,
                "layer_dt_ms": cfg.layer_dt_ms,
                "neural_inputs": [asdict(x) for x in cfg.neural_inputs or []],
                "neuron_to_neuron": [asdict(x) for x in cfg.neuron_to_neuron or []],
                "neuron_to_glia": [asdict(x) for x in cfg.neuron_to_glia or []],
                "glia_glia": [asdict(x) for x in cfg.glia_glia or []],
                "glia_to_neuron": [asdict(x) for x in cfg.glia_to_neuron or []],
                "enable_shared_glia_threshold": cfg.enable_shared_glia_threshold,
                "max_threshold_pairs_per_glia": cfg.max_threshold_pairs_per_glia,
                "activity_gate_shared_glia_threshold_targets": list(cfg.activity_gate_shared_glia_threshold_targets or []),
                "activity_gate_glia_microdomain_targets": list(cfg.activity_gate_glia_microdomain_targets or []),
                "tripartite_synapses": [asdict(x) for x in cfg.tripartite_synapses or []],
                "enable_tripartite_synapses": cfg.enable_tripartite_synapses,
                "enable_controlled_exchange": cfg.enable_controlled_exchange,
                "exchange_activation_threshold": cfg.exchange_activation_threshold,
                "enable_glia_exchange": cfg.enable_glia_exchange,
                "scale_glia_exchange_by_layers": cfg.scale_glia_exchange_by_layers,
                "activity_gate_glia_exchange": cfg.activity_gate_glia_exchange,
                "enable_kir_damping": cfg.enable_kir_damping,
                "initialize_neural_h": cfg.initialize_neural_h,
                "initialize_glial_baseline": cfg.initialize_glial_baseline,
                "neural_readout_mode": getattr(cfg, "neural_readout_mode", "phase"),
            },
            "hardware": asdict(hw),
            "local_shots": int(self.local_shots_var.get()),
        }

    def save_project_dialog(self):
        from tkinter import filedialog, messagebox
        try:
            payload = self.project_payload()
            p = filedialog.asksaveasfilename(
                title="Save V7.0 project",
                defaultextension=".json",
                initialfile="V7_0_project.json",
                filetypes=[("JSON files", "*.json")]
            )
            if p:
                Path(p).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def load_project_dialog(self):
        from tkinter import filedialog, messagebox
        try:
            p = filedialog.askopenfilename(
                title="Load V7.0 project",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not p:
                return
            data = json.loads(Path(p).read_text(encoding="utf-8"))
            b = data["biological_input"]
            for g, v in b["proteins"].items():
                if g in self.protein_vars:
                    self.protein_vars[g].set(str(v))
            self.ph_var.set(str(b.get("pH", 7.4)))
            self.k_var.set(str(b.get("K_out_mM", 3.5)))
            self.vm_var.set(str(b.get("Vm_mV", -80)))
            self.drive_var.set(str(b.get("neuronal_drive", 1.0)))
            self.gain_var.set(str(b.get("hardware_gain", 1.0)))
            self.scale_mode.set(b.get("scale_mode", "relative_linear"))
            self.v3_path_var.set(b.get("v3_summary", ""))
            self.kir_eta_var.set(str(b.get("kir_assembly_eta", 0.75)))

            n = data["network"]
            self.total_qubits_var.set(str(n["total_qubits"]))
            self.layers_var.set(str(n["layers"]))
            self.layer_dt_var.set(str(n.get("layer_dt_ms", 10.0)))
            self.input_tree.clear()
            for r in n.get("neural_inputs", []):
                self.input_tree.add([r["layer"], r["neuron"], r["amplitude"], r.get("phase_rad",0)])
            self.nn_tree.clear()
            for r in n.get("neuron_to_neuron", []):
                self.nn_tree.add([r["source"], r["target"], r["weight"]])
            self.ng_tree.clear()
            for r in n.get("neuron_to_glia", []):
                self.ng_tree.add([r["source"], r["target"], r["weight"]])
            self.gg_tree.clear()
            for r in n.get("glia_glia", []):
                self.gg_tree.add([r["a"], r["b"], r["weight"], "" if r.get("exchange_weight") is None else r.get("exchange_weight")])
            self.gn_tree.clear()
            for r in n.get("glia_to_neuron", []):
                self.gn_tree.add([r["source"], r["target"], r["weight"]])
            self.enable_threshold_var.set(bool(n.get("enable_shared_glia_threshold", True)))
            self.threshold_pairs_var.set(str(n.get("max_threshold_pairs_per_glia", 1)))
            self.enable_exchange_var.set(bool(n.get("enable_controlled_exchange", False)))
            self.exchange_threshold_var.set(str(n.get("exchange_activation_threshold", 0.75)))
            self.enable_glia_exchange_var.set(bool(n.get("enable_glia_exchange", True)))
            self.enable_damping_var.set(bool(n.get("enable_kir_damping", False)))
            self.initialize_neural_h_var.set(bool(n.get("initialize_neural_h", True)))
            self.initialize_glial_baseline_var.set(bool(n.get("initialize_glial_baseline", True)))
            self.neural_readout_mode_var.set(str(n.get("neural_readout_mode", "phase")))

            h = data.get("hardware", {})
            self.backend_var.set(str(h.get("backend","auto")))
            self.qpu_shots_var.set(str(h.get("shots",1000)))
            self.qpu_repeats_var.set(str(h.get("repeats",3)))
            self.batch_size_var.set(str(h.get("batch_size",5)))
            self.parallel_jobs_var.set(str(h.get("parallel_jobs",1)))
            self.local_safe_var.set(str(h.get("local_safe_qubits",20)))
            self.mapping_var.set(bool(h.get("mapping",True)))
            self.optimization_var.set(bool(h.get("optimization",True)))
            self.amend_var.set(bool(h.get("amend",True)))
            self.use_specified_blocks_var.set(bool(h.get("use_specified_blocks",False)))
            self.force_parallel_large_var.set(bool(h.get("force_parallel_large_qpu",False)))
            self.scheduler_retries_var.set(str(h.get("scheduler_retries",3)))
            self.scheduler_retry_wait_var.set(str(h.get("scheduler_retry_wait_s",30)))
            self.local_shots_var.set(str(data.get("local_shots",20000)))
            self.update_preview()
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    # ---------------------------------------------------------------------
    # Preview / resource check / run
    # ---------------------------------------------------------------------

    def update_preview(self):
        try:
            bio = self.collect_bio()
            s = compute_mechanistic_scores(bio)
            g = scores_to_layer_gains(s, bio.hardware_gain)
            vals = {
                "ratio": s.ratio_KCNJ16_over_KCNJ10,
                "f45": s.kir_f45,
                "gkir": s.kir_effective_g_ratio_to_reference,
                "gamma": g.kir_damping_gamma,
                "thr": g.threshold_gate_gain,
                "xchg": g.exchange_gate_gain,
                "kir": s.kir_buffer_score,
                "glu": s.glutamate_clearance_score,
                "ca": s.ca_signal_score,
                "gj": s.gap_junction_score,
                "exc": s.neural_excitability_score,
                "net": s.glial_network_score,
                "nn": g.neuron_to_neuron_gain,
                "ng": g.neuron_to_glia_gain,
                "gg_phase": g.glia_glia_phase_gain,
                "gg_xy": g.glia_glia_exchange_gain,
                "gn": g.glia_to_neuron_gain,
            }
            for k, v in vals.items():
                self.preview_vars[k].set(f"{v:.6g}")
        except Exception:
            for v in self.preview_vars.values():
                v.set("-")

    def refresh_api_status(self):
        ok = bool(os.environ.get("QPANDA_QCLOUD_API_KEY","").strip())
        self.api_status_var.set(
            "Environment API: QPANDA_QCLOUD_API_KEY detected"
            if ok else
            "Environment API: QPANDA_QCLOUD_API_KEY NOT detected"
        )

    def log(self, msg):
        def write():
            self.log_text.insert("end", str(msg)+"\n")
            self.log_text.see("end")
        self.root.after(0, write)

    def set_status(self, msg):
        self.root.after(0, lambda: self.status_var.set(msg))

    def resource_check(self):
        from tkinter import messagebox
        try:
            cfg = self.collect_network()
            bio = self.collect_bio()
            s = compute_mechanistic_scores(bio)
            gains = scores_to_layer_gains(s, bio.hardware_gain)
            est = circuit_resource_estimate(cfg)
            text = "\n".join(f"{k}: {v}" for k,v in est.items())
            if cfg.total_qubits >= 64 and est["approx_CNOT_count"] > 3000:
                text += (
                    "\n\nWARNING: very deep large-QPU circuit. "
                    "Reduce layers or use a sparser topology."
                )
            messagebox.showinfo("V6.8 resource estimate", text)
            self.log("\nResource estimate:")
            self.log(text)
            self.log("\nLayer gains:")
            for k,v in asdict(gains).items():
                self.log(f"  {k}: {v:.6g}")
        except Exception as exc:
            messagebox.showerror("Resource check failed", str(exc))

    def _save_manifest(self, outdir, bio, cfg, gains, hw, est):
        payload = self.project_payload()
        payload.update({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "mechanistic_scores": asdict(compute_mechanistic_scores(bio)),
            "layer_gains": asdict(gains),
            "resource_estimate_full": est,
            "scientific_scope": (
                "Hybrid quantum-classical neuroglial emulator. "
                "Protein->circuit mapping is mechanistically constrained but not a fitted kinetic law."
            ),
        })
        atomic_json_write(outdir / "V6_4_manifest.json", payload)
        pd.DataFrame([asdict(compute_mechanistic_scores(bio))]).to_csv(
            outdir / "V6_4_mechanistic_scores.csv", index=False
        )
        pd.DataFrame([asdict(gains)]).to_csv(
            outdir / "V6_4_layer_gains.csv", index=False
        )

    def start_run(self, mode):
        try:
            bio = self.collect_bio()
            cfg = self.collect_network()
            hw = self.collect_hardware()
            local_shots = int(self.local_shots_var.get())
            scores = compute_mechanistic_scores(bio)
            gains = scores_to_layer_gains(scores, bio.hardware_gain)
            variants = build_condition_variants(cfg)
            est = circuit_resource_estimate(cfg)
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror("Invalid input", str(exc))
            return

        def worker():
            try:
                self.set_status(f"Running {mode}...")
                outdir = Path(f"V6_4_{mode}_{timestamp_string()}")
                outdir.mkdir(parents=True, exist_ok=True)
                self._save_manifest(outdir, bio, cfg, gains, hw, est)

                self.log(f"\n=== {VERSION} {mode.upper()} ===")
                self.log(f"Output: {outdir.resolve()}")
                self.log(
                    f"Architecture: {cfg.n_neurons} N + {cfg.n_glia} G = "
                    f"{cfg.total_qubits} qubits; layers={cfg.layers}"
                )
                self.log(
                    f"Edges: N->N={len(cfg.neuron_to_neuron or [])}, N->G={len(cfg.neuron_to_glia or [])}, "
                    f"G-G={len(cfg.glia_glia or [])}, "
                    f"G->N={len(cfg.glia_to_neuron or [])}"
                )
                self.log(
                    f"Approx primitive gates={est['approx_primitive_gates']}; "
                    f"approx CNOT={est['approx_CNOT_count']}"
                )

                if mode == "local":
                    mean_dists = run_local(
                        variants, gains, local_shots, hw.local_safe_qubits,
                        outdir, self.log
                    )
                    prefix = "V6_4_simulator"
                elif mode == "qpu":
                    mean_dists, repeat_dists, cloud = run_qpu(
                        variants, gains, hw, outdir, self.log
                    )
                    prefix = "V6_4_QPU"
                else:
                    raise ValueError(mode)

                summary, marg = summarize_distributions(
                    mean_dists, cfg.n_neurons, outdir, prefix
                )

                self.log("\nMatched ablation summary:")
                for _, r in summary.iterrows():
                    self.log(
                        f"  {r['condition']}: "
                        f"JS_vs_full={r['JS_vs_full_bits']:.6g}, "
                        f"mean_active={r['mean_active_neurons']:.4f}, "
                        f"P_half+={r['P_half_or_more_active']:.4g}"
                    )
                self.log(f"\nDONE: {outdir.resolve()}")
                self.set_status(f"Finished: {outdir.name}")

            except Exception as exc:
                self.log("\nERROR:")
                self.log(str(exc))
                self.log(traceback.format_exc())
                self.set_status("Run failed — see log")

        threading.Thread(target=worker, daemon=True).start()

    def start_gate_validation(self, mode):
        try:
            bio=self.collect_bio(); hw=self.collect_hardware()
            scores=compute_mechanistic_scores(bio); gains=scores_to_layer_gains(scores,bio.hardware_gain)
            cases=make_gate_cases(set(ALL_GATES),gains)
            shots=int(self.local_shots_var.get()) if mode=="local" else int(hw.shots)
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror("Gate validation input error",str(exc)); return

        def worker():
            try:
                self.set_status(f"Running gate validation {mode}...")
                outdir=Path(f"V6_4_gate_validation_{mode}_{timestamp_string()}"); outdir.mkdir(parents=True,exist_ok=True)
                self.log(f"\n=== V6.6 GATE VALIDATION {mode.upper()} ===")
                self.log(f"Biology-linked gamma={gains.kir_damping_gamma:.6g}; threshold_gain={gains.threshold_gate_gain:.6g}; GG_phase={gains.glia_glia_phase_gain:.6g}; GG_XY={gains.glia_glia_exchange_gain:.6g}")
                if mode=="local":
                    summary,gs=run_gate_validation_local(cases,shots,outdir,gains,scores,self.log)
                else:
                    summary,gs=run_gate_validation_qpu(cases,hw,outdir,gains,scores,self.log)
                self.log("\nPer-gate max marginal MAE:")
                for _,r in gs.iterrows(): self.log(f"  {r['gate']}: {r['max_marginal_MAE']:.6g}")
                self.log(f"DONE: {outdir.resolve()}"); self.set_status(f"Gate validation finished: {outdir.name}")
            except Exception as exc:
                self.log("\nGATE VALIDATION ERROR:"); self.log(str(exc)); self.log(traceback.format_exc()); self.set_status("Gate validation failed")
        threading.Thread(target=worker,daemon=True).start()

    def run(self):
        self.root.mainloop()


# =============================================================================
# 9. Built-in V6.6 falsifiable gate validation
# =============================================================================

ALL_GATES = ("cry", "cnot", "cz", "zz", "xy", "ccry", "toffoli", "cswap", "damping", "reset")


@dataclass
class GateCase:
    name: str
    gate: str
    initial_bits: tuple[int, int, int] = (0, 0, 0)  # q0,q1,q2
    plus_qubits: tuple[int, ...] = ()
    theta: float | None = None
    gamma: float | None = None
    phase_readout_q: int | None = None
    expected_p1_q0: float | None = None
    expected_p1_q1: float | None = None
    expected_p1_q2: float | None = None
    biological_analogy: str = ""
    note: str = ""


def prepare_gate_case_state(program, case: GateCase, qp):
    for q, bit in enumerate(case.initial_bits):
        if int(bit):
            program << qp["RY"](q, math.pi)
    for q in case.plus_qubits:
        program << qp["H"](q)


def build_gate_case_program(case: GateCase, qp):
    prog = qp["QProg"]()
    prepare_gate_case_state(prog, case, qp)
    if case.gate == "cry":
        append_cry(prog, 0, 1, float(case.theta), qp)
    elif case.gate == "cnot":
        prog << qp["CNOT"](0, 1)
    elif case.gate == "cz":
        append_cz(prog, 0, 1, qp)
        prog << qp["H"](1)  # X-basis readout for target initially |+>
    elif case.gate == "zz":
        append_zz(prog, 0, 1, float(case.theta), qp)
    elif case.gate == "xy":
        append_xy_exchange(prog, 0, 1, float(case.theta), qp)
    elif case.gate == "ccry":
        append_ccry(prog, 0, 1, 2, float(case.theta), qp)
    elif case.gate == "toffoli":
        append_toffoli(prog, 0, 1, 2, qp)
    elif case.gate == "cswap":
        append_cswap(prog, 0, 1, 2, qp)
    elif case.gate == "damping":
        append_amplitude_damping_dilation(prog, 0, 1, float(case.gamma), qp)
    elif case.gate == "reset":
        append_amplitude_damping_dilation(prog, 0, 1, 1.0, qp)
    else:
        raise ValueError(case.gate)

    if case.phase_readout_q is not None:
        q = int(case.phase_readout_q)
        prog << qp["RZ"](q, -math.pi/2.0)
        prog << qp["H"](q)
    prog << qp["measure"]([0,1,2],[0,1,2])
    return prog


def make_gate_cases(selected_gates: set[str], gains: LayerGains | None = None) -> list[GateCase]:
    cases: list[GateCase] = []
    cry_theta = math.pi/2 if gains is None else clip(gains.neuron_to_glia_gain, 0.0, math.pi)
    zz_theta = math.pi/2 if gains is None else clip(gains.glia_glia_phase_gain, 0.0, math.pi)
    xy_theta = math.pi/4 if gains is None else clip(gains.glia_glia_exchange_gain, 0.0, math.pi/2)
    ccry_theta = math.pi/2 if gains is None else clip(math.pi*gains.threshold_gate_gain, 0.0, math.pi)
    gamma = 0.75 if gains is None else clip(gains.kir_damping_gamma, 0.0, 1.0)

    if "cry" in selected_gates:
        p = math.sin(cry_theta/2.0)**2
        cases += [
            GateCase("CRY_control0", "cry", (0,0,0), theta=cry_theta,
                     expected_p1_q0=0, expected_p1_q1=0, expected_p1_q2=0,
                     biological_analogy="inactive neuron leaves downstream glial/target rotation off"),
            GateCase("CRY_control1", "cry", (1,0,0), theta=cry_theta,
                     expected_p1_q0=1, expected_p1_q1=p, expected_p1_q2=0,
                     biological_analogy="active upstream neuron conditionally changes downstream activation probability",
                     note=f"biology-linked theta={cry_theta:.6g}"),
        ]
    if "cnot" in selected_gates:
        for c in (0,1):
            for t in (0,1):
                ot=t^c
                cases.append(GateCase(f"CNOT_{c}{t}_to_{c}{ot}","cnot",(c,t,0),
                    expected_p1_q0=c,expected_p1_q1=ot,expected_p1_q2=0,
                    biological_analogy="ideal conditional-flip limit of controlled neuroglial modulation"))
    if "cz" in selected_gates:
        cases += [
            GateCase("CZ_control0_phase_reference","cz",(0,0,0),plus_qubits=(1,),
                     expected_p1_q0=0,expected_p1_q1=0,expected_p1_q2=0,
                     biological_analogy="control-dependent phase with inactive control"),
            GateCase("CZ_control1_phase_flip","cz",(1,0,0),plus_qubits=(1,),
                     expected_p1_q0=1,expected_p1_q1=1,expected_p1_q2=0,
                     biological_analogy="active control flips target phase without direct population flip"),
        ]
    if "zz" in selected_gates:
        # For q1=|+>, append_zz gives RZ(+theta) if q0=0 and RZ(-theta) if q0=1.
        # RZ(-pi/2)+H readout gives P1=sin^2((phi-pi/2)/2).
        p0=math.sin((zz_theta-math.pi/2.0)/2.0)**2
        p1=math.sin((-zz_theta-math.pi/2.0)/2.0)**2
        cases += [
            GateCase("ZZ_control0_Yreadout","zz",(0,0,0),plus_qubits=(1,),theta=zz_theta,
                     phase_readout_q=1,expected_p1_q0=0,expected_p1_q1=p0,expected_p1_q2=0,
                     biological_analogy="shared-environment phase coupling"),
            GateCase("ZZ_control1_Yreadout","zz",(1,0,0),plus_qubits=(1,),theta=zz_theta,
                     phase_readout_q=1,expected_p1_q0=1,expected_p1_q1=p1,expected_p1_q2=0,
                     biological_analogy="shared-environment phase coupling with opposite conditional phase"),
        ]
    if "xy" in selected_gates:
        # Under U_XY(theta), a single excitation oscillates between q0 and q1:
        # P(stay)=cos^2(theta), P(exchange)=sin^2(theta).
        p_swap = math.sin(xy_theta)**2
        p_stay = math.cos(xy_theta)**2
        cases += [
            GateCase(
                "XY_10_partial_exchange", "xy", (1,0,0), theta=xy_theta,
                expected_p1_q0=p_stay, expected_p1_q1=p_swap, expected_p1_q2=0,
                biological_analogy=(
                    "gap-junction-like glial state exchange analogue: activity on one "
                    "glial node can spread to its coupled neighbor"
                ),
                note=f"biology-linked theta_XY={xy_theta:.6g}",
            ),
            GateCase(
                "XY_01_partial_exchange", "xy", (0,1,0), theta=xy_theta,
                expected_p1_q0=p_swap, expected_p1_q1=p_stay, expected_p1_q2=0,
                biological_analogy="symmetric bidirectional glial state-exchange analogue",
                note=f"biology-linked theta_XY={xy_theta:.6g}",
            ),
            GateCase(
                "XY_00_fixed", "xy", (0,0,0), theta=xy_theta,
                expected_p1_q0=0, expected_p1_q1=0, expected_p1_q2=0,
                biological_analogy="no excitation to exchange",
            ),
            GateCase(
                "XY_11_fixed", "xy", (1,1,0), theta=xy_theta,
                expected_p1_q0=1, expected_p1_q1=1, expected_p1_q2=0,
                biological_analogy="two occupied nodes preserve total excitation under XY exchange",
            ),
        ]
    if "ccry" in selected_gates:
        p=math.sin(ccry_theta/2.0)**2
        cases += [
            GateCase("CCRY_controls10_no_threshold","ccry",(1,0,0),theta=ccry_theta,
                     expected_p1_q0=1,expected_p1_q1=0,expected_p1_q2=0,
                     biological_analogy="one neuronal input alone does not cross shared-astrocyte coincidence rule"),
            GateCase("CCRY_controls11_threshold","ccry",(1,1,0),theta=ccry_theta,
                     expected_p1_q0=1,expected_p1_q1=1,expected_p1_q2=p,
                     biological_analogy="two neuronal inputs jointly recruit the shared glial target",
                     note=f"biology-linked theta={ccry_theta:.6g}"),
        ]
    if "toffoli" in selected_gates:
        for a in (0,1):
            for b in (0,1):
                for t in (0,1):
                    ot=t^(a&b)
                    cases.append(GateCase(f"TOFFOLI_{a}{b}{t}_to_{a}{b}{ot}","toffoli",(a,b,t),
                        expected_p1_q0=a,expected_p1_q1=b,expected_p1_q2=ot,
                        biological_analogy="hard-threshold limit of two-input astrocytic gating"))
    if "cswap" in selected_gates:
        cases += [
            GateCase("CSWAP_control0_01_stays_01","cswap",(0,0,1),
                     expected_p1_q0=0,expected_p1_q1=0,expected_p1_q2=1,
                     biological_analogy="inactive glial switch leaves two network channels unchanged"),
            GateCase("CSWAP_control1_01_to_10","cswap",(1,0,1),
                     expected_p1_q0=1,expected_p1_q1=1,expected_p1_q2=0,
                     biological_analogy="active glial switch enables controlled exchange"),
        ]
    if "damping" in selected_gates:
        cases += [
            GateCase("DAMPING_system1","damping",(1,0,0),gamma=gamma,
                     expected_p1_q0=1-gamma,expected_p1_q1=gamma,expected_p1_q2=0,
                     biological_analogy="Kir/homeostatic restoration transfers excitation to a clean environment ancilla",
                     note=f"KCNJ10:16-linked gamma={gamma:.6g}"),
            GateCase("DAMPING_system0","damping",(0,0,0),gamma=gamma,
                     expected_p1_q0=0,expected_p1_q1=0,expected_p1_q2=0,
                     biological_analogy="resting state is a fixed point of damping"),
        ]
    if "reset" in selected_gates:
        cases += [
            GateCase("RESET_system1_to_0","reset",(1,0,0),gamma=1,
                     expected_p1_q0=0,expected_p1_q1=1,expected_p1_q2=0,
                     biological_analogy="maximal restoration/reset limit"),
            GateCase("RESET_system0_stays_0","reset",(0,0,0),gamma=1,
                     expected_p1_q0=0,expected_p1_q1=0,expected_p1_q2=0,
                     biological_analogy="resting state remains resting"),
        ]
    return cases


def gate_case_analysis(case: GateCase, dist: dict[int,float]):
    p=neuron_marginals(dist,3)
    exp=np.array([
        np.nan if case.expected_p1_q0 is None else case.expected_p1_q0,
        np.nan if case.expected_p1_q1 is None else case.expected_p1_q1,
        np.nan if case.expected_p1_q2 is None else case.expected_p1_q2,
    ],float)
    mask=np.isfinite(exp)
    mae=float(np.mean(np.abs(p[mask]-exp[mask]))) if mask.any() else float("nan")
    return {
        "case":case.name,"gate":case.gate,
        "initial_q0":case.initial_bits[0],"initial_q1":case.initial_bits[1],"initial_q2":case.initial_bits[2],
        "plus_qubits":",".join(map(str,case.plus_qubits)),"theta":case.theta,"gamma":case.gamma,
        "P1_q0_observed":float(p[0]),"P1_q1_observed":float(p[1]),"P1_q2_observed":float(p[2]),
        "P1_q0_expected":case.expected_p1_q0,"P1_q1_expected":case.expected_p1_q1,"P1_q2_expected":case.expected_p1_q2,
        "marginal_MAE":mae,"biological_analogy":case.biological_analogy,"note":case.note,
    }


def gate_dist_rows(case_name: str, repeat: int, dist: dict[int,float]):
    return [{"case":case_name,"repeat":repeat,"state_int":int(i),"state_q2q1q0":f"{i:03b}","probability":float(p)}
            for i,p in sorted(dist.items())]


def save_gate_validation(outdir: Path, cases, mean_dists, raw_rows, mode, gains=None, scores=None):
    summary=pd.DataFrame([gate_case_analysis(next(c for c in cases if c.name==name),dist)
                          for name,dist in mean_dists.items()])
    summary.to_csv(outdir/"gate_validation_summary.csv",index=False)
    pd.DataFrame(raw_rows).to_csv(outdir/"gate_validation_probabilities.csv",index=False)
    gate_summary=(summary.groupby("gate",as_index=False)
                  .agg(cases=("case","count"),mean_marginal_MAE=("marginal_MAE","mean"),max_marginal_MAE=("marginal_MAE","max")))
    gate_summary.to_csv(outdir/"gate_validation_gate_summary.csv",index=False)
    payload={
        "version":VERSION,"mode":mode,"qubit_convention":"q0,q1,q2 logical; displayed states q2q1q0",
        "cases":[asdict(c) for c in cases],
        "layer_gains":None if gains is None else asdict(gains),
        "mechanistic_scores":None if scores is None else asdict(scores),
        "scientific_scope":"Effective quantum-formal gate validation; not evidence that biological glia are physical qubits.",
    }
    atomic_json_write(outdir/"gate_validation_manifest.json",payload)
    return summary,gate_summary


def run_gate_validation_local(cases, shots: int, outdir: Path, gains=None, scores=None, log=print):
    qp=import_qpanda_core(); qvm=qp["CPUQVM"](); mean_dists={}; raw=[]
    for i,case in enumerate(cases,1):
        log(f"[GATE CPUQVM] {i}/{len(cases)} {case.name}")
        prog=build_gate_case_program(case,qp); qvm.run(prog,shots=int(shots))
        dist=normalized_sparse_distribution(qvm.result().get_counts(),3)
        mean_dists[case.name]=dist; raw.extend(gate_dist_rows(case.name,0,dist))
    return save_gate_validation(outdir,cases,mean_dists,raw,"local",gains,scores)


def run_gate_validation_qpu(cases, hw: HardwareSettings, outdir: Path, gains=None, scores=None, log=print):
    qp=import_qpanda_core(); qcloud=import_qcloud(); key=get_api_key(); QCloudService=qcloud["QCloudService"]
    try: service=QCloudService(api_key=key)
    except TypeError: service=QCloudService(key)
    request=os.environ.get("QPANDA_QCLOUD_BACKEND",hw.backend or "auto").strip() or "auto"
    backend_name,backend=choose_real_backend(service,request,3,log)
    log(f"Selected real QPU for gate validation: {backend_name}")
    # Gate circuits are only 3 logical qubits, so small-circuit parallelism is safe;
    # scheduler retry logic from V6.2.1 is still reused unchanged.
    ghw=replace(hw,backend=backend_name,batch_size=max(1,min(hw.batch_size,len(cases))),parallel_jobs=max(1,hw.parallel_jobs))
    blocks=choose_physical_block_pool(backend,3,ghw,log)
    records=[]
    for rep in range(int(ghw.repeats)):
        for case in cases:
            records.append({"repeat":rep,"condition":case.name,"program":build_gate_case_program(case,qp),"measured_bits":3})
    repeat_dists,_=submit_programs_in_waves(backend,records,blocks,ghw,qcloud,outdir,log)
    mean_dists=mean_sparse_distributions(repeat_dists,[c.name for c in cases])
    raw=[]
    for (rep,name),dist in sorted(repeat_dists.items()): raw.extend(gate_dist_rows(name,rep,dist))
    return save_gate_validation(outdir,cases,mean_dists,raw,"qpu",gains,scores)


# =============================================================================
# 10. CLI no-GUI helpers
# =============================================================================

def project_from_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    b = data["biological_input"]
    bio = BiologicalInput(
        proteins={k: float(v) for k,v in b["proteins"].items()},
        pH=float(b.get("pH",7.4)),
        K_out_mM=float(b.get("K_out_mM",3.5)),
        Vm_mV=float(b.get("Vm_mV",-80)),
        neuronal_drive=float(b.get("neuronal_drive",1.0)),
        hardware_gain=float(b.get("hardware_gain",1.0)),
        scale_mode=b.get("scale_mode","relative_linear"),
        v3_summary=b.get("v3_summary",""),
        kir_assembly_eta=float(b.get("kir_assembly_eta",0.75)),
    )
    n = data["network"]
    cfg = NetworkConfig(
        total_qubits=int(n["total_qubits"]),
        layers=int(n["layers"]),
        layer_dt_ms=float(n.get("layer_dt_ms",10.0)),
        neural_inputs=[
            NeuralInputEvent(
                int(x["layer"]), int(x["neuron"]),
                float(x["amplitude"]), float(x.get("phase_rad",0.0)),
                str(x.get("event_kind", "evidence"))
            ) for x in n.get("neural_inputs",[])
        ],
        neuron_to_neuron=[
            DirectedEdge(int(x["source"]),int(x["target"]),float(x["weight"]))
            for x in n.get("neuron_to_neuron",[])
        ],
        neuron_to_glia=[
            DirectedEdge(int(x["source"]),int(x["target"]),float(x["weight"]))
            for x in n.get("neuron_to_glia",[])
        ],
        glia_glia=[
            UndirectedEdge(
                int(x["a"]), int(x["b"]), float(x["weight"]),
                None if x.get("exchange_weight") is None else float(x["exchange_weight"])
            )
            for x in n.get("glia_glia",[])
        ],
        glia_to_neuron=[
            DirectedEdge(int(x["source"]),int(x["target"]),float(x["weight"]))
            for x in n.get("glia_to_neuron",[])
        ],
        enable_shared_glia_threshold=bool(n.get("enable_shared_glia_threshold", True)),
        max_threshold_pairs_per_glia=int(n.get("max_threshold_pairs_per_glia", 1)),
        activity_gate_shared_glia_threshold_targets=[int(x) for x in n.get("activity_gate_shared_glia_threshold_targets", [])],
        activity_gate_glia_microdomain_targets=[int(x) for x in n.get("activity_gate_glia_microdomain_targets", [])],
        tripartite_synapses=[
            TripartiteSynapse(
                int(x["sensory_neuron"]), int(x["context_glia"]),
                int(x["target_neuron"]), float(x.get("weight",1.0)),
                bool(x.get("require_phasic_sensory", True))
            ) for x in n.get("tripartite_synapses",[])
        ],
        enable_tripartite_synapses=bool(n.get("enable_tripartite_synapses", True)),
        enable_controlled_exchange=bool(n.get("enable_controlled_exchange", False)),
        exchange_activation_threshold=float(n.get("exchange_activation_threshold", 0.75)),
        enable_glia_exchange=bool(n.get("enable_glia_exchange", True)),
        scale_glia_exchange_by_layers=bool(n.get("scale_glia_exchange_by_layers", False)),
        activity_gate_glia_exchange=bool(n.get("activity_gate_glia_exchange", True)),
        enable_kir_damping=bool(n.get("enable_kir_damping", False)),
        initialize_neural_h=bool(n.get("initialize_neural_h", True)),
        initialize_glial_baseline=bool(n.get("initialize_glial_baseline", True)),
        glial_initial_probabilities=(
            None if n.get("glial_initial_probabilities") is None
            else [float(x) for x in n.get("glial_initial_probabilities", [])]
        ),
        neural_readout_mode=str(n.get("neural_readout_mode", "phase")),
    )
    h = data.get("hardware",{})
    hw = HardwareSettings(
        backend=h.get("backend","auto"),
        shots=int(h.get("shots",1000)),
        repeats=int(h.get("repeats",3)),
        batch_size=int(h.get("batch_size",5)),
        parallel_jobs=int(h.get("parallel_jobs",1)),
        mapping=bool(h.get("mapping",True)),
        optimization=bool(h.get("optimization",True)),
        amend=bool(h.get("amend",True)),
        use_specified_blocks=bool(h.get("use_specified_blocks",False)),
        prefer_disjoint_blocks=bool(h.get("prefer_disjoint_blocks",False)),
        scheduler_retries=int(h.get("scheduler_retries",3)),
        scheduler_retry_wait_s=int(h.get("scheduler_retry_wait_s",30)),
        force_parallel_large_qpu=bool(h.get("force_parallel_large_qpu",False)),
        local_safe_qubits=int(h.get("local_safe_qubits",20)),
    )
    return bio, cfg, hw, int(data.get("local_shots",20000))


def write_example_project(path: Path):
    bio = BiologicalInput(proteins=dict(REFERENCE_PROTEINS))
    cfg = default_network(16,3)
    hw = HardwareSettings()
    payload = {
        "version": VERSION,
        "biological_input": asdict(bio),
        "network": {
            "total_qubits": cfg.total_qubits,
            "layers": cfg.layers,
            "layer_dt_ms": cfg.layer_dt_ms,
            "neural_inputs": [asdict(x) for x in cfg.neural_inputs or []],
            "neuron_to_neuron": [asdict(x) for x in cfg.neuron_to_neuron or []],
            "neuron_to_glia": [asdict(x) for x in cfg.neuron_to_glia or []],
            "glia_glia": [asdict(x) for x in cfg.glia_glia or []],
            "glia_to_neuron": [asdict(x) for x in cfg.glia_to_neuron or []],
            "enable_shared_glia_threshold": cfg.enable_shared_glia_threshold,
            "max_threshold_pairs_per_glia": cfg.max_threshold_pairs_per_glia,
            "enable_controlled_exchange": cfg.enable_controlled_exchange,
            "exchange_activation_threshold": cfg.exchange_activation_threshold,
            "enable_glia_exchange": cfg.enable_glia_exchange,
            "scale_glia_exchange_by_layers": cfg.scale_glia_exchange_by_layers,
            "activity_gate_glia_exchange": cfg.activity_gate_glia_exchange,
            "enable_kir_damping": cfg.enable_kir_damping,
            "initialize_neural_h": getattr(cfg, "initialize_neural_h", True),
            "initialize_glial_baseline": getattr(cfg, "initialize_glial_baseline", True),
            "neural_readout_mode": getattr(cfg, "neural_readout_mode", "phase"),
        },
        "hardware": asdict(hw),
        "local_shots": 20000,
    }
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="V7.1 neuroglial gate / dual-readout real-QPU emulator"
    )
    parser.add_argument("--project", default=None)
    parser.add_argument("--mode", choices=["gui","resource","local","qpu","gates-local","gates-qpu","gates-manifest"], default="gui")
    parser.add_argument("--gates", nargs="+", choices=ALL_GATES, default=list(ALL_GATES))
    parser.add_argument("--write-example", default=None)
    args = parser.parse_args()

    if args.write_example:
        p = Path(args.write_example)
        write_example_project(p)
        print("Wrote:", p.resolve())
        return 0

    if args.mode == "gui":
        V64App().run()
        return 0

    # Gate validation can run on reference biology without a project, or use a
    # project so KCNJ10:16/ITPR2/GJA1 parameters set theta/gamma.
    if args.mode.startswith("gates-"):
        if args.project:
            bio, cfg, hw, local_shots = project_from_json(Path(args.project))
        else:
            bio=BiologicalInput(proteins=dict(REFERENCE_PROTEINS))
            cfg=default_network(16,3); hw=HardwareSettings(); local_shots=20000
        scores=compute_mechanistic_scores(bio); gains=scores_to_layer_gains(scores,bio.hardware_gain)
        cases=make_gate_cases(set(args.gates),gains)
        gmode=args.mode.split("-",1)[1]
        outdir=Path(f"V6_4_gate_validation_{gmode}_{timestamp_string()}"); outdir.mkdir(parents=True,exist_ok=True)
        if gmode=="manifest":
            atomic_json_write(outdir/"gate_validation_manifest.json",{
                "version":VERSION,"cases":[asdict(c) for c in cases],
                "mechanistic_scores":asdict(scores),"layer_gains":asdict(gains)})
        elif gmode=="local":
            run_gate_validation_local(cases,local_shots,outdir,gains,scores,print)
        else:
            run_gate_validation_qpu(cases,hw,outdir,gains,scores,print)
        print("DONE:",outdir.resolve()); return 0

    if not args.project:
        raise SystemExit("--project is required for resource/local/qpu network mode.")

    bio, cfg, hw, local_shots = project_from_json(Path(args.project))
    validate_network_config(cfg)
    scores = compute_mechanistic_scores(bio)
    gains = scores_to_layer_gains(scores, bio.hardware_gain)

    if args.mode == "resource":
        print(json.dumps(circuit_resource_estimate(cfg), indent=2))
        print(json.dumps(asdict(gains), indent=2))
        return 0

    variants = build_condition_variants(cfg)
    outdir = Path(f"V6_4_{args.mode}_{timestamp_string()}")
    outdir.mkdir(parents=True, exist_ok=True)
    if args.mode == "local":
        mean_dists = run_local(
            variants, gains, local_shots, hw.local_safe_qubits, outdir, print
        )
        summarize_distributions(
            mean_dists, cfg.n_neurons, outdir, "V6_4_simulator"
        )
    else:
        mean_dists, _, _ = run_qpu(
            variants, gains, hw, outdir, print
        )
        summarize_distributions(
            mean_dists, cfg.n_neurons, outdir, "V6_4_QPU"
        )
    print("DONE:", outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
