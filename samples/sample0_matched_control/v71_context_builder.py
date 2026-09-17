"""V7.1 builder with one optional hook, otherwise copied verbatim.
Bound to original V7.1 globals by v71_bridge.py; no global monkey-patching.
The hook is a MODEL EXTENSION, not a claim of biological quantum coherence.
"""
from __future__ import annotations

def build_layered_program(
    cfg: NetworkConfig,
    gains: LayerGains,
    qp=None,
    upto_layer: int | None = None,
    measure_glia: bool = False,
    neural_readout_mode: str | None = None,
    context_hook=None,
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

    if bool(getattr(cfg, "initialize_glial_baseline", True)):
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

        # Sample0 extension hook: after local context phase, before feedback.
        # None exactly preserves V7.1; the hook reuses existing local N->G edges.
        if context_hook is not None:
            context_hook(prog, cfg, gains, qp, layer, evidence_by_layer)

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
