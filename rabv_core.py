"""
rabv_core.py — NeuroViral Lab RABV Engine  v1.0
Multi-compartment viral neuroinvasion ODE model

Architecture mirrors bbb_core.py exactly:
  - Dataclass input/output schemas
  - Euler integration (same dt pattern as BBB engine)
  - FastAPI-compatible: called by rabv_main.py on port 8001
  - JS engine calls /simulate when Python backend available,
    falls back to heuristics when offline

Viral compartments (normalised 0-1 load):
  V_wound   — inoculation site (muscle/subcutaneous)
  V_nmj     — neuromuscular junction (G-protein binding + endocytosis)
  V_pns     — peripheral axon (dynein retrograde transport)
  V_drg     — dorsal root / autonomic ganglia (local amplification)
  V_sc      — spinal cord entry zone
  V_bs      — brainstem (cranial nerve nuclei, autonomic centres)
  V_limbic  — limbic system (hippocampus, amygdala, hypothalamus)
  V_cortex  — cortex + thalamus
  V_sal     — salivary glands (centrifugal anterograde)

Immune compartments:
  I_innate  — innate immune (IFN, NK, macrophage)
  I_adapt   — adaptive immune (neutralising Ab, T cells)

All rates /hour. Viral load normalised (0-1 per compartment).
"""

import math
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ── BIOLOGICAL CONSTANTS ─────────────────────────────────────────────────────

# Transport (mm/day → mm/hour internally)
V_DYNEIN_BASE_MM_DAY  = 200.0   # baseline retrograde
V_P75NTR_MAX_BOOST    = 2.8     # p75NTR co-transport ceiling multiplier
V_ANTEROGRADE_MM_DAY  = 300.0   # kinesin anterograde

# Replication (Hill function)
R_REPLICATE_BASE      = 0.065   # /h  — t½ ~10h per neuron cycle
REPLICATION_HILL_N    = 2.0     # cooperativity
REPLICATION_K_HALF    = 0.25    # half-max viral load
MAX_VIRAL_LOAD        = 1.0

# Immune clearance rates (/h)
K_INNATE_CLEAR        = 0.035
K_ADAPT_CLEAR         = 0.110
K_INNATE_RISE         = 0.055   # innate activation kinetics
K_INNATE_DECAY        = 0.018
K_ADAPT_RISE          = 0.007   # days-scale adaptive rise
K_ADAPT_DECAY         = 0.003

# P protein immune evasion fractions
P_IFN_BLOCK           = 0.90    # TBK1 → IFN-β suppression (wild-type)
P_TBK1_BLOCK          = 0.85    # condensate formation block

# N protein: 11-mer RNP assembly boosts effective replication
N_RNP_REP_BOOST       = 1.22

# M protein: NF-κB attenuation (BBB preservation)
M_NFKB_ATT            = 0.32

# Intra-CNS trans-synaptic spread (/h)
K_TRANSYN             = 0.022

# Wound → NMJ entry (/h)
K_WOUND_TO_NMJ        = 0.18
K_WOUND_DECAY         = 0.04

# Centrifugal salivary spread (/h)
K_SAL_SPREAD          = 0.012

# Nerve distances (mm)
NERVE_DISTANCES = {
    "face": 200, "hand": 900, "upper_arm": 500,
    "torso": 300, "thigh": 600, "foot": 1200,
}

# nAChR density relative — higher density = faster NMJ entry
NACHR_DENSITY = {
    "face": 1.85, "hand": 1.20, "upper_arm": 1.00,
    "torso": 0.70, "thigh": 0.90, "foot": 0.80,
}

# Initial inoculum (fraction of max)
DOSE_V0 = {"low": 0.04, "med": 0.18, "high": 0.60}

# Wound wash inoculum reduction
WOUND_WASH = {
    "within_15min": 0.07,
    "within_1h":    0.18,
    "within_6h":    0.42,
    "none":         1.00,
}

# Peripheral spread probability by dose
PERIPH_PROB = {"low": 0.05, "med": 0.15, "high": 0.45}


# ── INPUT SCHEMA ─────────────────────────────────────────────────────────────
@dataclass
class RABVInput:
    # Infection
    bite_location:       str   = "foot"    # face|hand|upper_arm|torso|thigh|foot
    bite_depth:          str   = "muscle"  # superficial|muscle|deep
    viral_dose:          str   = "med"     # low|med|high
    wound_washing:       str   = "none"    # within_15min|within_1h|within_6h|none

    # Transport
    base_velocity_mm_day: float = 200.0
    p75_active:          bool  = True
    replication_lag_h:   float = 9.0

    # Viral protein toggles (mirror JS simulator)
    prot_G: bool = True   # receptor binding / NMJ entry
    prot_P: bool = True   # dynein transport + IFN block
    prot_N: bool = True   # RNP encapsidation
    prot_M: bool = True   # NF-κB attenuation
    prot_L: bool = True   # RdRp replication

    # Host immune
    vaccinated:          bool  = False
    pep_given:           bool  = False
    immunocompromised:   bool  = False
    pep_day:             float = 0.0     # day PEP started
    pep_hrig_given:      bool  = False   # passive HRIG

    # Therapeutic interventions
    p_inhibitor_active:   bool  = False
    p_inhibitor_efficacy: float = 0.75
    p_inhibitor_day:      float = 0.0
    n_dn_active:          bool  = False
    n_dn_efficacy:        float = 0.70
    n_dn_day:             float = 0.0
    favipiravir_active:   bool  = False
    favipiravir_efficacy: float = 0.40
    favipiravir_day:      float = 0.0

    # Simulation control
    t_days:              float = 60.0
    dt_hours:            float = 0.25
    output_interval_h:   float = 2.0


# ── OUTPUT SCHEMA ─────────────────────────────────────────────────────────────
@dataclass
class RABVOutput:
    # Timing milestones (days)
    day_cns_entry:     float = 0.0
    day_brainstem:     float = 0.0
    day_limbic:        float = 0.0
    day_symptoms:      float = 0.0
    day_fatal:         float = 0.0
    transport_days:    float = 0.0
    incubation_days:   float = 0.0

    # Burden
    burden_at_end:     float = 0.0
    peak_cns_burden:   float = 0.0

    # BBB state at end
    bbb_tj_final:      float = 95.0
    bbb_nfkb_final:    float = 5.0

    # Immune peaks
    innate_peak:       float = 0.0
    adapt_peak:        float = 0.0
    pep_effective:     bool  = False

    # Outcome
    outcome:           str   = "fatal"
    outcome_prob:      float = 0.0

    # Spread
    peripheral_spread_prob: float = 0.15

    # Phase end times (days): [local, pns_end, cns_invasion, centrifugal]
    phase_ends:        List[float] = field(default_factory=list)

    # Time-course arrays
    timeline_days:     List[float] = field(default_factory=list)
    timeline_wound:    List[float] = field(default_factory=list)
    timeline_pns:      List[float] = field(default_factory=list)
    timeline_sc:       List[float] = field(default_factory=list)
    timeline_bs:       List[float] = field(default_factory=list)
    timeline_limbic:   List[float] = field(default_factory=list)
    timeline_cortex:   List[float] = field(default_factory=list)
    timeline_sal:      List[float] = field(default_factory=list)
    timeline_innate:   List[float] = field(default_factory=list)
    timeline_adapt:    List[float] = field(default_factory=list)
    timeline_bbb_tj:   List[float] = field(default_factory=list)
    timeline_bbb_nfkb: List[float] = field(default_factory=list)

    # Final compartment snapshot
    compartments_final: Dict[str, float] = field(default_factory=dict)

    # Metadata
    confidence_note:   str   = ""
    engine_version:    str   = "1.0"
    elapsed_ms:        float = 0.0


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _transport_params(inp: RABVInput) -> dict:
    dist  = NERVE_DISTANCES.get(inp.bite_location, 1200)
    nachr = NACHR_DENSITY.get(inp.bite_location, 1.0)

    base_v_h = inp.base_velocity_mm_day / 24.0
    if inp.p75_active:
        p75 = min(V_P75NTR_MAX_BOOST, (V_P75NTR_MAX_BOOST * 200.0) / max(inp.base_velocity_mm_day, 12.0))
    else:
        p75 = 1.0
    eff_v_h = min(400.0 / 24.0, base_v_h * p75)

    depth_h = {"superficial": 120.0, "muscle": 48.0, "deep": 12.0}.get(inp.bite_depth, 48.0)
    dose_f  = {"low": 1.60, "med": 1.00, "high": 0.55}.get(inp.viral_dose, 1.00)
    local_h = depth_h * dose_f

    transport_h = dist / eff_v_h
    wash        = WOUND_WASH.get(inp.wound_washing, 1.0)
    v0          = DOSE_V0.get(inp.viral_dose, 0.18) * wash

    return dict(
        dist=dist, nachr=nachr, eff_v_h=eff_v_h, p75=p75,
        local_h=local_h, transport_h=transport_h,
        v0=v0, periph=PERIPH_PROB.get(inp.viral_dose, 0.15),
    )


def _pf(inp: RABVInput, t_h: float) -> dict:
    """Effective protein activity factors at time t_h."""

    # G — NMJ entry
    g = 1.0 if inp.prot_G else 0.02

    # P — transport + immune evasion
    if inp.p_inhibitor_active and t_h >= inp.p_inhibitor_day * 24.0:
        eff = inp.p_inhibitor_efficacy
        p_tr     = max(0.05, 1.0 - eff)
        p_ifn    = P_IFN_BLOCK  * (1.0 - eff)
        p_tkb    = P_TBK1_BLOCK * (1.0 - eff)
    else:
        p_tr  = 1.0 if inp.prot_P else 0.08
        p_ifn = P_IFN_BLOCK  if inp.prot_P else 0.0
        p_tkb = P_TBK1_BLOCK if inp.prot_P else 0.0

    # N — RNP boost
    if inp.n_dn_active and t_h >= inp.n_dn_day * 24.0:
        n_boost = max(0.25, 1.0 + (N_RNP_REP_BOOST - 1.0) * (1.0 - inp.n_dn_efficacy))
    else:
        n_boost = N_RNP_REP_BOOST if inp.prot_N else 0.55

    # M — NF-κB attenuation
    m_att = M_NFKB_ATT if inp.prot_M else 0.0

    # L — RdRp
    if inp.favipiravir_active and t_h >= inp.favipiravir_day * 24.0:
        l_eff = max(0.05, 1.0 - inp.favipiravir_efficacy)
    else:
        l_eff = 1.0 if inp.prot_L else 0.04

    return dict(g=g, p_tr=p_tr, p_ifn=p_ifn, p_tkb=p_tkb,
                n_boost=n_boost, m_att=m_att, l_eff=l_eff)


def _rep(v: float, n_boost: float, l_eff: float) -> float:
    """Hill-function replication rate."""
    if v <= 0:
        return 0.0
    v = min(v, MAX_VIRAL_LOAD)
    h = v**REPLICATION_HILL_N / (REPLICATION_K_HALF**REPLICATION_HILL_N + v**REPLICATION_HILL_N)
    return R_REPLICATE_BASE * n_boost * l_eff * h * (1.0 - v / MAX_VIRAL_LOAD)


def _imm_cl(v: float, inn: float, ada: float, p_ifn: float, p_tkb: float) -> float:
    """Immune clearance. P protein suppresses innate arm."""
    eff_inn = inn * (1.0 - p_ifn) * (1.0 - p_tkb * 0.5)
    return (K_INNATE_CLEAR * eff_inn + K_ADAPT_CLEAR * ada) * v


def _bbb(v_bs: float, v_lim: float, inn: float, m_att: float, p_ifn: float) -> dict:
    """
    Compute BBB state from CNS burden + immune state.
    Wild-type RABV: P active → IFN low → NF-κB low → TJ intact.
    Therapeutic P neutralisation → IFN rises → NF-κB rises → mild BBB opening.
    """
    cns = min(1.0, v_bs + v_lim * 0.5)
    nfkb_raw = cns * inn * 0.75
    nfkb_eff = nfkb_raw * (1.0 - m_att) * (1.0 - p_ifn * 0.55)
    tj   = max(55.0, 100.0 - nfkb_eff * 45.0 - cns * 12.0)
    nfkb = min(100.0, nfkb_eff * 100.0)
    return dict(tj=tj, nfkb=nfkb)


def _clamp(v: float) -> float:
    return max(0.0, min(MAX_VIRAL_LOAD, v))


# ── EULER INTEGRATION ────────────────────────────────────────────────────────
def run_euler(inp: RABVInput) -> dict:
    tp = _transport_params(inp)
    dt = inp.dt_hours
    T_h = inp.t_days * 24.0
    steps = int(T_h / dt) + 1
    sample_n = max(1, int(inp.output_interval_h / dt))

    # Initial compartments
    V_wound = tp["v0"]
    V_nmj = V_pns = V_drg = V_sc = 0.0
    V_bs = V_lim = V_ctx = V_sal = 0.0

    # Immune
    I_inn = 0.02
    I_ada = (0.40 if inp.vaccinated else 0.0) + (0.55 if inp.pep_hrig_given else 0.0)
    I_mem = 0.50 if inp.vaccinated else 0.0
    I_ada = min(1.0, I_ada)

    # Axon progress: fraction of nerve traversed (0→1 = CNS entry)
    axon_prog   = 0.0
    spd_frac_h  = tp["eff_v_h"] / tp["dist"]   # fraction of total nerve / hour

    # Replication lag
    rep_lag = inp.replication_lag_h
    pep_activated = False

    # Output lists
    t_arr = []; w_arr = []; pns_arr = []; sc_arr = []
    bs_arr = []; lim_arr = []; ctx_arr = []; sal_arr = []
    inn_arr = []; ada_arr = []; tj_arr = []; nfkb_arr = []

    # Milestones
    day_cns = day_bs = day_lim = day_sym = day_fat = None
    peak_cns = 0.0
    bbb_last = dict(tj=95.0, nfkb=5.0)

    t = 0.0

    for step in range(steps):
        t_d = t / 24.0
        pf  = _pf(inp, t)

        # PEP activation
        if inp.pep_given and not pep_activated and t_d >= inp.pep_day:
            pep_activated = True
            I_ada = min(1.0, I_ada + 0.15)
            I_inn = min(1.0, I_inn + 0.08)

        # Replication active?
        if rep_lag > 0:
            rep_lag = max(0.0, rep_lag - dt)
            rep_on = False
        else:
            rep_on = True

        # ── Immune dynamics ─────────────────────────────────────────────────
        total_v = V_wound + V_nmj + V_sc + V_bs + V_lim + V_ctx
        inn_stim = total_v * (1.0 - pf["p_ifn"])
        dI_inn = K_INNATE_RISE * inn_stim * (1.0 - I_inn) - K_INNATE_DECAY * I_inn
        if inp.immunocompromised:
            dI_inn *= 0.25

        pep_boost = 0.0
        if pep_activated and t_d <= inp.pep_day + 28:
            pep_boost = 0.022 * math.exp(-0.004 * (t_d - inp.pep_day) * 24.0)
        ada_stim = max(0.0, I_inn - 0.08)
        dI_ada = K_ADAPT_RISE * ada_stim + pep_boost - K_ADAPT_DECAY * I_ada
        if I_ada < I_mem and I_inn > 0.10:
            dI_ada += K_ADAPT_RISE * I_mem * 2.5   # memory recall
        if inp.immunocompromised:
            dI_ada *= 0.18

        # ── Wound ───────────────────────────────────────────────────────────
        r_w  = _rep(V_wound, pf["n_boost"], pf["l_eff"]) if rep_on else 0.0
        cl_w = _imm_cl(V_wound, I_inn, I_ada, pf["p_ifn"], pf["p_tkb"])
        to_nmj = K_WOUND_TO_NMJ * V_wound * pf["g"] * tp["nachr"]
        dV_w = r_w - to_nmj - K_WOUND_DECAY * V_wound - cl_w

        # ── NMJ ─────────────────────────────────────────────────────────────
        r_nmj  = _rep(V_nmj, pf["n_boost"], pf["l_eff"]) if rep_on else 0.0
        cl_nmj = _imm_cl(V_nmj, I_inn, I_ada, pf["p_ifn"], pf["p_tkb"])
        nmj_to_axon = V_nmj * spd_frac_h * pf["p_tr"] * 0.55
        dV_nmj = to_nmj + r_nmj - nmj_to_axon - cl_nmj

        # ── Axon progress + PNS load ────────────────────────────────────────
        if axon_prog < 1.0 and V_nmj > 0.004:
            d_prog = spd_frac_h * pf["p_tr"] * dt
        else:
            d_prog = 0.0

        r_pns  = _rep(V_pns * 0.25, pf["n_boost"], pf["l_eff"]) * 0.25 if rep_on else 0.0
        cl_pns = _imm_cl(V_pns, I_inn * 0.35, I_ada * 0.25, pf["p_ifn"], pf["p_tkb"])
        pns_to_sc = V_pns * 0.035 if axon_prog >= 0.95 else 0.0
        dV_pns = nmj_to_axon + r_pns - pns_to_sc - cl_pns

        # ── DRG (local amplification at ganglia) ────────────────────────────
        drg_in = pns_to_sc * 0.14
        r_drg  = _rep(V_drg, pf["n_boost"], pf["l_eff"]) if rep_on else 0.0
        cl_drg = _imm_cl(V_drg, I_inn * 0.55, I_ada, pf["p_ifn"], pf["p_tkb"])
        dV_drg = drg_in + r_drg - 0.045 * V_drg - cl_drg

        # ── Spinal cord ──────────────────────────────────────────────────────
        sc_in  = pns_to_sc * 0.86 + V_drg * 0.038
        r_sc   = _rep(V_sc, pf["n_boost"], pf["l_eff"]) if rep_on else 0.0
        cl_sc  = _imm_cl(V_sc, I_inn * 0.48, I_ada * 0.65, pf["p_ifn"], pf["p_tkb"])
        sc_to_bs = V_sc * K_TRANSYN
        dV_sc  = sc_in + r_sc - sc_to_bs - cl_sc

        # ── Brainstem ────────────────────────────────────────────────────────
        r_bs   = _rep(V_bs, pf["n_boost"], pf["l_eff"]) if rep_on else 0.0
        cl_bs  = _imm_cl(V_bs, I_inn * 0.42, I_ada * 0.58, pf["p_ifn"], pf["p_tkb"])
        bs_to_lim = V_bs * K_TRANSYN * 1.25
        bs_to_sal = V_bs * K_SAL_SPREAD
        dV_bs  = sc_to_bs + r_bs - bs_to_lim - bs_to_sal - cl_bs

        # ── Limbic ───────────────────────────────────────────────────────────
        r_lim  = _rep(V_lim, pf["n_boost"], pf["l_eff"]) if rep_on else 0.0
        cl_lim = _imm_cl(V_lim, I_inn * 0.38, I_ada * 0.52, pf["p_ifn"], pf["p_tkb"])
        lim_to_ctx = V_lim * K_TRANSYN * 0.75
        dV_lim = bs_to_lim + r_lim - lim_to_ctx - cl_lim

        # ── Cortex ───────────────────────────────────────────────────────────
        r_ctx  = _rep(V_ctx, pf["n_boost"], pf["l_eff"]) if rep_on else 0.0
        cl_ctx = _imm_cl(V_ctx, I_inn * 0.32, I_ada * 0.45, pf["p_ifn"], pf["p_tkb"])
        dV_ctx = lim_to_ctx + r_ctx - cl_ctx

        # ── Salivary ─────────────────────────────────────────────────────────
        r_sal  = _rep(V_sal, pf["n_boost"], pf["l_eff"]) if rep_on else 0.0
        dV_sal = bs_to_sal + r_sal - 0.025 * V_sal

        # ── BBB state ────────────────────────────────────────────────────────
        bbb_last = _bbb(V_bs, V_lim, I_inn, pf["m_att"], pf["p_ifn"])

        # ── Euler update ─────────────────────────────────────────────────────
        V_wound = _clamp(V_wound + dt * dV_w)
        V_nmj   = _clamp(V_nmj   + dt * dV_nmj)
        V_pns   = _clamp(V_pns   + dt * dV_pns)
        V_drg   = _clamp(V_drg   + dt * dV_drg)
        V_sc    = _clamp(V_sc    + dt * dV_sc)
        V_bs    = _clamp(V_bs    + dt * dV_bs)
        V_lim   = _clamp(V_lim   + dt * dV_lim)
        V_ctx   = _clamp(V_ctx   + dt * dV_ctx)
        V_sal   = _clamp(V_sal   + dt * dV_sal)

        I_inn = max(0.02, min(1.0, I_inn + dt * dI_inn))
        I_ada = max(0.00, min(1.0, I_ada + dt * dI_ada))
        axon_prog = min(1.0, axon_prog + d_prog)

        # ── Milestones ───────────────────────────────────────────────────────
        if day_cns is None and V_sc    > 0.01: day_cns = t_d
        if day_bs  is None and V_bs    > 0.04: day_bs  = t_d
        if day_lim is None and V_lim   > 0.04: day_lim = t_d
        if day_sym is None and (V_bs > 0.14 or V_lim > 0.18): day_sym = t_d
        if day_fat is None and V_bs > 0.42 and V_lim > 0.32:  day_fat = t_d

        cns_total = V_sc + V_bs + V_lim + V_ctx
        if cns_total > peak_cns:
            peak_cns = cns_total

        # ── Sample ───────────────────────────────────────────────────────────
        if step % sample_n == 0:
            t_arr.append(round(t_d, 3))
            w_arr.append(round(V_wound, 5))
            pns_arr.append(round(V_pns, 5))
            sc_arr.append(round(V_sc, 5))
            bs_arr.append(round(V_bs, 5))
            lim_arr.append(round(V_lim, 5))
            ctx_arr.append(round(V_ctx, 5))
            sal_arr.append(round(V_sal, 5))
            inn_arr.append(round(I_inn, 4))
            ada_arr.append(round(I_ada, 4))
            tj_arr.append(round(bbb_last["tj"],   2))
            nfkb_arr.append(round(bbb_last["nfkb"], 2))

        t += dt

    # ── Post-integration ─────────────────────────────────────────────────────
    sim_end = inp.t_days
    day_cns = day_cns or sim_end
    day_bs  = day_bs  or sim_end
    day_lim = day_lim or sim_end
    day_sym = day_sym or sim_end
    day_fat = day_fat or (sim_end + 7.0)

    pep_effective = inp.pep_given and inp.pep_day < day_cns and I_ada > 0.32

    final_cns = V_sc + V_bs + V_lim + V_ctx
    if final_cns < 0.04 and day_cns == sim_end:
        outcome, outcome_prob = "prevented",  0.92
    elif final_cns < 0.12 and I_ada > 0.45:
        outcome, outcome_prob = "controlled", 0.68
    elif final_cns < 0.38 and (inp.pep_given or inp.p_inhibitor_active):
        outcome, outcome_prob = "salvage",    0.28
    else:
        outcome, outcome_prob = "fatal",      0.03

    phase_ends = [
        round(tp["local_h"]  / 24.0, 2),
        round(day_cns, 2),
        round(day_sym, 2),
        round(day_sym + 5.0, 2),
    ]

    return dict(
        day_cns_entry      = round(day_cns, 2),
        day_brainstem      = round(day_bs,  2),
        day_limbic         = round(day_lim, 2),
        day_symptoms       = round(day_sym, 2),
        day_fatal          = round(day_fat, 2),
        transport_days     = round(tp["transport_h"] / 24.0, 2),
        incubation_days    = round(day_sym, 2),
        burden_at_end      = round(min(1.0, final_cns), 4),
        peak_cns_burden    = round(min(1.0, peak_cns), 4),
        bbb_tj_final       = round(bbb_last["tj"],   2),
        bbb_nfkb_final     = round(bbb_last["nfkb"], 2),
        innate_peak        = round(max(inn_arr) if inn_arr else 0.0, 4),
        adapt_peak         = round(max(ada_arr) if ada_arr else 0.0, 4),
        pep_effective      = pep_effective,
        outcome            = outcome,
        outcome_prob       = round(outcome_prob, 3),
        peripheral_spread_prob = tp["periph"],
        phase_ends         = phase_ends,
        timeline_days      = t_arr,
        timeline_wound     = w_arr,
        timeline_pns       = pns_arr,
        timeline_sc        = sc_arr,
        timeline_bs        = bs_arr,
        timeline_limbic    = lim_arr,
        timeline_cortex    = ctx_arr,
        timeline_sal       = sal_arr,
        timeline_innate    = inn_arr,
        timeline_adapt     = ada_arr,
        timeline_bbb_tj    = tj_arr,
        timeline_bbb_nfkb  = nfkb_arr,
        compartments_final = dict(
            wound=round(V_wound,4), nmj=round(V_nmj,4), pns=round(V_pns,4),
            drg=round(V_drg,4), sc=round(V_sc,4), bs=round(V_bs,4),
            limbic=round(V_lim,4), cortex=round(V_ctx,4), sal=round(V_sal,4),
            innate=round(I_inn,4), adapt=round(I_ada,4),
        ),
        confidence_note = (
            "RABV v1.0 — 9-compartment Euler ODE. "
            "Hill replication, Michaelis-Menten immune clearance, "
            "P/N/M/G/L protein factors, axon progress transport model."
        ),
    )


# ── MAIN ENTRY ────────────────────────────────────────────────────────────────
def simulate(inp: RABVInput) -> RABVOutput:
    t0  = _time.perf_counter()
    raw = run_euler(inp)
    ms  = round((_time.perf_counter() - t0) * 1000, 2)
    raw["elapsed_ms"] = ms
    raw["engine_version"] = "1.0"

    return RABVOutput(**{k: raw[k] for k in RABVOutput.__dataclass_fields__ if k in raw})
