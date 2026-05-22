"""
bbb_core.py — NeuroViral Lab BBB Engine  Phase 2
QSS-Euler 7-compartment BBB transport model

Compartments:
  C_bl   — total blood (free + bound, quasi-steady-state for PPB)
  C_endo — endosomal pool (RMT internalized cargo)
  C_abl  — abluminal release pool (post-transcytosis, pre-ISF)
  C_isf  — brain interstitial fluid
  C_nic  — neuronal intracellular

Key scientific improvements over Phase 1 JS heuristics:
  1. Passive diffusion: Pardridge PPB correction + exponential MW decay
  2. RMT: Michaelis-Menten occupancy + affinity-efficiency paradox +
     endosomal sorting (f_trans/f_recycle/f_lysosome) + pH-sensitive release
  3. Efflux: Michaelis-Menten Vmax/Km per transporter class
     NF-κB inflammatory upregulation of P-gp expression
  4. Time-course output: ISF concentration over time, not single score
  5. Kp,uu,BBB: brain ISF AUC / blood AUC — standard pharmacokinetic metric
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ── PHYSICAL / PHARMACOKINETIC CONSTANTS ──────────────────────────────────────
BLOOD_PH      = 7.4
ENDOSOME_PH   = 5.8
BRAIN_ISF_PH  = 7.3

# Passive diffusion (Abraham/Clark model, Pardridge PPB correction)
PM_MW_DECAY           = 0.0025   # /Da — exponential MW penalty
PM_LOGP_OPTIMUM       = 1.7      # optimal logP for CNS passive diffusion
PM_LOGP_WIDTH         = 2.5      # Gaussian SD
PM_PARDRIDGE_BLEND_MAX = 2.0     # logP at which PPB constraint negligible
PM_FLOW_LIMIT_LOGP    = 3.5      # logP above which flow-limited correction applies
CBF_NORMAL_ML_100G    = 50.0     # mL/100g/min normal CBF
CAPILLARY_SA_REL      = 1.0      # relative capillary surface area

# RMT — TfR (human BBB healthy, literature)
TFR_BMAX_NM           = 50.0     # nM receptor sites luminal surface
TFR_KD_OPTIMAL_NM     = 30.0     # nM — optimal for transcytosis efficiency
TFR_AFFINITY_WIDTH    = 0.8      # controls paradox Gaussian width (log10 scale)
TFR_K_INT             = 0.12     # /h — internalization rate constant
TFR_K_SORT            = 2.0      # /h — endosomal sorting rate constant
TFR_F_TRANS           = 0.35     # fraction → productive transcytosis
TFR_F_RECYCLE         = 0.45     # fraction → receptor recycling
TFR_K_EXO_ABL        = 0.50     # /h — abluminal exocytosis rate
# TFR_F_LYSOSOME = 1 - TFR_F_TRANS - TFR_F_RECYCLE = 0.20

# Efflux Km (nM) and Vmax (relative) — human BBB luminal membrane
# Sources: Doran et al., Szakacs et al., human PET verapamil/quinidine studies
PGP_KM_NM   = 1000.0
PGP_VMAX    = 1.00
BCRP_KM_NM  = 800.0
BCRP_VMAX   = 0.75
MRP_KM_NM   = 500.0
MRP_VMAX    = 0.55

# RVG-29 nAChR shuttle kinetics (Vmax/Km calibrated to tier-2 evidence; Doxzen 2021)
RVG_VMAX    = 0.12   # relative Vmax (vs passive-diffusion baseline of 1.0)
RVG_KM_NM   = 45.0  # nM — saturation constant for nAChR-mediated uptake

# Species efflux scaling relative to human
SPECIES_PGP_SCALE = {"human": 1.0, "rat": 1.7, "mouse": 1.9, "nhp": 1.1}

# Neuronal uptake
K_NEURON_UP  = 0.08   # /h
K_NEURON_OUT = 0.04   # /h

# Systemic clearance default
CL_SYS_DEFAULT = 5.0  # /h (rough first-order)


# ── INPUT SCHEMA ───────────────────────────────────────────────────────────────
@dataclass
class BBBInput:
    # Molecule identity
    name: str = "Molecule"
    mw: float = 350.0          # Da
    logp: float = 2.0
    hbd: int = 2
    hba: int = 4
    ppb: float = 0.30          # fraction 0-1
    pka_acid: Optional[float] = None
    pka_base: Optional[float] = None
    mol_type: str = "sm"       # sm | glucose | aa | monocarb | peptide | protein | gas

    # Transporter flags
    pgp: bool = False
    bcrp: bool = False
    mrp: bool = False
    rmt: bool = False          # TfR-RMT conjugate
    rmt_kd_nm: float = 30.0   # Kd for TfR (nM) — Phase 2 exposes this explicitly
    rvg: bool = False
    cation: bool = False

    # BBB / NVU state (0-100 sliders from UI)
    tj: float = 100.0
    aj: float = 100.0
    mmp: float = 0.0
    nfkb: float = 0.0
    wnt: float = 100.0
    shh: float = 100.0
    pericyte: float = 100.0
    notch: float = 100.0
    angpt: float = 100.0
    ptm: float = 0.0
    cbf: float = 100.0         # % of normal CBF

    # Delivery context
    region: str = "cortex"
    species: str = "human"
    dose_nm: float = 100.0     # nmol/L blood concentration at t=0

    # Simulation
    t_hours: float = 8.0       # simulation duration
    dt_hours: float = 0.02     # Euler step size


# ── OUTPUT SCHEMA ──────────────────────────────────────────────────────────────
@dataclass
class RouteDetail:
    name: str
    score: float               # 0-100 heuristic score (legacy compatibility)
    flux_rel: float            # relative flux contribution (0-1)
    evidence_tier: int         # 1=validated, 2=experimental
    assumption_log: List[str] = field(default_factory=list)
    band_low: float = 0.0
    band_high: float = 0.0

@dataclass
class BBBOutput:
    # Legacy compatibility (returned for all callers)
    net_score: float           # 0-100 composite heuristic (backward compat)
    best_route: str
    net_band_low: float
    net_band_high: float
    efflux_penalty: float
    is_heuristic: bool = True

    # Phase 2 additions: kinetic outputs
    kpuu_pct: float = 0.0      # Kp,uu,BBB: ISF_AUC / blood_AUC × 100
    isf_peak_pct: float = 0.0  # ISF peak / initial blood dose × 100
    neuron_pct: float = 0.0    # neuronal intracellular at t_end / dose × 100
    t_peak_h: float = 0.0      # time to ISF peak (hours)

    # Time-course arrays (downsampled for API efficiency)
    timeline_t: List[float] = field(default_factory=list)
    timeline_isf: List[float] = field(default_factory=list)
    timeline_blood: List[float] = field(default_factory=list)
    timeline_endo: List[float] = field(default_factory=list)

    # Barrier state
    paracellular_breach_pct: float = 0.0
    transcytosis_breach_pct: float = 0.0
    endothelial_activation_pct: float = 0.0
    failure_mode: str = "intact"

    # Per-route details
    routes: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    confidence_note: str = ""
    phase: int = 2


# ── SIGNALLING MODEL ───────────────────────────────────────────────────────────
def compute_signaling(inp: BBBInput) -> dict:
    wnt      = inp.wnt      / 100
    shh      = inp.shh      / 100
    pericyte = inp.pericyte / 100
    notch    = inp.notch    / 100
    angpt    = inp.angpt    / 100
    mmp      = inp.mmp      / 100
    nfkb     = inp.nfkb     / 100
    ptm      = inp.ptm      / 100
    aj_raw   = inp.aj       / 100
    tj_raw   = inp.tj       / 100
    cbf      = inp.cbf      / 100

    wnt_eff = max(0, wnt - nfkb * 0.35)
    shh_eff = max(0, shh - nfkb * 0.40)

    # TJ integrity (paracellular barrier)
    e_tj = max(0, min(1,
        tj_raw
        * (1 - mmp  * 0.70)
        * (1 - ptm  * 0.40)
        * (1 - (1 - aj_raw) * 0.22)
        * (0.45 + wnt_eff * 0.55)
    ))

    # MFSD2A/caveolar suppression (transcytosis barrier — INDEPENDENT from TJ)
    e_mfsd2a = max(0, min(1,
        0.40 * wnt_eff + 0.30 * notch + 0.30 * pericyte
    ))

    e_aj = max(0, min(1,
        aj_raw * (0.30 + angpt * 0.70) * (1 - nfkb * 0.40) * (1 - mmp * 0.35)
    ))

    endoth_act    = min(1, nfkb * 0.7 + mmp * 0.3)
    leuko_adhesion= min(1, nfkb * 0.85)
    transmig_comp = min(1, nfkb * 0.6 + mmp * 0.4)
    ecm_damage    = min(1, mmp * 0.80 + ptm * 0.20)

    pgp_inflam_boost  = 1.0 + nfkb * 0.45
    bcrp_inflam_boost = 1.0 + nfkb * 0.30

    para_breach  = round((1 - e_tj)     * 100)
    tcy_breach   = round((1 - e_mfsd2a) * 100)
    immune_breach= round(transmig_comp  * 100)

    if (1-e_tj) > 0.30 and (1-e_mfsd2a) > 0.30:
        failure_mode = "mixed"
    elif (1-e_tj) > 0.30:
        failure_mode = "paracellular"
    elif (1-e_mfsd2a) > 0.30:
        failure_mode = "transcellular"
    elif transmig_comp > 0.50:
        failure_mode = "immune"
    else:
        failure_mode = "intact"

    return dict(
        e_tj=e_tj, e_aj=e_aj, e_mfsd2a=e_mfsd2a,
        wnt_eff=wnt_eff, shh_eff=shh_eff, pericyte=pericyte,
        mmp=mmp, nfkb=nfkb, ptm=ptm, cbf=cbf,
        endoth_act=endoth_act, leuko_adhesion=leuko_adhesion,
        transmig_comp=transmig_comp, ecm_damage=ecm_damage,
        pgp_inflam_boost=pgp_inflam_boost, bcrp_inflam_boost=bcrp_inflam_boost,
        para_breach=para_breach, tcy_breach=tcy_breach, immune_breach=immune_breach,
        failure_mode=failure_mode,
        immune_trafficking=min(1, nfkb + mmp * 0.30),
    )


# ── IONISATION ─────────────────────────────────────────────────────────────────
def effective_charge(inp: BBBInput, ph: float) -> float:
    if inp.pka_acid is None and inp.pka_base is None:
        return 0.0
    charge = 0.0
    if inp.pka_acid is not None:
        charge -= 1 / (1 + 10 ** (inp.pka_acid - ph))
    if inp.pka_base is not None:
        charge += 1 / (1 + 10 ** (ph - inp.pka_base))
    return charge


# ── REGIONAL FACTORS ───────────────────────────────────────────────────────────
REGIONAL_FACTORS = {
    "cortex"        : dict(glut1=1.00, lat1=1.00, pgp=1.00, bcrp=1.00, tfr=1.00, nachr=0.60, ca_sa=1.00),
    "hippocampus"   : dict(glut1=1.30, lat1=1.20, pgp=0.80, bcrp=0.85, tfr=1.20, nachr=1.00, ca_sa=1.10),
    "brainstem"     : dict(glut1=0.80, lat1=0.90, pgp=1.40, bcrp=1.30, tfr=0.90, nachr=1.40, ca_sa=0.95),
    "cerebellum"    : dict(glut1=0.90, lat1=0.80, pgp=1.20, bcrp=1.15, tfr=0.95, nachr=0.70, ca_sa=0.90),
    "thalamus"      : dict(glut1=1.10, lat1=1.00, pgp=0.90, bcrp=0.95, tfr=1.05, nachr=0.80, ca_sa=1.05),
    "choroid_plexus": dict(glut1=1.50, lat1=1.30, pgp=0.60, bcrp=0.70, tfr=1.40, nachr=0.50, ca_sa=1.80),
}
REGIONAL_CV = dict(cortex=0.30, hippocampus=0.35, brainstem=0.38,
                   cerebellum=0.32, thalamus=0.33, choroid_plexus=0.45)
TRANSPORTER_CV = dict(pgp=0.42, bcrp=0.35, mrp=0.48, tfr=0.31, glut1=0.27, lat1=0.33,
                      rvg=0.38)   # RVG-29 nAChR shuttle CV (Vmax=0.12, Km=45 nM; Cabantous-class tier-2)


# ── QSS-EULER 7-COMPARTMENT MODEL ─────────────────────────────────────────────
def run_qss_euler(inp: BBBInput, sig: dict) -> dict:
    """
    Quasi-steady-state Euler integration.
    Tracks: C_bl (total blood), C_endo, C_abl, C_isf, C_nic
    Protein binding handled by QSS: C_free = C_bl * (1 - ppb)
    """
    rf  = REGIONAL_FACTORS.get(inp.region, REGIONAL_FACTORS["cortex"])
    ppb = max(0.0, min(0.99, inp.ppb))
    dt  = inp.dt_hours
    T   = inp.t_hours
    C0  = inp.dose_nm

    # ── Passive permeability (Pardridge PPB correction) ──────────────────────
    mw_factor    = math.exp(-PM_MW_DECAY * max(0, inp.mw - 100))
    logp_factor  = math.exp(-0.5 * ((inp.logp - PM_LOGP_OPTIMUM) / PM_LOGP_WIDTH) ** 2)
    hb_penalty   = max(0, 1 - inp.hbd * 0.13 - inp.hba * 0.04)
    b_charge     = effective_charge(inp, BLOOD_PH)
    ion_factor   = max(0.02, 1 - abs(b_charge) * 0.55)
    Pm_intrinsic = logp_factor * mw_factor * hb_penalty * ion_factor
    # Pardridge blend: logP > PM_PARDRIDGE_BLEND_MAX → PPB irrelevant
    pard_blend   = max(0, min(1, inp.logp / PM_PARDRIDGE_BLEND_MAX))

    # ── RMT parameters ───────────────────────────────────────────────────────
    kd_tfr   = inp.rmt_kd_nm if inp.rmt else 1e9
    Bmax     = TFR_BMAX_NM  if inp.rmt else 0.0
    # Affinity-efficiency paradox: Gaussian around optimal Kd (Neuron 2014)
    affinity_eff = math.exp(-0.5 * ((math.log10(max(kd_tfr, 0.001))
                              - math.log10(TFR_KD_OPTIMAL_NM)) / TFR_AFFINITY_WIDTH) ** 2)
    # pH-sensitive release: tighter binders release less at endosomal pH 5.8
    pH_release   = max(0.3, 1 - math.exp(-kd_tfr / 50.0))
    f_trans_eff  = TFR_F_TRANS   * (0.40 + 0.60 * pH_release)
    f_rec_eff    = TFR_F_RECYCLE * (1.00 - 0.40 * pH_release)
    k_int_eff    = TFR_K_INT * affinity_eff * rf["tfr"]

    # ── Efflux parameters ────────────────────────────────────────────────────
    pgp_vmax  = PGP_VMAX  * rf["pgp"]  * sig["pgp_inflam_boost"] * SPECIES_PGP_SCALE.get(inp.species, 1.0)
    bcrp_vmax = BCRP_VMAX * rf["bcrp"] * sig["bcrp_inflam_boost"]
    mrp_vmax  = MRP_VMAX

    # ── Paracellular rate ─────────────────────────────────────────────────────
    e_tj    = sig["e_tj"]
    pore_sieve = 1.0 if inp.mw < 200 else (0.55 if inp.mw < 500 else (0.15 if inp.mw < 2000 else 0.02))
    c_sel   = math.exp(-0.8 * abs(b_charge))
    P_para  = (1 - e_tj) * 0.04 * c_sel * pore_sieve   # scaled rate constant

    # ── CMT rate ─────────────────────────────────────────────────────────────
    if inp.mol_type == "glucose":
        glut1_expr = 0.60 + sig["wnt_eff"] * 0.40
        k_cmt = 0.90 * glut1_expr * 0.70 * rf["glut1"]   # 0.70 = physiological saturation
    elif inp.mol_type == "aa":
        lat1_expr  = 0.55 + sig["wnt_eff"] * 0.45
        k_cmt = 0.82 * lat1_expr * 0.75 * rf["lat1"]
    elif inp.mol_type == "monocarb":
        k_cmt = 0.76 * (0.65 + sig["wnt_eff"] * 0.35)
    else:
        k_cmt = 0.0

    # ── Integration ──────────────────────────────────────────────────────────
    C_bl   = C0
    C_endo = 0.0
    C_abl  = 0.0
    C_isf  = 0.0
    C_nic  = 0.0

    t_arr   = []
    bl_arr  = []
    isf_arr = []
    endo_arr= []

    t = 0.0
    steps = int(T / dt) + 1

    for _ in range(steps):
        C_free = C_bl * (1.0 - ppb)   # QSS free fraction

        # Pardridge effective free (membrane partitioning for lipophilics)
        eff_free_pas = (1 - ppb) * (1 - pard_blend) + 0.90 * pard_blend
        C_eff_pas = C_bl * eff_free_pas

        # Passive diffusion flux
        J_passive = Pm_intrinsic * C_eff_pas

        # Paracellular flux
        J_para = P_para * C_free

        # CMT flux
        J_cmt = k_cmt * C_free

        # RMT: MM occupancy → internalization
        if inp.rmt and Bmax > 0:
            occ    = Bmax * C_free / (kd_tfr + C_free + 1e-12)
            J_rmt  = k_int_eff * occ
        else:
            J_rmt  = 0.0

        # Endosomal sorting
        J_trans  = TFR_K_SORT * f_trans_eff  * C_endo
        J_rec    = TFR_K_SORT * f_rec_eff    * C_endo
        J_lys    = TFR_K_SORT * max(0, 1 - f_trans_eff - f_rec_eff) * C_endo

        # Abluminal exocytosis
        J_abl_rel = TFR_K_EXO_ABL * C_abl

        # Efflux (MM from ISF)
        J_efflux = 0.0
        if inp.pgp:
            J_efflux += pgp_vmax  * C_isf / (PGP_KM_NM  + C_isf + 1e-12)
        if inp.bcrp:
            J_efflux += bcrp_vmax * C_isf / (BCRP_KM_NM + C_isf + 1e-12)
        if inp.mrp:
            J_efflux += mrp_vmax  * C_isf / (MRP_KM_NM  + C_isf + 1e-12)

        # Neuronal uptake
        J_neur_up  = K_NEURON_UP  * C_isf
        J_neur_out = K_NEURON_OUT * C_nic

        # Systemic clearance (on free fraction)
        J_cl = CL_SYS_DEFAULT * C_free

        # Euler step
        dC_bl  = -(J_passive + J_para + J_cmt + J_rmt) + J_rec + J_efflux - J_cl
        dC_endo=   J_rmt  - J_trans - J_rec - J_lys
        dC_abl =   J_trans - J_abl_rel
        dC_isf =  (J_passive + J_para + J_cmt + J_abl_rel) - J_efflux - J_neur_up
        dC_nic =   J_neur_up - J_neur_out

        C_bl   = max(0, C_bl   + dt * dC_bl)
        C_endo = max(0, C_endo + dt * dC_endo)
        C_abl  = max(0, C_abl  + dt * dC_abl)
        C_isf  = max(0, C_isf  + dt * dC_isf)
        C_nic  = max(0, C_nic  + dt * dC_nic)

        # Record every 5th step (~0.1h resolution)
        if _ % 5 == 0:
            t_arr.append(round(t, 3))
            bl_arr.append(round(C_bl, 6))
            isf_arr.append(round(C_isf, 6))
            endo_arr.append(round(C_endo, 6))

        t += dt

    # ── Kinetic metrics ───────────────────────────────────────────────────────
    ta  = np.array(t_arr)
    ia  = np.array(isf_arr)
    ba  = np.array(bl_arr)

    isf_peak  = float(ia.max()) / C0 * 100 if C0 > 0 else 0
    auc_bl    = float(np.trapezoid(ba, ta)) if len(ta) > 1 else 1e-9
    auc_isf   = float(np.trapezoid(ia, ta)) if len(ta) > 1 else 0
    kpuu      = auc_isf / max(auc_bl, 1e-9) * 100
    neur_pct  = C_nic / C0 * 100 if C0 > 0 else 0
    t_peak    = float(ta[int(ia.argmax())]) if len(ia) > 0 else 0

    return dict(
        isf_peak_pct=round(isf_peak, 4),
        kpuu_pct    =round(kpuu, 4),
        neuron_pct  =round(neur_pct, 6),
        t_peak_h    =t_peak,
        timeline_t  =t_arr,
        timeline_isf=isf_arr,
        timeline_blood=bl_arr,
        timeline_endo=endo_arr,
        C_isf_final =round(C_isf, 6),
        C_nic_final =round(C_nic, 6),
    )


# ── ROUTE SCORES (legacy compatibility + Phase 2 enriched) ────────────────────
def compute_routes_and_score(inp: BBBInput, sig: dict, kinetics: dict) -> dict:
    """Compute per-route scores (0-100 heuristic) for backward compat with JS callers."""
    rf  = REGIONAL_FACTORS.get(inp.region, REGIONAL_FACTORS["cortex"])
    ppb = inp.ppb
    b_charge = effective_charge(inp, BLOOD_PH)

    def uncertainty_band(score, tier, cv=0.35):
        mult = 1.0 if tier == 1 else 1.6
        cv2  = cv * mult
        return dict(low=max(0, round(score*(1-cv2))), expected=round(score), high=min(100, round(score*(1+cv2))))

    # Passive diffusion score
    mw_f  = math.exp(-PM_MW_DECAY * max(0, inp.mw - 100))
    lp_f  = math.exp(-0.5 * ((inp.logp - PM_LOGP_OPTIMUM) / PM_LOGP_WIDTH) ** 2)
    hb_f  = max(0, 1 - inp.hbd * 0.13 - inp.hba * 0.04)
    ion_f = max(0.02, 1 - abs(b_charge) * 0.55)
    pard  = max(0, min(1, inp.logp / PM_PARDRIDGE_BLEND_MAX))
    eff_f = (1-ppb)*(1-pard) + 0.90*pard
    pd_score = min(95, mw_f * lp_f * hb_f * ion_f * eff_f * 100)

    # Paracellular
    tj_open   = 1 - sig["e_tj"]
    c_sel     = math.exp(-0.8 * abs(b_charge))
    pore_sieve= 1.0 if inp.mw<200 else (0.55 if inp.mw<500 else (0.15 if inp.mw<2000 else 0.02))
    para_score= min(80, tj_open * 40 * c_sel * pore_sieve * rf.get("pgp", 1.0))  # reuse pgp scale for TJ

    # CMT
    if inp.mol_type == "glucose":
        cmt_score = min(95, 90*(0.60+sig["wnt_eff"]*0.40)*0.70*rf["glut1"])
    elif inp.mol_type == "aa":
        cmt_score = min(95, 82*(0.55+sig["wnt_eff"]*0.45)*0.75*rf["lat1"])
    elif inp.mol_type == "monocarb":
        cmt_score = min(95, 76*(0.65+sig["wnt_eff"]*0.35))
    else:
        cmt_score = 0.0

    # RMT
    if inp.rmt:
        kd_tfr    = inp.rmt_kd_nm
        occ       = TFR_BMAX_NM / (TFR_BMAX_NM + kd_tfr)
        aff_eff   = math.exp(-0.5*((math.log10(max(kd_tfr,0.001))-math.log10(TFR_KD_OPTIMAL_NM))/TFR_AFFINITY_WIDTH)**2)
        pH_rel    = max(0.3, 1 - math.exp(-kd_tfr/50.0))
        f_t       = TFR_F_TRANS*(0.40+0.60*pH_rel)
        mfsd_supp = 0.50 + sig["e_mfsd2a"]*0.50
        mw_pen    = 1.0 if inp.mw<=5000 else (0.85 if inp.mw<=10000 else (0.65 if inp.mw<=30000 else (0.45 if inp.mw<=80000 else 0.20)))
        rmt_score = min(72, occ*aff_eff*f_t*mfsd_supp*mw_pen*rf["tfr"]*72)
        rmt_log   = [
            f"MM occupancy={occ:.2f} (Bmax={TFR_BMAX_NM}nM, Kd={kd_tfr}nM)",
            f"AffinityEff={aff_eff:.2f} (paradox: optimal Kd~{TFR_KD_OPTIMAL_NM}nM)",
            f"f_trans={f_t:.2f} (pH-sensitive release)",
        ]
    elif inp.mol_type == "protein" and inp.mw > 5000:
        rmt_score = 3.0
        rmt_log   = ["Non-targeted biologic: ~0.05% via bulk-phase transcytosis"]
    else:
        rmt_score = 0.0
        rmt_log   = []

    # RVG-29
    if inp.rvg:
        mw_pen_rvg = 1.0 if inp.mw<=3000 else (0.80 if inp.mw<=8000 else (0.55 if inp.mw<=15000 else (0.35 if inp.mw<=20000 else 0.15)))
        sp_factor  = 0.65 if inp.species=="human" else (0.80 if inp.species=="nhp" else 1.0)
        mfsd_s     = 0.55 + sig["e_mfsd2a"]*0.45
        rvg_score  = min(40, 32*mw_pen_rvg*rf["nachr"]*mfsd_s*sp_factor*(1-ppb))
        rvg_log    = [
            "EVIDENCE TIER 2 — mechanism debated (nAChR vs GABA_B vs caveolae)",
            f"Species factor={sp_factor} (most data from rodents)",
        ]
    else:
        rvg_score = 0.0
        rvg_log   = []

    # Caveolar
    mfsd_loss = 1 - sig["e_mfsd2a"]
    ion_pen   = max(0.06, 1 - abs(b_charge)*0.60)
    if inp.mw > 50000:   cav_score = min(50, mfsd_loss*0.40*45*ion_pen)
    elif inp.mw > 5000:  cav_score = min(50, mfsd_loss*0.70*45*ion_pen)
    elif inp.mw > 1000:  cav_score = min(50, mfsd_loss*0.90*40*ion_pen)
    elif inp.mw > 400:   cav_score = min(50, mfsd_loss*0.22*28*ion_pen)
    else:                cav_score = 0.0

    # AMT
    amt_elig = inp.cation or b_charge >= 0.5
    if amt_elig:
        tcy_l = sig.get("e_mfsd2a", 0.5)
        tcy_f = 0.35 if tcy_l > 0.65 else (0.65 if tcy_l > 0.35 else 1.0)
        amt_score = min(38, (abs(b_charge)+0.5)*8*tcy_f)
    else:
        amt_score = 0.0

    # Efflux
    assumed_conc = 500.0  # nM — Phase 2 uses actual ISF concentration
    efflux_score = 0.0
    eff_log = []
    if inp.pgp:
        pgp_v = PGP_VMAX * rf["pgp"] * sig["pgp_inflam_boost"] * SPECIES_PGP_SCALE.get(inp.species, 1.0)
        efflux_score += pgp_v * assumed_conc/(PGP_KM_NM+assumed_conc) * 36
        eff_log.append(f"P-gp MM: Vmax={pgp_v:.2f} inflam_boost={sig['pgp_inflam_boost']:.2f}")
    if inp.bcrp:
        bcrp_v = BCRP_VMAX * sig["bcrp_inflam_boost"]
        efflux_score += bcrp_v * assumed_conc/(BCRP_KM_NM+assumed_conc) * 26
        eff_log.append(f"BCRP MM: Vmax={bcrp_v:.2f}")
    if inp.mrp:
        efflux_score += MRP_VMAX * assumed_conc/(MRP_KM_NM+assumed_conc) * 18
        eff_log.append("MRP MM")
    efflux_score = min(75, efflux_score)

    # Immune
    imm_score = min(100, sig["immune_trafficking"]*100)

    # Best influx route
    influx_routes = {
        "Passive diffusion":pd_score, "Paracellular (TJ+AJ)":para_score,
        "CMT / SLC":cmt_score, "RMT (TfR/IR/LRP)":rmt_score,
        "RVG-29 (nAChR)":rvg_score, "Caveolar (MFSD2A)":cav_score,
        "AMT (adsorptive)":amt_score,
    }
    best_route = max(influx_routes, key=influx_routes.get)
    best_score = influx_routes[best_route]
    net_score  = max(0, min(100, best_score - efflux_score * 0.6))

    # Uncertainty band on net
    cv = TRANSPORTER_CV.get("pgp", 0.35)
    r_cv = REGIONAL_CV.get(inp.region, 0.35)
    band_cv = (cv + r_cv) / 2
    band = uncertainty_band(net_score, 1, band_cv)

    return dict(
        net_score  =round(net_score, 1),
        best_route =best_route,
        best_score =round(best_score, 1),
        efflux_score=round(efflux_score, 1),
        net_band   =band,
        routes     =dict(
            passive =dict(score=round(pd_score,1),  tier=1, log=[
                f"Pm_intrinsic={mw_f*lp_f*hb_f*ion_f*100:.1f}·eff_free={eff_f*100:.0f}% (Pardridge blend={pard*100:.0f}%)",
                f"MW_factor={mw_f:.3f}·logP_factor={lp_f:.3f}"]),
            paracellular=dict(score=round(para_score,1), tier=1, log=[
                f"TJ_breach={sig['para_breach']}% (separate from transcytosis)",
                f"charge_selectivity={c_sel:.3f} (continuous sigmoid)"]),
            cmt    =dict(score=round(cmt_score,1),  tier=1, log=[f"type={inp.mol_type}"]),
            rmt    =dict(score=round(rmt_score,1),  tier=1, log=rmt_log),
            rvg    =dict(score=round(rvg_score,1),  tier=2, log=rvg_log),
            caveolar=dict(score=round(cav_score,1), tier=2, log=[f"MFSD2A={sig['e_mfsd2a']*100:.0f}%"]),
            amt    =dict(score=round(amt_score,1),  tier=2, log=[f"charge={b_charge:.2f}"]),
            immune =dict(score=round(imm_score,1),  tier=1, log=[f"NF-κB={sig['nfkb']*100:.0f}%"]),
            efflux =dict(score=round(efflux_score,1), tier=1, log=eff_log),
        ),
    )


# ── MAIN ANALYSIS ENTRY POINT ──────────────────────────────────────────────────
def analyze(inp: BBBInput) -> BBBOutput:
    """Full BBB analysis: signalling + kinetics + route scores."""
    sig      = compute_signaling(inp)
    kinetics = run_qss_euler(inp, sig)
    scores   = compute_routes_and_score(inp, sig, kinetics)

    return BBBOutput(
        # Legacy compatibility
        net_score     = scores["net_score"],
        best_route    = scores["best_route"],
        net_band_low  = scores["net_band"]["low"],
        net_band_high = scores["net_band"]["high"],
        efflux_penalty= scores["efflux_score"],
        is_heuristic  = False,   # Phase 2: ODE-computed

        # Kinetic outputs
        kpuu_pct      = kinetics["kpuu_pct"],
        isf_peak_pct  = kinetics["isf_peak_pct"],
        neuron_pct    = kinetics["neuron_pct"],
        t_peak_h      = kinetics["t_peak_h"],
        timeline_t    = kinetics["timeline_t"],
        timeline_isf  = kinetics["timeline_isf"],
        timeline_blood= kinetics["timeline_blood"],
        timeline_endo = kinetics["timeline_endo"],

        # Barrier state
        paracellular_breach_pct    = sig["para_breach"],
        transcytosis_breach_pct    = sig["tcy_breach"],
        endothelial_activation_pct = round(sig["endoth_act"]*100),
        failure_mode               = sig["failure_mode"],

        # Per-route
        routes        = scores["routes"],
        confidence_note = "Phase 2 ODE engine — Kpuu and time-course computed via QSS-Euler integration.",
        phase         = 2,
    )
