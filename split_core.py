"""
split_core.py  —  NeuroViral Lab  Split-Protein Designer Engine  v2.0
Pure-Python computation backend for Module 03.
Port 8002 via split_main.py.

Scientific models:
  1. BBB fragment scoring (MW, RMT, RVG, regional factors)
  2. Co-localization (same-cell × same-compartment × arrival-sync × half-life)
  3. Reassembly (Michaelis-Menten Kd, Npu DnaE kinetics)
  4. Therapeutic window (4-phase RABV infection model)
  5. Host state modifiers (immunocompromised, inflammation, brainstem, vaccination)
  6. Architecture mechanism (A: IFN-β, B: N-oligo, C: DN-L, D: Tripartite)
  7. Manufacturing feasibility (6-factor GMP scorecard)
  8. Safety (off-target, immunogenicity, TfR competition, neuronal load)
  9. Uncertainty propagation (analytical CV-based bounds)
  10. Monte Carlo rank stability (n=500 biological noise sampling)
  11. Explainability (structured drivers / limiters / upgrade steps)
"""

import math
import time as _time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ── SPLIT SYSTEMS ─────────────────────────────────────────────────────────────
SPLIT_SYSTEMS: Dict[str, Dict] = {
    "split_intein_npu": dict(
        name="Npu DnaE intein", kd_nM=0.001, t_half_s=1.0,
        reversible=False, activity_recovery=0.85, covalent=True,
        note="BEST — irreversible covalent ligation, t½~1 s.",
    ),
    "split_gfp": dict(
        name="Split-GFP (GFP1-10/11)", kd_nM=0.5, t_half_s=30.0,
        reversible=False, activity_recovery=0.90, covalent=False,
        note="Good scaffold — Kd <1 nM, effectively irreversible.",
    ),
    "nanobit": dict(
        name="NanoBiT (LgBiT/SmBiT)", kd_nM=190_000.0, t_half_s=300.0,
        reversible=True, activity_recovery=1.0, covalent=False,
        note="POOR — Kd 190 μM >> achievable CNS concentration.",
    ),
    "fkbp_frb": dict(
        name="FKBP/FRB + Rapamycin", kd_nM=0.2, t_half_s=600.0,
        reversible=False, activity_recovery=0.75, covalent=False,
        note="Conditional — requires rapamycin co-dose.",
        requires_trigger="rapamycin",
    ),
    "leucine_zipper": dict(
        name="Leucine Zipper", kd_nM=100.0, t_half_s=120.0,
        reversible=True, activity_recovery=0.70, covalent=False,
        note="Moderate affinity — may fail at low CNS concentrations.",
    ),
}

# ── RABV TARGETS ──────────────────────────────────────────────────────────────
RABV_TARGETS: Dict[str, Dict] = {
    "P_LC8"  : dict(name="P — LC8 (dynein)",    residues="aa 218-225", kd_nM=1,    druggability="High",   effect="Transport 200→12 mm/day"),
    "P_TBK1" : dict(name="P — TBK1 (IFN)",      residues="Ser179",     kd_nM=5,    druggability="High",   effect="IFN-β restored → antiviral state"),
    "N_oligo": dict(name="N — oligomerisation",  residues="N-N contact",kd_nM=None, druggability="Medium", effect="RNP collapse → RNA degraded"),
    "P+N dual": dict(name="P + N dual",          residues="P-LC8+N-N",  kd_nM=5,    druggability="Medium", effect="Transport stall + RNP collapse"),
}

# ── ARCHITECTURES ─────────────────────────────────────────────────────────────
ARCHITECTURES: Dict[str, Dict] = {
    "A": dict(
        name="Split-Nanobody vs P protein", short="Anti-P VHH",
        target="P_TBK1", split_system="split_intein_npu",
        best_window="pre_cns_intercept", mechanism_speed="fast",
        occupancy_required="low", tripartite=False,
        f1_mw=8200,  f1_rmt=True,  f1_rvg=False,
        f2_mw=11400, f2_rmt=False, f2_rvg=True,
        strengths=["Both P-protein interfaces biochemically validated",
                   "Pep2 precedent confirms P-LC8 druggable",
                   "IFN-β restoration harnesses natural immune kill",
                   "Smallest fragment MW — best BBB penetration"],
        weaknesses=["Depends on intact host immune response",
                    "No approved anti-P nanobody yet",
                    "IFN-β restoration may cause transient neuroinflammation"],
        desc="Anti-P nanobody split on Npu DnaE intein. F1 (TfR-RMT) blocks TBK1 phosphorylation; F2 (RVG-29) blocks LC8 binding → transport stall.",
    ),
    "B": dict(
        name="Split-DARPin vs N protein", short="Anti-N DARPin",
        target="N_oligo", split_system="split_intein_npu",
        best_window="early_cns_control", mechanism_speed="medium",
        occupancy_required="high", tripartite=False,
        f1_mw=10500, f1_rmt=True,  f1_rvg=False,
        f2_mw=10200, f2_rmt=False, f2_rvg=True,
        strengths=["Direct replication block — immune-agnostic",
                   "N structurally essential with no redundancy",
                   "Crystal structure available (RCSB 8FFR)"],
        weaknesses=["No validated N inhibitor in vivo yet",
                    "Larger fragments — worse BBB penetration",
                    "Requires high intraneuronal occupancy (≥65%)"],
        desc="Anti-N DARPin split on Npu DnaE intein. F1 (TfR-RMT) + F2 (RVG) reassemble to block N-oligomerisation → RNP collapse.",
    ),
    "C": dict(
        name="Split-L (Rapamycin-gated)", short="Anti-L Conditional",
        target="P_LC8", split_system="fkbp_frb",
        best_window="early_cns_control", mechanism_speed="slow",
        occupancy_required="high", tripartite=False,
        f1_mw=15000, f1_rmt=True,  f1_rvg=False,
        f2_mw=14000, f2_rmt=False, f2_rvg=False,
        strengths=["Conditional rapamycin gate = safety off-switch",
                   "Rapamycin crosses BBB well (logP 4.3)",
                   "Polymerase inhibition is direct antiviral"],
        weaknesses=["L protein not fully structurally solved",
                    "Largest fragments — worst BBB penetration",
                    "Requires rapamycin co-dose (PK complexity)"],
        desc="Dominant-negative L split on FKBP/FRB gated by rapamycin. Rapamycin acts as molecular glue to activate the split L inhibitor.",
    ),
    "D": dict(
        name="Tripartite (P+N+RVG)", short="Tripartite Dual-Target",
        target="P+N dual", split_system="split_gfp",
        best_window="early_cns_control", mechanism_speed="medium",
        occupancy_required="medium", tripartite=True,
        f1_mw=7000,  f1_rmt=True,  f1_rvg=False,
        f2_mw=8000,  f2_rmt=False, f2_rvg=True,
        f3_mw=5500,  f3_rvg=True,
        strengths=["Synergistic dual target — escape requires two simultaneous mutations",
                   "RVG-29 confers infected-neuron specificity",
                   "Mechanism-agnostic to immune state"],
        weaknesses=["Tripartite reconstitution ~40% efficiency vs bipartite",
                    "Three independent BBB crossings required",
                    "Highest manufacturing complexity"],
        desc="Tripartite split-GFP targeting both P and N proteins simultaneously. Three-fragment co-delivery via TfR-RMT + RVG-29.",
    ),
}

# ── REGIONAL FACTORS ──────────────────────────────────────────────────────────
REGIONAL_FACTORS: Dict[str, Dict] = {
    "cortex"     : dict(pgp=1.00, bcrp=1.00, tfr=1.00, nachr=0.60),
    "hippocampus": dict(pgp=0.80, bcrp=0.85, tfr=1.20, nachr=1.00),
    "brainstem"  : dict(pgp=1.40, bcrp=1.30, tfr=0.90, nachr=1.40),
    "cerebellum" : dict(pgp=1.20, bcrp=1.15, tfr=0.95, nachr=0.70),
    "thalamus"   : dict(pgp=0.90, bcrp=0.95, tfr=1.05, nachr=0.80),
}

# ── CLINICAL SCENARIOS ────────────────────────────────────────────────────────
CLINICAL_SCENARIOS: List[Dict] = [
    dict(label="Face bite · Day 0",         bite_mm=200,  treat_day=0,  velocity=200, viral_load="high"),
    dict(label="Hand bite · Day 2",         bite_mm=900,  treat_day=2,  velocity=200),
    dict(label="Foot bite · Day 5",         bite_mm=1200, treat_day=5,  velocity=200),
    dict(label="Foot bite · Day 10",        bite_mm=1200, treat_day=10, velocity=200),
    dict(label="Foot bite · Day 14",        bite_mm=1200, treat_day=14, velocity=200),
    dict(label="Vaccinated · Hand · Day 3", bite_mm=900,  treat_day=3,  velocity=200, prior_vaccination=True),
    dict(label="Immunocompromised · Day 3", bite_mm=600,  treat_day=3,  velocity=200, immunocompromised=True),
    dict(label="Brainstem · Face · Day 4",  bite_mm=200,  treat_day=4,  velocity=200, brainstem_dominant=True),
    dict(label="High inflam · Day 5",       bite_mm=900,  treat_day=5,  velocity=200, high_inflammation=True),
    dict(label="Elderly / slow velocity",   bite_mm=900,  treat_day=6,  velocity=80),
    dict(label="Pediatric / fast velocity", bite_mm=600,  treat_day=3,  velocity=350, viral_load="high"),
]


# ── INPUT DATACLASS ───────────────────────────────────────────────────────────
@dataclass
class DesignerInput:
    arch_id:           str   = "A"
    region:            str   = "hippocampus"
    bite_mm:           float = 1200.0
    velocity:          float = 200.0
    treat_day:         float = 0.0
    conc_nm:           float = 5.0
    viral_burden:      float = 0.30
    split_system:      str   = "split_intein_npu"
    f1_mw:             float = 0.0
    f1_rmt:            bool  = True
    f1_rvg:            bool  = False
    f2_mw:             float = 0.0
    f2_rmt:            bool  = False
    f2_rvg:            bool  = True
    immunocompromised: bool  = False
    high_inflammation: bool  = False
    brainstem_dominant:bool  = False
    prior_vaccination: bool  = False
    viral_load:        str   = "med"
    isf_peak_pct:      Optional[float] = 0.0
    kpuu_pct:          Optional[float] = 0.0
    n_samples:         int   = 500


# ── BBB FRAGMENT SCORE ────────────────────────────────────────────────────────
def _bbb_score(mw: float, rmt: bool, rvg: bool, region: str) -> tuple:
    """Returns (score 0–100, route_label)."""
    rf   = REGIONAL_FACTORS.get(region, REGIONAL_FACTORS["hippocampus"])
    mw_f = math.exp(-0.0025 * max(0, mw - 100))
    logp = -2.5 - mw / 10_000
    lp_f = math.exp(-0.5 * ((logp - 1.7) / 2.5) ** 2)
    hb_p = max(0, 1 - (mw / 2200) * 0.13 - (mw / 1600) * 0.04)
    pm   = lp_f * mw_f * hb_p

    if rmt:
        occ   = 50.0 / (50.0 + 30.0)
        f_tr  = 0.35 * (0.40 + 0.60 * 0.85)
        mw_p  = 1.0 if mw <= 5000 else (0.85 if mw <= 10000 else (0.65 if mw <= 30000 else 0.45))
        score = min(72.0, occ * f_tr * 0.90 * mw_p * rf["tfr"] * 72)
        return round(score, 1), "RMT (TfR)"

    if rvg:
        mw_p  = 1.0 if mw <= 3000 else (0.80 if mw <= 8000 else (0.55 if mw <= 15000 else 0.35))
        score = min(40.0, 32 * mw_p * rf["nachr"] * 0.90 * 0.65)
        return round(score, 1), "RVG-29 (nAChR)"

    if mw < 500:
        return round(min(95.0, pm * 0.80 * 100), 1), "Passive diffusion"

    return round(max(0.0, min(25.0, pm * 0.30 * 100)), 1), "Passive diffusion"


# ── REASSEMBLY ────────────────────────────────────────────────────────────────
def compute_reassembly(split_system: str, conc_nm: float, time_h: float = 20.0) -> dict:
    sys = SPLIT_SYSTEMS.get(split_system, SPLIT_SYSTEMS["split_intein_npu"])
    kd  = sys["kd_nM"]
    fb  = conc_nm / (conc_nm + kd + 1e-12)
    th  = sys["t_half_s"] / 3600.0
    lam = math.log(2) / max(th, 1e-6)
    tf  = (min(1.0, time_h / (th * 5)) if sys["reversible"]
           else 1 - math.exp(-lam * time_h))
    prob     = fb * sys["activity_recovery"] * min(1.0, tf)
    prob_pct = round(prob * 100)
    t_min    = th * 3 * 60

    warnings = []
    if conc_nm < kd * 0.1:
        warnings.append(f"Conc {conc_nm:.2f} nM << Kd {kd} nM — reassembly negligible")
    if split_system == "nanobit":
        warnings.append("NanoBiT Kd=190 μM unsuitable for CNS concentrations")
    if sys.get("requires_trigger"):
        warnings.append(f"Requires co-administration of {sys['requires_trigger']}")

    return dict(
        system=sys["name"], kd_nM=kd, conc_nm=conc_nm,
        probability=round(prob, 4), probabilityPct=prob_pct,
        tToReassembly_min=round(t_min, 1),
        covalent=sys.get("covalent", False),
        reversible=sys["reversible"],
        warnings=warnings, note=sys["note"],
    )


# ── CO-LOCALIZATION ───────────────────────────────────────────────────────────
def compute_colocalization(f1_bbb: float, f2_bbb: float,
                           f1_rmt: bool, f2_rmt: bool,
                           f1_mw: float, f2_mw: float,
                           viral_burden: float, region: str,
                           host_state: dict) -> dict:
    rf           = REGIONAL_FACTORS.get(region, REGIONAL_FACTORS["hippocampus"])
    base_codeliv = min(f1_bbb, f2_bbb) / 100.0
    region_ov    = min(1.0, viral_burden * 1.5 * rf["nachr"])
    same_route   = (f1_rmt and f2_rmt) or (not f1_rmt and not f2_rmt)
    cell_entry   = 0.65 if same_route else 0.38
    base_coloc   = base_codeliv * region_ov * cell_entry

    t1 = 6.0 if f1_mw > 10000 else 14.0
    t2 = 6.0 if f2_mw > 10000 else 14.0
    hl = 0.60 + 0.40 * min(t1, t2) / max(t1, t2)

    arr1  = 4.0 if f1_rmt else 8.0
    arr2  = 4.0 if f2_rmt else 8.0
    delta = abs(arr1 - arr2)
    avg_t = (t1 + t2) / 2
    async_p = 1.0 if delta <= 1 else (0.80 if delta <= 4 else (0.65 if delta / avg_t < 0.5 else 0.40))

    deg_p    = 0.75 if abs((0.08 if f1_mw > 10000 else 0.02) - (0.08 if f2_mw > 10000 else 0.02)) > 0.03 else 1.0
    immuno_p = 0.80 if host_state.get("immunocompromised") else 1.0

    coloc     = max(0.0, min(1.0, base_coloc * hl * async_p * deg_p * immuno_p))
    coloc_pct = round(coloc * 100)

    steps = [
        dict(label="Min fragment BBB",   value=round(min(f1_bbb, f2_bbb))),
        dict(label="Region/viral overlap",value=round(region_ov * 100)),
        dict(label="Cell entry overlap",  value=round(cell_entry * 100)),
        dict(label="Half-life match",     value=round(hl * 100)),
        dict(label="Arrival synchrony",   value=round(async_p * 100)),
    ]
    steps.sort(key=lambda x: x["value"])

    return dict(
        colocScore=round(coloc, 4), colocPct=coloc_pct,
        breakdown=dict(
            baseCodeliv=round(base_codeliv * 100),
            regionOverlap=round(region_ov * 100),
            cellEntryOverlap=round(cell_entry * 100),
            baseColoc=round(base_coloc * 100),
            halfLifeMismatch=round(hl * 100),
            asyncPenalty=round(async_p * 100),
            degradMismatch=round(deg_p * 100),
            immunoPenalty=round(immuno_p * 100),
        ),
        bottleneck=steps,
        bottleneckLabel=steps[0]["label"],
    )


# ── THERAPEUTIC WINDOW ────────────────────────────────────────────────────────
def compute_window(inp: "DesignerInput", arch: dict,
                   f1_arr_h: float, f2_arr_h: float, reassembly_h: float) -> dict:
    cns_day     = (2.0 + inp.bite_mm / inp.velocity) / 24.0 * 24 / 24
    cns_day     = (2.0 + inp.bite_mm / inp.velocity) / 24.0
    symptom_day = cns_day + 5.0
    fatal_day   = symptom_day + 7.0

    days_post   = max(0.0, inp.treat_day - cns_day)
    burden      = min(0.95, 0.02 * (2 ** (days_post * 24 / 20)))

    late_h      = max(f1_arr_h, f2_arr_h)
    onset_h     = 1.0 if arch["best_window"] == "pre_cns_intercept" else (2.0 if arch["best_window"] == "early_cns_control" else 4.0)
    total_h     = late_h + reassembly_h + onset_h

    if inp.treat_day + total_h / 24 < cns_day:
        phase, phaseLabel = "pre_cns_intercept", "PNS intercept (pre-CNS)"
        wscore = min(100.0, 95 - inp.treat_day * 4)
        can_prevent = can_control = can_salvage = True
    elif inp.treat_day < symptom_day and burden < 0.20:
        phase, phaseLabel = "early_cns_control", "Early CNS control"
        bonus  = 10 if arch["best_window"] == "early_cns_control" else (-5 if arch["best_window"] == "pre_cns_intercept" else 0)
        wscore = min(75.0, 75 - (inp.treat_day - cns_day) * 8 - burden * 40 + bonus)
        can_prevent = False; can_control = True; can_salvage = True
    elif inp.treat_day < fatal_day and burden < 0.70:
        phase, phaseLabel = "late_cns_salvage", "Late CNS salvage"
        bonus  = 10 if arch["best_window"] == "late_cns_salvage" else -5
        wscore = max(5.0, 40 - (inp.treat_day - symptom_day) * 6 - burden * 30 + bonus)
        can_prevent = False; can_control = False; can_salvage = True
    else:
        phase, phaseLabel = "too_late", "Beyond salvage window"
        wscore = max(0.0, 10 - (inp.treat_day - fatal_day) * 5)
        can_prevent = can_control = can_salvage = False

    wscore = max(0.0, min(100.0, round(wscore)))
    color  = "#3ecf8e" if wscore >= 70 else ("#f59e0b" if wscore >= 40 else ("#f87171" if wscore >= 10 else "#7f1d1d"))
    verdict= "Excellent" if wscore >= 70 else ("Marginal" if wscore >= 40 else ("Poor" if wscore >= 10 else "No window"))

    return dict(
        cnsDay=round(cns_day, 2), symptomDay=round(symptom_day, 2),
        fatalDay=round(fatal_day, 2), treatDay=inp.treat_day,
        burden=round(burden, 3), burdenPct=round(burden * 100),
        f1ArrivalH=f1_arr_h, f2ArrivalH=f2_arr_h,
        lateArrivalH=late_h, reassemblyH=round(reassembly_h, 3),
        inhibOnsetH=onset_h, totalH=round(total_h, 2),
        phase=phase, phaseLabel=phaseLabel,
        windowScore=wscore, color=color, verdict=verdict,
        canPrevent=can_prevent, canControl=can_control, canSalvage=can_salvage,
    )


# ── HOST MODIFIERS ────────────────────────────────────────────────────────────
def _host_mods(arch_id: str, host_state: dict) -> dict:
    m = dict(bbbMod=1.0, reassemblyMod=1.0, mechanismMod=1.0, windowMod=1.0, notes=[])
    if host_state.get("immunocompromised"):
        if arch_id == "A": m["mechanismMod"] *= 0.45; m["notes"].append("Immunocompromised: IFN-β strategy severely weakened")
        if arch_id == "D": m["mechanismMod"] *= 0.60; m["notes"].append("Immunocompromised: P-component of tripartite loses efficacy")
        m["reassemblyMod"] *= 0.85; m["notes"].append("Fragment t½ reduced in immunocompromised state")
    if host_state.get("high_inflammation"):
        m["bbbMod"] *= 1.25; m["mechanismMod"] *= 0.90
        m["notes"].append("High inflammation: BBB delivery +25%, efficacy window narrows")
    if host_state.get("brainstem_dominant"):
        m["windowMod"] *= 0.65; m["mechanismMod"] *= 0.80
        m["notes"].append("Brainstem-dominant: vital centres narrow margin")
        if arch_id in ("B", "D"): m["mechanismMod"] *= 1.15; m["notes"].append("Anti-N retains value in brainstem (immune-independent)")
    if host_state.get("prior_vaccination"):
        if arch_id == "A": m["mechanismMod"] *= 1.30; m["notes"].append("Prior vaccination + P neutralisation: IFN + memory T-cell synergy")
        m["windowMod"] *= 1.15; m["notes"].append("Prior vaccination extends salvage window ~15%")
    vl = host_state.get("viral_load", "med")
    if vl == "high": m["mechanismMod"] *= 0.75; m["reassemblyMod"] *= 0.90; m["notes"].append("High viral load: fragment:virus ratio unfavourable")
    elif vl == "low": m["mechanismMod"] *= 1.15; m["notes"].append("Low viral load: favourable fragment:virus ratio")
    return m


# ── MECHANISM DETAIL ──────────────────────────────────────────────────────────
def _mechanism(arch_id: str, inp: "DesignerInput", lim_conc: float) -> dict:
    vb = inp.viral_burden
    if arch_id == "A":
        ic      = 0.25 if inp.immunocompromised else (1.30 if inp.prior_vaccination else 1.0)
        ifn_r   = min(1.0, lim_conc / 5) * ic
        transp  = min(1.0, lim_conc / 3) * 0.65
        clear_b = ifn_r * 0.70
        eff     = 1 - (1 - ifn_r) * (1 - transp) * (1 - clear_b)
        return dict(mechanism="IFN-β restoration + retrograde transport stall",
                    ifn_restore_pct=round(ifn_r*100), transport_slow_pct=round(transp*100),
                    clearance_boost_pct=round(clear_b*100), immune_factor=round(ic,2),
                    limiting_step="IFN restoration" if ifn_r<transp else "Transport disruption",
                    efficacy_pct=round(eff*100))
    if arch_id == "B":
        occ_n   = 0.65
        ach     = min(1.0, lim_conc / 8)
        ok      = ach >= occ_n
        rnp     = (min(1.0,(ach-occ_n*0.8)/0.3) if ok else ach*0.3)
        rb      = rnp * 0.85
        return dict(mechanism="N-oligomer disruption → RNP collapse → RNA degradation",
                    occupancy_needed=f"{round(occ_n*100)}%", achieved_occ_pct=round(ach*100),
                    occupancy_met=ok, rnp_collapse_pct=round(rnp*100), replication_block=round(rb*100),
                    limiting_step="Occupancy threshold not reached (need ≥65%)" if not ok else "RNP disassembly kinetics",
                    efficacy_pct=round(rb*100),
                    note="⚠ Subthreshold" if not ok else "✓ Above threshold")
    if arch_id == "C":
        dnr  = lim_conc / max(0.5, vb*10)
        comp = min(1.0, dnr/(dnr+1))
        rdrp = comp * 0.90
        ri   = rdrp * (1-vb*0.30)
        return dict(mechanism="Dominant-negative L competes with native RdRp",
                    dn_ratio=round(dnr,2), competition_pct=round(comp*100),
                    rdrp_block_pct=round(rdrp*100), replication_inh=round(ri*100),
                    limiting_step="DN:viral L ratio too low" if dnr<0.5 else "RdRp active-site competition",
                    efficacy_pct=round(ri*100), note="L-RdRp structure partially solved")
    if arch_id == "D":
        ep   = min(1.0, lim_conc/5)  * 0.60
        en   = min(1.0, lim_conc/8)  * 0.55
        syn  = 1 - (1-ep)*(1-en)
        fin  = syn * (1-0.40) * (1-0.25)
        return dict(mechanism="Dual P+N synergy via tripartite reassembly",
                    e_p_pct=round(ep*100), e_n_pct=round(en*100), synergy_pct=round(syn*100),
                    tripartite_penalty="−40%", delivery_complexity="−25%",
                    synergy_formula="E = 1 − (1−E_P)(1−E_N)",
                    limiting_step="Anti-P arm" if ep<en else "Anti-N arm",
                    efficacy_pct=round(fin*100))
    return {}


# ── MANUFACTURING ─────────────────────────────────────────────────────────────
def compute_manufacturing(arch_id: str, f1_mw: float, f2_mw: float,
                           f1_rmt: bool, f2_rvg: bool, tripartite: bool,
                           split_system: str) -> dict:
    n_frags = 3 if tripartite else 2
    frag_s  = 45 if tripartite else 80
    avg_mw  = (f1_mw + f2_mw) / 2
    mw_s    = 90 if avg_mw<=5000 else (75 if avg_mw<=10000 else (55 if avg_mw<=20000 else (30 if avg_mw<=50000 else 15)))
    conj_p  = (18 if f1_rmt else 0) + (12 if f2_rvg else 0)
    conj_s  = max(0, 100-conj_p)
    sys_s   = {"split_intein_npu":75,"split_gfp":65,"fkbp_frb":55,"leucine_zipper":85,"nanobit":80}.get(split_system,65)
    agg_s   = 30 if avg_mw>15000 else (60 if avg_mw>8000 else 85)
    cov     = "intein" in split_system
    shelf_s = 75 if cov else 55
    cost_s  = max(0, 100 - n_frags*12 - (10 if f1_rmt else 0) - (8 if f2_rvg else 0) - (15 if avg_mw>15000 else 0))

    feas = round(max(0, min(100,
        frag_s*0.20 + mw_s*0.20 + conj_s*0.15 + sys_s*0.15 + agg_s*0.15 + shelf_s*0.10 + cost_s*0.05
    )))
    grade = "A" if feas>=75 else ("B" if feas>=60 else ("C" if feas>=45 else ("D" if feas>=30 else "F")))
    subs  = [
        dict(name="Fragment complexity", v=frag_s),
        dict(name="MW / expression",     v=mw_s),
        dict(name="Conjugate burden",    v=conj_s),
        dict(name="Split system MFG",    v=sys_s),
        dict(name="Aggregation risk",    v=agg_s),
        dict(name="Shelf-life",          v=shelf_s),
    ]
    subs.sort(key=lambda x: x["v"])
    return dict(
        feasibility=feas, grade=grade, subScores=subs,
        limitingFactor=subs[0]["name"], nFragments=n_frags, isCovalent=cov,
        costNote=("Feasible for GMP development" if feas>=65 else
                  "Significant manufacturing challenges" if feas>=45 else
                  "Manufacturing bottleneck — redesign recommended"),
    )


# ── SAFETY ────────────────────────────────────────────────────────────────────
def compute_safety(arch_id: str, f1_mw: float, f2_mw: float,
                   f1_rmt: bool, f2_rvg: bool, host_state: dict) -> dict:
    flags = []; pen = 0.0; avg_mw = (f1_mw + f2_mw) / 2
    ot = {"A":20,"B":15,"C":25,"D":18}.get(arch_id,20); pen += ot*0.20
    if ot>18: flags.append(dict(risk="Off-target host protein binding", severity="medium" if ot>22 else "low", note="Nanobody/DARPin may cross-react with host proteins"))
    mhc = 30 if avg_mw>15000 else (18 if avg_mw>8000 else 10); pen += mhc*0.20
    if mhc>20: flags.append(dict(risk="MHC-II immunogenicity", severity="medium", note="Large fragments generate more T-cell epitopes"))
    if f1_rmt: pen += 12*0.15; flags.append(dict(risk="TfR receptor competition (transferrin)", severity="low", note="High-dose TfR-RMT may compete with endogenous transferrin"))
    if f2_rvg: pen += 15*0.15; flags.append(dict(risk="RVG-29 / GABA-B off-target", severity="low", note="RVG-29 may interact with GABA-B receptors at high CNS concentrations"))
    nl = {"A":15,"B":15,"C":22,"D":25}.get(arch_id,15); pen += nl*0.15
    if nl>20: flags.append(dict(risk="Neuronal proteostatic burden", severity="medium", note="Large/tripartite fragments add protein load to RABV-infected neurons"))
    if arch_id=="A" and host_state.get("prior_vaccination") and host_state.get("high_inflammation"):
        pen += 20*0.15; flags.append(dict(risk="Immune over-activation", severity="high", note="IFN-β + vaccination + inflammation = excessive neuroinflammation risk"))

    score = max(0, min(100, round(100-pen)))
    level = "low" if score>=75 else ("medium" if score>=55 else "high")
    flags.sort(key=lambda f: {"high":0,"medium":1,"low":2}[f["severity"]])
    return dict(
        safetyScore=score, riskLevel=level, riskFlags=flags,
        safetyNote=("Acceptable safety profile" if level=="low" else
                    "Manageable risks — monitoring required" if level=="medium" else
                    "Significant safety concerns — redesign needed"),
    )


# ── UNCERTAINTY ───────────────────────────────────────────────────────────────
def _uncertainty(arch_id: str, final_score: float) -> dict:
    cv_b, cv_c, cv_r, cv_w, cv_m = 0.30, 0.35, 0.20, 0.25, 0.20
    rv   = math.sqrt(cv_b**2 + cv_c**2 + cv_r**2 + cv_w**2 + cv_m**2)
    mod  = {"A":0.85,"B":1.10,"C":1.40,"D":1.25}.get(arch_id, 1.0)
    sd   = rv * mod
    exp  = final_score
    opt  = min(100, round(exp*(1+sd)))
    pess = max(0,   round(exp*(1-sd)))
    conf = "medium" if sd<0.35 else ("low" if sd<0.55 else "very_low")
    dom  = {"A":"Host immune competence — IFN-β restoration efficacy is patient-dependent",
            "B":"N-protein occupancy threshold — unknown minimum fraction needed for RNP collapse",
            "C":"L-protein structure unknown — rational split site is speculative",
            "D":"Three-fragment co-localization — no in vivo validation of tripartite CNS delivery"}.get(arch_id,"Model parameter uncertainty")
    breakdown = [
        dict(param="BBB delivery",       cvPct=round(cv_b*100), contributionPct=round(cv_b**2/rv**2*100)),
        dict(param="Co-localization",    cvPct=round(cv_c*100), contributionPct=round(cv_c**2/rv**2*100)),
        dict(param="Reassembly",         cvPct=round(cv_r*100), contributionPct=round(cv_r**2/rv**2*100)),
        dict(param="Therapeutic window", cvPct=round(cv_w*100), contributionPct=round(cv_w**2/rv**2*100)),
        dict(param="Mechanism efficacy", cvPct=round(cv_m*100), contributionPct=round(cv_m**2/rv**2*100)),
    ]
    breakdown.sort(key=lambda x: -x["contributionPct"])
    return dict(expected=exp, optimistic=opt, pessimistic=pess,
                relSdPct=round(sd*100), confidence=conf,
                dominantUncertainty=dom, breakdown=breakdown)


# ── MAIN SCORE ────────────────────────────────────────────────────────────────
def score_architecture(inp: DesignerInput) -> dict:
    t0   = _time.perf_counter()
    arch = ARCHITECTURES[inp.arch_id]

    f1_mw  = inp.f1_mw  if inp.f1_mw  > 0 else arch["f1_mw"]
    f2_mw  = inp.f2_mw  if inp.f2_mw  > 0 else arch["f2_mw"]
    f1_rmt = inp.f1_rmt
    f2_rvg = inp.f2_rvg
    tri    = arch.get("tripartite", False)
    eff_sys= "split_gfp" if tri else inp.split_system

    # BBB
    f1_bbb, f1_route = _bbb_score(f1_mw, f1_rmt, False, inp.region)
    f2_bbb, f2_route = _bbb_score(f2_mw, False, f2_rvg, inp.region)
    f3_bbb, f3_route = 0.0, ""
    if tri and arch.get("f3_mw"):
        f3_bbb, f3_route = _bbb_score(arch["f3_mw"], False, arch.get("f3_rvg", True), inp.region)

    # Limiting conc: BBB score (0–100) represents % of systemic dose reaching CNS.
    # Multiply the administered conc_nm by the fractional BBB score of the weaker fragment.
    lim_conc = min(inp.conc_nm, inp.conc_nm * min(f1_bbb, f2_bbb) / 100)

    # Reassembly
    rr = compute_reassembly(eff_sys, lim_conc, 20.0)
    if tri:
        rr["probability"]    = rr["probability"] * 0.40
        rr["probabilityPct"] = round(rr["probability"] * 100)

    # Co-loc
    hs = dict(immunocompromised=inp.immunocompromised, high_inflammation=inp.high_inflammation,
               brainstem_dominant=inp.brainstem_dominant, prior_vaccination=inp.prior_vaccination,
               viral_load=inp.viral_load)
    cl = compute_colocalization(f1_bbb, f2_bbb, f1_rmt, inp.f2_rmt, f1_mw, f2_mw,
                                 inp.viral_burden, inp.region, hs)
    if tri:
        pen = (f3_bbb / 100 * 0.22) ** 0.5
        cl["colocScore"] = cl["colocScore"] * pen
        cl["colocPct"]   = round(cl["colocScore"] * 100)

    # Host mods
    mods = _host_mods(inp.arch_id, hs)

    # Mechanism
    mech_d  = _mechanism(inp.arch_id, inp, lim_conc)
    mech_s  = mech_d.get("efficacy_pct", 0) / 100
    tgt     = RABV_TARGETS.get(arch["target"], {})
    drug_s  = {"High":80,"Medium":50,"Low":25}.get(tgt.get("druggability","Medium"),50)/100
    occ_p   = (min(1.0, lim_conc/5) if arch["occupancy_required"]=="low"
               else min(1.0, lim_conc/8) if arch["occupancy_required"]=="high" else 0.70)
    mech_c  = min(1.0, drug_s * occ_p * mods["mechanismMod"])

    # Arrival / window
    f1_h = 4.0 if f1_rmt else 8.0
    f2_h = 4.0 if f2_rvg  else 8.0
    win  = compute_window(inp, arch, f1_h, f2_h, rr["tToReassembly_min"]/60)

    # Probability chain
    p_bbb  = (f1_bbb/100) * (f2_bbb/100) * mods["bbbMod"]
    p_col  = cl["colocScore"]
    p_rea  = rr["probability"] * mods["reassemblyMod"]
    p_mec  = mech_c
    p_over = p_bbb * p_col * p_rea * p_mec
    ov_pct = round(p_over * 100, 2)

    # Final score
    w_adj  = win["windowScore"] * mods["windowMod"]
    final  = max(0, min(100, round(
        ov_pct       * 0.35 +
        mech_c*100   * 0.25 +
        w_adj        * 0.25 +
        min(f1_bbb, f2_bbb) * 0.10 +
        rr["probability"]*100 * 0.05
    )))
    verdict = "Excellent" if final>=65 else ("Good" if final>=40 else ("Marginal" if final>=20 else "Poor"))

    # Bottleneck / upgrade
    bn_list = sorted([
        ("F1 BBB crossing",        f1_bbb),
        ("F2 BBB crossing",        f2_bbb),
        ("Co-localization",        cl["colocPct"]),
        ("Reassembly probability", rr["probabilityPct"]),
        ("Therapeutic window",     win["windowScore"]),
        ("Mechanism efficacy",     mech_d.get("efficacy_pct",0)),
    ], key=lambda x: x[1])
    worst = bn_list[0][0]
    upgrade = {
        "F1 BBB crossing"        : "Upgrade F1: switch to TfR-RMT (Kd ~30 nM). Reduce MW below 8 kDa.",
        "F2 BBB crossing"        : "Upgrade F2: add RVG-29, reduce HBD count. Consider TAT-CPP co-conjugation.",
        "Co-localization"        : "Route F1 and F2 via the SAME BBB pathway. Consider PEGylation for half-life synchrony.",
        "Reassembly probability" : "Switch to Npu DnaE intein (Kd 0.001 nM, t½~1 s) — irreversible at picomolar concentrations.",
        "Therapeutic window"     : "Treat earlier — each day post-CNS reduces window score ~8 pts. Anti-P (Arch A) has fastest onset.",
        "Mechanism efficacy"     : "Target occupancy insufficient. Increase dose or switch to Architecture A (low occupancy required).",
    }.get(worst, "Iterate on fragment physicochemistry.")

    # Uncertainty
    unc = _uncertainty(inp.arch_id, final)

    # Mfg + safety
    mfg = compute_manufacturing(inp.arch_id, f1_mw, f2_mw, f1_rmt, f2_rvg, tri, eff_sys)
    saf = compute_safety(inp.arch_id, f1_mw, f2_mw, f1_rmt, f2_rvg, hs)

    return dict(
        archId=inp.arch_id, finalScore=final, verdict=verdict,
        overallPct=ov_pct, mechanismScore=round(mech_d.get("efficacy_pct",0)),
        windowScore=win["windowScore"], windowPhase=win["phaseLabel"],
        windowCanPrevent=win["canPrevent"], windowCanControl=win["canControl"],
        f1Bbb=f1_bbb, f2Bbb=f2_bbb, f1Route=f1_route, f2Route=f2_route,
        f3Bbb=f3_bbb, f3Route=f3_route,
        pBbbBoth=round(p_bbb*100, 2), pColoc=cl["colocPct"],
        pReassembly=rr["probabilityPct"], pMechanism=round(mech_c*100),
        colocalization=cl, reassembly=rr, window=win,
        hostMods=mods, mechanismDetail=mech_d,
        manufacturing=mfg, safety=saf,
        uncertainty=unc,
        scorePessimistic=unc["pessimistic"], scoreExpected=unc["expected"], scoreOptimistic=unc["optimistic"],
        confidence=unc["confidence"], dominantUncertainty=unc["dominantUncertainty"],
        criticalBottleneck=worst, upgradeHint=upgrade,
        limConcNm=round(lim_conc, 3),
        strengths=arch.get("strengths",[]), weaknesses=arch.get("weaknesses",[]),
        elapsedMs=round((_time.perf_counter()-t0)*1000, 2),
    )


# ── SCORE ALL ─────────────────────────────────────────────────────────────────
def score_all(inp: DesignerInput) -> list:
    results = []
    for aid in ["A","B","C","D"]:
        inp2 = DesignerInput(**{**inp.__dict__, "arch_id": aid})
        results.append(score_architecture(inp2))
    results.sort(key=lambda r: -r["finalScore"])
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


# ── MONTE CARLO ───────────────────────────────────────────────────────────────
def run_monte_carlo(inp: DesignerInput, n: int = 500) -> dict:
    t0  = _time.perf_counter()
    rng = random.Random()

    def rn():
        u = max(1e-9, rng.random())
        return math.sqrt(-2*math.log(u)) * math.cos(2*math.pi*rng.random())

    def sp(c, cv, lo, hi, log=False):
        z = rn()
        if log:
            mu, sig = math.log(max(1e-9,c)), math.sqrt(math.log(1+cv*cv))
            v = math.exp(mu+sig*z)
        else:
            v = c + c*cv*z
        return max(lo, min(hi, v))

    ids = ["A","B","C","D"]
    rank_ct = {i:0 for i in ids}
    scores  = {i:[] for i in ids}

    for _ in range(n):
        s = DesignerInput(
            arch_id="A", region=inp.region,
            bite_mm=sp(inp.bite_mm, 0.10, 100, 1400),
            velocity=sp(inp.velocity, 0.25, 50, 400),
            treat_day=inp.treat_day,
            conc_nm=sp(inp.conc_nm, 0.40, 0.1, 100, log=True),
            viral_burden=sp(inp.viral_burden, 0.30, 0.05, 0.95),
            split_system=inp.split_system,
            f1_mw=inp.f1_mw, f1_rmt=inp.f1_rmt, f1_rvg=inp.f1_rvg,
            f2_mw=inp.f2_mw, f2_rmt=inp.f2_rmt, f2_rvg=inp.f2_rvg,
            immunocompromised=inp.immunocompromised, high_inflammation=inp.high_inflammation,
            brainstem_dominant=inp.brainstem_dominant, prior_vaccination=inp.prior_vaccination,
            viral_load=inp.viral_load,
        )
        sample = []
        for aid in ids:
            s2  = DesignerInput(**{**s.__dict__, "arch_id": aid})
            r   = score_architecture(s2)
            ns  = max(0, min(100, round(r["finalScore"] * (1 + 0.25*rn()))))
            sample.append((aid, ns))
        sample.sort(key=lambda x: -x[1])
        rank_ct[sample[0][0]] += 1
        for aid, sc in sample:
            scores[aid].append(sc)

    def pct(arr, p):
        s = sorted(arr)
        return s[int(p/100*(len(s)-1))]

    results = []
    for aid in ids:
        lst = scores[aid]
        results.append(dict(
            archId=aid,
            rankPct1=round(rank_ct[aid]/n*100),
            expected=round(sum(lst)/len(lst)),
            p10=pct(lst,10), p25=pct(lst,25), p75=pct(lst,75), p90=pct(lst,90),
            pessimistic=pct(lst,5), optimistic=pct(lst,95),
        ))
    results.sort(key=lambda r: -r["rankPct1"])
    for i,r in enumerate(results): r["mcRank"] = i+1
    best = results[0]
    certainty = ("dominant" if best["rankPct1"]>=50 else
                 "preferred" if best["rankPct1"]>=35 else
                 "marginal"  if best["rankPct1"]>=20 else "toss-up")
    return dict(results=results, n=n, bestArch=best["archId"], certainty=certainty,
                elapsedMs=round((_time.perf_counter()-t0)*1000,2))


# ── EXPLAINABILITY ────────────────────────────────────────────────────────────
def explain_score(r: dict, mfg: dict, saf: dict) -> dict:
    drivers, limiters, risks, upgrades = [], [], [], []
    win = r.get("window", {}); rr = r.get("reassembly", {}); unc = r.get("uncertainty", {})

    if r.get("f1Bbb",0)>=20: drivers.append(dict(label="F1 BBB crossing", value=r["f1Bbb"], note="Fragment 1 achieves significant BBB entry via "+r.get("f1Route","")))
    if r.get("f2Bbb",0)>=20: drivers.append(dict(label="F2 BBB crossing", value=r["f2Bbb"], note="Fragment 2 achieves significant BBB entry via "+r.get("f2Route","")))
    if r.get("mechanismScore",0)>=60: drivers.append(dict(label="Mechanism strength", value=r["mechanismScore"], note="Target is druggable and effect is mechanistically grounded"))
    if win.get("canPrevent"): drivers.append(dict(label="Pre-CNS timing", value=win.get("windowScore",0), note="Treatment before CNS seeding"))
    elif win.get("canControl"): drivers.append(dict(label="Early CNS timing", value=win.get("windowScore",0), note="Early CNS phase — containment still achievable"))
    if rr.get("probabilityPct",0)>=50: drivers.append(dict(label="Reassembly efficiency", value=rr.get("probabilityPct",0), note=rr.get("system","")+" — high assembly probability"))
    if mfg.get("feasibility",0)>=65: drivers.append(dict(label="Manufacturing feasibility", value=mfg["feasibility"], note="Grade "+mfg.get("grade","")+" — manufacturable at GMP scale"))
    if saf.get("safetyScore",0)>=70: drivers.append(dict(label="Safety profile", value=saf["safetyScore"], note="Low overall risk"))
    drivers.sort(key=lambda x: -x["value"])

    if r.get("f1Bbb",100)<15: limiters.append(dict(label="F1 BBB entry", value=r["f1Bbb"], note="Fragment 1 barely crossing"))
    if r.get("f2Bbb",100)<15: limiters.append(dict(label="F2 BBB entry", value=r["f2Bbb"], note="Fragment 2 barely crossing"))
    cl = r.get("colocalization",{})
    if cl.get("colocPct",100)<15: limiters.append(dict(label="Co-localization", value=cl.get("colocPct",0), note="Fragments unlikely to meet in same infected neuron"))
    if rr.get("probabilityPct",100)<30: limiters.append(dict(label="Reassembly probability", value=rr.get("probabilityPct",0), note="Split system Kd poorly matched to intracellular concentration"))
    if win.get("windowScore",100)<30: limiters.append(dict(label="Therapeutic window", value=win.get("windowScore",0), note="Treatment too late — "+win.get("phaseLabel","")))
    if r.get("mechanismScore",100)<35: limiters.append(dict(label="Mechanism efficacy", value=r.get("mechanismScore",0), note="Target druggability or occupancy threshold not achieved"))
    if mfg.get("feasibility",100)<50: limiters.append(dict(label="Manufacturing", value=mfg.get("feasibility",0), note="Grade "+mfg.get("grade","")+" — "+mfg.get("limitingFactor","")+" is limiting"))
    limiters.sort(key=lambda x: x["value"])

    for f in saf.get("riskFlags",[]): risks.append(dict(label=f["risk"], severity=f["severity"], note=f["note"]))
    if limiters:
        top = limiters[0]["label"]
        if "BBB"       in top: upgrades += ["Switch to TfR-RMT (optimal Kd ~30 nM) — largest single BBB improvement", "Reduce fragment MW below 8 kDa"]
        elif "Co-loc"  in top: upgrades += ["Route F1 and F2 via SAME BBB pathway — different routes halve co-localization", "PEGylate shorter-lived fragment to extend intracellular t½"]
        elif "Reassem" in top: upgrades += ["Switch to Npu DnaE split-intein (Kd 0.001 nM, t½~1 s) — irreversible at picomolar concentrations"]
        elif "window"  in top.lower(): upgrades += ["Earlier treatment — each day post-CNS reduces window score ~8 points", "Anti-P (Arch A) has fastest mechanism onset (~4 h)"]
        elif "Mfg"     in top or "Manu" in top: upgrades += ["Simplify to single conjugate: TfR-RMT OR RVG-29", "Reduce MW to <8 kDa"]
    if unc.get("dominantUncertainty"): upgrades.append("Primary uncertainty: "+unc["dominantUncertainty"]+" — validate experimentally before advancing")

    summary = ("Architecture "+r["archId"]+" shows strong performance." if not limiters else
               "Architecture "+r["archId"]+" is limited primarily by "+limiters[0]["label"]
               +" ("+str(limiters[0]["value"])+"%). Resolving this unlocks the next bottleneck.")
    return dict(archId=r["archId"], drivers=drivers, limiters=limiters, risks=risks, upgrades=upgrades, summary=summary)
