"""
rabv_main.py — NeuroViral Lab RABV Engine API  v1.0
Local FastAPI server: uvicorn rabv_main:app --reload --port 8001

Mirrors main.py (BBB engine on port 8000) exactly in structure.

Endpoints:
  POST /simulate          — full neuroinvasion simulation
  POST /simulate/batch    — run multiple scenarios (≤20)
  POST /simulate/therapy  — compare therapy strategies on one scenario
  GET  /health            — server health check
  GET  /presets           — built-in infection scenarios
"""

import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rabv_core import RABVInput, simulate

app = FastAPI(
    title="NeuroViral Lab — RABV Engine API",
    description="v1.0 9-compartment ODE backend for RABV Neuroinvasion Simulator",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost", "http://localhost:8001",
                   "http://127.0.0.1", "http://127.0.0.1:8001",
                   "null"],          # file:// origin appears as "null"
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ── REQUEST MODEL ─────────────────────────────────────────────────────────────
class SimulateRequest(BaseModel):
    bite_location:       str   = "foot"
    bite_depth:          str   = "muscle"
    viral_dose:          str   = "med"
    wound_washing:       str   = "none"

    base_velocity_mm_day: float = 200.0
    p75_active:          bool  = True
    replication_lag_h:   float = 9.0

    prot_G: bool = True
    prot_P: bool = True
    prot_N: bool = True
    prot_M: bool = True
    prot_L: bool = True

    vaccinated:          bool  = False
    pep_given:           bool  = False
    immunocompromised:   bool  = False
    pep_day:             float = 0.0
    pep_hrig_given:      bool  = False

    p_inhibitor_active:   bool  = False
    p_inhibitor_efficacy: float = 0.75
    p_inhibitor_day:      float = 0.0
    n_dn_active:          bool  = False
    n_dn_efficacy:        float = 0.70
    n_dn_day:             float = 0.0
    favipiravir_active:   bool  = False
    favipiravir_efficacy: float = 0.40
    favipiravir_day:      float = 0.0

    t_days:              float = Field(60.0, ge=1.0, le=180.0)
    dt_hours:            float = Field(0.25, ge=0.05, le=1.0)
    output_interval_h:   float = Field(2.0,  ge=0.5, le=12.0)


class BatchRequest(BaseModel):
    scenarios: List[SimulateRequest]


# ── BUILT-IN PRESETS ──────────────────────────────────────────────────────────
INFECTION_PRESETS = {
    "standard_foot_bite": dict(
        bite_location="foot", bite_depth="muscle", viral_dose="med",
        wound_washing="none", base_velocity_mm_day=200, p75_active=True,
        vaccinated=False, pep_given=False,
    ),
    "face_bite_severe": dict(
        bite_location="face", bite_depth="deep", viral_dose="high",
        wound_washing="none", base_velocity_mm_day=200, p75_active=True,
        vaccinated=False, pep_given=False,
    ),
    "bat_hand_exposure": dict(
        bite_location="hand", bite_depth="superficial", viral_dose="low",
        wound_washing="within_1h", base_velocity_mm_day=200, p75_active=True,
        vaccinated=False, pep_given=False,
    ),
    "vaccinated_foot_pep": dict(
        bite_location="foot", bite_depth="muscle", viral_dose="med",
        wound_washing="within_1h", base_velocity_mm_day=200, p75_active=True,
        vaccinated=True, pep_given=True, pep_day=0.0, pep_hrig_given=True,
    ),
    "immunocompromised_foot": dict(
        bite_location="foot", bite_depth="muscle", viral_dose="med",
        wound_washing="none", base_velocity_mm_day=200, p75_active=True,
        vaccinated=False, pep_given=False, immunocompromised=True,
    ),
    "p_inhibitor_day3": dict(
        bite_location="foot", bite_depth="muscle", viral_dose="med",
        wound_washing="none", base_velocity_mm_day=200, p75_active=True,
        vaccinated=False, pep_given=False,
        p_inhibitor_active=True, p_inhibitor_efficacy=0.80, p_inhibitor_day=3.0,
    ),
    "n_dn_day5": dict(
        bite_location="foot", bite_depth="muscle", viral_dose="med",
        wound_washing="none", base_velocity_mm_day=200, p75_active=True,
        vaccinated=False, pep_given=False,
        n_dn_active=True, n_dn_efficacy=0.70, n_dn_day=5.0,
    ),
    "split_protein_arch_a": dict(
        bite_location="foot", bite_depth="muscle", viral_dose="med",
        wound_washing="none", base_velocity_mm_day=200, p75_active=True,
        vaccinated=False, pep_given=False,
        p_inhibitor_active=True, p_inhibitor_efficacy=0.78, p_inhibitor_day=4.0,
        n_dn_active=False,
    ),
}

# Therapy comparison strategies for /simulate/therapy endpoint
THERAPY_STRATEGIES = [
    dict(label="No treatment",        p_inhibitor_active=False, n_dn_active=False, favipiravir_active=False, pep_given=False),
    dict(label="PEP day 0",           pep_given=True,  pep_day=0.0, pep_hrig_given=True),
    dict(label="PEP day 3",           pep_given=True,  pep_day=3.0, pep_hrig_given=False),
    dict(label="P-inhibitor day 0",   p_inhibitor_active=True, p_inhibitor_efficacy=0.80, p_inhibitor_day=0.0),
    dict(label="P-inhibitor day 3",   p_inhibitor_active=True, p_inhibitor_efficacy=0.80, p_inhibitor_day=3.0),
    dict(label="N dominant-neg day 4",n_dn_active=True, n_dn_efficacy=0.70, n_dn_day=4.0),
    dict(label="Favipiravir day 0",   favipiravir_active=True, favipiravir_efficacy=0.40, favipiravir_day=0.0),
    dict(label="Split-protein Arch A",p_inhibitor_active=True, p_inhibitor_efficacy=0.78, p_inhibitor_day=4.0, n_dn_active=False),
    dict(label="PEP + P-inhibitor",   pep_given=True, pep_day=0.0, pep_hrig_given=True, p_inhibitor_active=True, p_inhibitor_efficacy=0.78, p_inhibitor_day=4.0),
]


# ── SHARED RESPONSE BUILDER ───────────────────────────────────────────────────
def _build_response(result, elapsed_ms: float) -> dict:
    return {
        # Timing milestones
        "day_cns_entry":      result.day_cns_entry,
        "day_brainstem":      result.day_brainstem,
        "day_limbic":         result.day_limbic,
        "day_symptoms":       result.day_symptoms,
        "day_fatal":          result.day_fatal,
        "transport_days":     result.transport_days,
        "incubation_days":    result.incubation_days,

        # Burden
        "burden_at_end":      result.burden_at_end,
        "peak_cns_burden":    result.peak_cns_burden,

        # BBB
        "bbb_tj_final":       result.bbb_tj_final,
        "bbb_nfkb_final":     result.bbb_nfkb_final,

        # Immune
        "innate_peak":        result.innate_peak,
        "adapt_peak":         result.adapt_peak,
        "pep_effective":      result.pep_effective,

        # Outcome
        "outcome":            result.outcome,
        "outcome_prob":       result.outcome_prob,

        # Spread
        "peripheral_spread_prob": result.peripheral_spread_prob,

        # Phase structure (for JS phase navigator)
        "phase_ends":         result.phase_ends,

        # Time-courses
        "timeline_days":      result.timeline_days,
        "timeline_wound":     result.timeline_wound,
        "timeline_pns":       result.timeline_pns,
        "timeline_sc":        result.timeline_sc,
        "timeline_bs":        result.timeline_bs,
        "timeline_limbic":    result.timeline_limbic,
        "timeline_cortex":    result.timeline_cortex,
        "timeline_sal":       result.timeline_sal,
        "timeline_innate":    result.timeline_innate,
        "timeline_adapt":     result.timeline_adapt,
        "timeline_bbb_tj":    result.timeline_bbb_tj,
        "timeline_bbb_nfkb":  result.timeline_bbb_nfkb,

        # Compartment snapshot
        "compartments_final": result.compartments_final,

        # Metadata
        "confidence_note":    result.confidence_note,
        "engine_version":     result.engine_version,
        "elapsed_ms":         elapsed_ms,
    }


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "engine": "RABV v1.0", "version": "1.0.0"}


@app.get("/presets")
def get_presets():
    return {"presets": INFECTION_PRESETS}


@app.post("/simulate")
def simulate_infection(req: SimulateRequest):
    t0 = time.perf_counter()
    try:
        inp    = RABVInput(**req.dict())
        result = simulate(inp)
        ms     = round((time.perf_counter() - t0) * 1000, 2)
        return _build_response(result, ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate/batch")
def simulate_batch(req: BatchRequest):
    if len(req.scenarios) > 20:
        raise HTTPException(status_code=400, detail="Batch limit: 20 scenarios")
    t0 = time.perf_counter()
    results = []
    for s in req.scenarios:
        inp = RABVInput(**s.dict())
        r   = simulate(inp)
        results.append({
            "bite_location":   s.bite_location,
            "viral_dose":      s.viral_dose,
            "day_cns_entry":   r.day_cns_entry,
            "day_symptoms":    r.day_symptoms,
            "incubation_days": r.incubation_days,
            "outcome":         r.outcome,
            "outcome_prob":    r.outcome_prob,
            "peak_cns_burden": r.peak_cns_burden,
            "bbb_tj_final":    r.bbb_tj_final,
        })
    ms = round((time.perf_counter() - t0) * 1000, 2)
    return {"results": results, "count": len(results), "elapsed_ms": ms}


@app.post("/simulate/therapy")
def simulate_therapy_comparison(req: SimulateRequest):
    """
    Run the same infection scenario against all therapy strategies.
    Returns a ranked comparison — equivalent to BBB /analyze/disease-compare.
    """
    t0   = time.perf_counter()
    base = req.dict()
    comparison = []

    for strategy in THERAPY_STRATEGIES:
        merged = {**base, **strategy}
        # Remove label before building input
        label = merged.pop("label", "Unknown")
        try:
            inp = RABVInput(**merged)
            r   = simulate(inp)
            comparison.append({
                "strategy":       label,
                "outcome":        r.outcome,
                "outcome_prob":   r.outcome_prob,
                "day_cns_entry":  r.day_cns_entry,
                "day_symptoms":   r.day_symptoms,
                "peak_cns_burden":r.peak_cns_burden,
                "pep_effective":  r.pep_effective,
                "bbb_tj_final":   r.bbb_tj_final,
                "bbb_nfkb_final": r.bbb_nfkb_final,
                "adapt_peak":     r.adapt_peak,
            })
        except Exception as e:
            comparison.append({"strategy": label, "error": str(e)})

    # Sort by outcome_prob descending
    comparison.sort(key=lambda x: x.get("outcome_prob", 0), reverse=True)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    return {"comparison": comparison, "count": len(comparison), "elapsed_ms": ms}
