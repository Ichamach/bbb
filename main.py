"""
main.py — NeuroViral Lab BBB Engine API  Phase 2
Local FastAPI server: uvicorn main:app --reload --port 8000

Endpoints:
  POST /analyze        — full analysis (scores + kinetics + time-course)
  POST /analyze/batch  — run multiple molecules in one request
  GET  /health         — server health check
  GET  /presets        — return built-in molecule presets
  GET  /disease-params — return disease state parameter sets
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time

from bbb_core import BBBInput, analyze, REGIONAL_FACTORS, TRANSPORTER_CV

app = FastAPI(
    title="NeuroViral Lab — BBB Engine API",
    description="Phase 2 local Python ODE backend for the BBB Simulator",
    version="2.0.0",
)

# Allow requests from the local HTML files (file:// or localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost", "http://localhost:8000",
                   "http://127.0.0.1", "http://127.0.0.1:8000",
                   "null"],          # file:// origin appears as "null"
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ── REQUEST / RESPONSE MODELS ──────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    name:       str   = "Molecule"
    mw:         float = 350.0
    logp:       float = 2.0
    hbd:        int   = 2
    hba:        int   = 4
    ppb:        float = Field(0.30, ge=0.0, le=0.99)
    pka_acid:   Optional[float] = None
    pka_base:   Optional[float] = None
    mol_type:   str   = "sm"

    pgp:        bool  = False
    bcrp:       bool  = False
    mrp:        bool  = False
    rmt:        bool  = False
    rmt_kd_nm:  float = 30.0
    rvg:        bool  = False
    cation:     bool  = False

    tj:         float = 100.0
    aj:         float = 100.0
    mmp:        float = 0.0
    nfkb:       float = 0.0
    wnt:        float = 100.0
    shh:        float = 100.0
    pericyte:   float = 100.0
    notch:      float = 100.0
    angpt:      float = 100.0
    ptm:        float = 0.0
    cbf:        float = 100.0

    region:     str   = "cortex"
    species:    str   = "human"
    dose_nm:    float = 100.0
    t_hours:    float = 8.0
    dt_hours:   float = 0.02


class BatchRequest(BaseModel):
    molecules: List[AnalyzeRequest]


# ── DISEASE PRESETS (mirror JS engine) ─────────────────────────────────────────
DISEASE_PARAMS = {
    "none"               : dict(tj=100,aj=100,mmp=0,  nfkb=0,  wnt=100,shh=100,pericyte=100,notch=100,angpt=100,ptm=0,  cbf=100),
    "rabv_wildtype"      : dict(tj=95, aj=97, mmp=5,  nfkb=5,  wnt=95, shh=90, pericyte=95, notch=95, angpt=95, ptm=2,  cbf=100),
    "rabv_p_neutralized" : dict(tj=78, aj=82, mmp=18, nfkb=32, wnt=78, shh=78, pericyte=90, notch=88, angpt=82, ptm=10, cbf=100),
    "rabv_late_stage"    : dict(tj=70, aj=75, mmp=25, nfkb=40, wnt=72, shh=70, pericyte=85, notch=80, angpt=75, ptm=18, cbf=95),
    "stroke_acute"       : dict(tj=15, aj=20, mmp=95, nfkb=80, wnt=20, shh=30, pericyte=30, notch=30, angpt=5,  ptm=80, cbf=25),
    "stroke_chronic"     : dict(tj=60, aj=65, mmp=35, nfkb=45, wnt=55, shh=60, pericyte=65, notch=60, angpt=50, ptm=30, cbf=80),
    "ms"                 : dict(tj=45, aj=55, mmp=45, nfkb=90, wnt=55, shh=50, pericyte=65, notch=55, angpt=40, ptm=40, cbf=90),
    "ad"                 : dict(tj=70, aj=75, mmp=30, nfkb=40, wnt=45, shh=55, pericyte=35, notch=50, angpt=55, ptm=25, cbf=70),
    "meningitis"         : dict(tj=15, aj=20, mmp=70, nfkb=95, wnt=30, shh=35, pericyte=50, notch=40, angpt=20, ptm=50, cbf=80),
    "tumor"              : dict(tj=55, aj=60, mmp=55, nfkb=55, wnt=50, shh=45, pericyte=50, notch=50, angpt=35, ptm=30, cbf=120),
}

MOLECULE_PRESETS = {
    "glucose"     : dict(name="Glucose",     mw=180,   logp=-3.0,hbd=5, hba=6,  ppb=0.00, mol_type="glucose"),
    "caffeine"    : dict(name="Caffeine",     mw=194,   logp=-0.1,hbd=0, hba=3,  ppb=0.36, mol_type="sm"),
    "morphine"    : dict(name="Morphine",     mw=285,   logp=0.9, hbd=2, hba=4,  ppb=0.35, mol_type="sm",  pgp=True, pka_base=8.0),
    "diazepam"    : dict(name="Diazepam",     mw=285,   logp=2.9, hbd=0, hba=2,  ppb=0.98, mol_type="sm",  pgp=True),
    "haloperidol" : dict(name="Haloperidol",  mw=376,   logp=4.3, hbd=1, hba=3,  ppb=0.92, mol_type="sm"),
    "doxorubicin" : dict(name="Doxorubicin",  mw=544,   logp=1.3, hbd=6, hba=12, ppb=0.74, mol_type="sm",  pgp=True, bcrp=True, pka_base=8.2),
    "favipiravir" : dict(name="Favipiravir",  mw=157,   logp=-1.1,hbd=2, hba=3,  ppb=0.54, mol_type="sm"),
    "igg_plain"   : dict(name="IgG (no shuttle)", mw=150000,logp=-5.0,hbd=400,hba=800,ppb=0.00,mol_type="protein"),
    "tfr_rmt"     : dict(name="TfR-RMT 8kDa",mw=8200,  logp=-2.5,hbd=18,hba=28, ppb=0.20, mol_type="peptide",rmt=True, rmt_kd_nm=30.0),
    "rvg29"       : dict(name="RVG-29 ~3kDa",mw=3256,  logp=-1.8,hbd=8, hba=12, ppb=0.10, mol_type="peptide",rvg=True),
}


# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "engine": "BBB Phase 2", "version": "2.0.0"}


@app.get("/presets")
def get_presets():
    return {"presets": MOLECULE_PRESETS}


@app.get("/disease-params")
def get_disease_params():
    return {"disease_params": DISEASE_PARAMS}


@app.post("/analyze")
def analyze_molecule(req: AnalyzeRequest):
    t0 = time.perf_counter()
    try:
        inp = BBBInput(**req.dict())
        result = analyze(inp)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            # Legacy-compatible fields (JS engine can use these directly)
            "net_score"      : result.net_score,
            "best_route"     : result.best_route,
            "net_band_low"   : result.net_band_low,
            "net_band_high"  : result.net_band_high,
            "efflux_penalty" : result.efflux_penalty,
            "is_heuristic"   : result.is_heuristic,

            # Phase 2 kinetic outputs
            "kpuu_pct"       : result.kpuu_pct,
            "isf_peak_pct"   : result.isf_peak_pct,
            "neuron_pct"     : result.neuron_pct,
            "t_peak_h"       : result.t_peak_h,

            # Time-course (for plotting)
            "timeline_t"     : result.timeline_t,
            "timeline_isf"   : result.timeline_isf,
            "timeline_blood" : result.timeline_blood,
            "timeline_endo"  : result.timeline_endo,

            # Barrier state
            "paracellular_breach_pct"   : result.paracellular_breach_pct,
            "transcytosis_breach_pct"   : result.transcytosis_breach_pct,
            "endothelial_activation_pct": result.endothelial_activation_pct,
            "failure_mode"              : result.failure_mode,

            # Per-route
            "routes"         : result.routes,
            "confidence_note": result.confidence_note,
            "phase"          : result.phase,
            "elapsed_ms"     : elapsed_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch")
def analyze_batch(req: BatchRequest):
    if len(req.molecules) > 20:
        raise HTTPException(status_code=400, detail="Batch limit: 20 molecules")
    t0 = time.perf_counter()
    results = []
    for mol_req in req.molecules:
        inp = BBBInput(**mol_req.dict())
        r   = analyze(inp)
        results.append({
            "name"       : mol_req.name,
            "net_score"  : r.net_score,
            "kpuu_pct"   : r.kpuu_pct,
            "isf_peak_pct": r.isf_peak_pct,
            "best_route" : r.best_route,
            "failure_mode": r.failure_mode,
        })
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    results.sort(key=lambda x: x["kpuu_pct"], reverse=True)
    return {"results": results, "count": len(results), "elapsed_ms": elapsed_ms}


@app.post("/analyze/disease-compare")
def disease_compare(req: AnalyzeRequest):
    """Run the same molecule across all disease states and return comparison."""
    t0 = time.perf_counter()
    comparison = []
    base = req.dict()
    for disease, params in DISEASE_PARAMS.items():
        merged = {**base, **params}
        inp = BBBInput(**merged)
        r   = analyze(inp)
        comparison.append({
            "disease"    : disease,
            "net_score"  : r.net_score,
            "kpuu_pct"   : r.kpuu_pct,
            "failure_mode": r.failure_mode,
            "para_breach": r.paracellular_breach_pct,
            "tcy_breach" : r.transcytosis_breach_pct,
        })
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {"molecule": req.name, "comparison": comparison, "elapsed_ms": elapsed_ms}
