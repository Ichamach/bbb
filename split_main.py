"""
split_main.py  —  NeuroViral Lab  Split-Protein Designer API  v2.0
FastAPI server on port 8002.

Run:  uvicorn split_main:app --reload --port 8002

Endpoints:
  GET  /health
  POST /score          — score one architecture
  POST /score/all      — score all 4 architectures, ranked
  POST /montecarlo     — Monte Carlo rank stability (n samples)
  POST /compare/scenarios — batch clinical scenario comparison
  POST /explain        — structured explainability report
"""

import time
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from split_core import (
    DesignerInput, score_architecture, score_all,
    run_monte_carlo, explain_score,
    compute_manufacturing, compute_safety,
    ARCHITECTURES, SPLIT_SYSTEMS, CLINICAL_SCENARIOS,
)

# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroViral Lab — Split-Protein Designer",
    version="2.0.0",
    description="Split-protein therapeutic architecture scoring engine for rabies CNS therapy.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # covers file://, localhost:3000, localhost:8002, and any dev origin
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# ── REQUEST MODELS ────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    arch_id:           str   = "A"
    region:            str   = "hippocampus"
    bite_mm:           float = 1200.0
    velocity:          float = 200.0
    treat_day:         float = 0.0
    conc_nm:           float = Field(5.0, ge=0.01, le=1000.0)
    viral_burden:      float = Field(0.30, ge=0.01, le=0.99)
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
    n_samples:         int   = Field(500, ge=50, le=2000)
    # ── Real physicochemical properties from FASTA sequence import ───────────
    # All optional — when None, split_core falls back to MW-derived estimates
    f1_logp:           Optional[float] = None
    f1_hbd:            Optional[int]   = None
    f1_hba:            Optional[int]   = None
    f1_ppb:            Optional[float] = None
    f1_charge:         Optional[float] = None
    f1_pka_acid:       Optional[float] = None
    f1_pka_base:       Optional[float] = None
    f2_logp:           Optional[float] = None
    f2_hbd:            Optional[int]   = None
    f2_hba:            Optional[int]   = None
    f2_ppb:            Optional[float] = None
    f2_charge:         Optional[float] = None
    f2_pka_acid:       Optional[float] = None
    f2_pka_base:       Optional[float] = None


class ScenarioRequest(BaseModel):
    base: ScoreRequest = ScoreRequest()
    scenarios: Optional[List[dict]] = None   # None → use built-in CLINICAL_SCENARIOS


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _to_input(req: ScoreRequest) -> DesignerInput:
    return DesignerInput(
        arch_id=req.arch_id, region=req.region,
        bite_mm=req.bite_mm, velocity=req.velocity,
        treat_day=req.treat_day, conc_nm=req.conc_nm,
        viral_burden=req.viral_burden, split_system=req.split_system,
        f1_mw=req.f1_mw, f1_rmt=req.f1_rmt, f1_rvg=req.f1_rvg,
        f2_mw=req.f2_mw, f2_rmt=req.f2_rmt, f2_rvg=req.f2_rvg,
        immunocompromised=req.immunocompromised,
        high_inflammation=req.high_inflammation,
        brainstem_dominant=req.brainstem_dominant,
        prior_vaccination=req.prior_vaccination,
        viral_load=req.viral_load,
        isf_peak_pct=req.isf_peak_pct or 0.0,
        kpuu_pct=req.kpuu_pct or 0.0,
        n_samples=req.n_samples,
        # Real physicochemical props from FASTA import
        f1_logp=req.f1_logp, f1_hbd=req.f1_hbd, f1_hba=req.f1_hba,
        f1_ppb=req.f1_ppb,   f1_charge=req.f1_charge,
        f1_pka_acid=req.f1_pka_acid, f1_pka_base=req.f1_pka_base,
        f2_logp=req.f2_logp, f2_hbd=req.f2_hbd, f2_hba=req.f2_hba,
        f2_ppb=req.f2_ppb,   f2_charge=req.f2_charge,
        f2_pka_acid=req.f2_pka_acid, f2_pka_base=req.f2_pka_base,
    )


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"    : "ok",
        "engine"    : "Split-Protein Designer",
        "version"   : "2.0.0",
        "port"      : 8002,
        "architectures": list(ARCHITECTURES.keys()),
        "split_systems": list(SPLIT_SYSTEMS.keys()),
    }


@app.post("/score")
def score_one(req: ScoreRequest):
    t0 = time.perf_counter()
    try:
        r = score_architecture(_to_input(req))
        r["requestedArch"] = req.arch_id
        r["elapsedMs"]     = round((time.perf_counter() - t0) * 1000, 2)
        return r
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Unknown arch_id or split_system: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/all")
def score_all_archs(req: ScoreRequest):
    t0 = time.perf_counter()
    try:
        results = score_all(_to_input(req))
        return {
            "results"   : results,
            "bestArch"  : results[0]["archId"],
            "elapsedMs" : round((time.perf_counter() - t0) * 1000, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/montecarlo")
def monte_carlo(req: ScoreRequest):
    try:
        return run_monte_carlo(_to_input(req), n=req.n_samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare/scenarios")
def compare_scenarios(req: ScenarioRequest):
    t0       = time.perf_counter()
    base_inp = _to_input(req.base)
    scenarios = req.scenarios or CLINICAL_SCENARIOS

    comparison = []
    for sc in scenarios:
        label = sc.get("label", sc.get("scenario", "Scenario"))
        scores = {}
        window_phases = {}
        for aid in ["A", "B", "C", "D"]:
            merged = DesignerInput(
                arch_id=aid,
                region=base_inp.region,
                bite_mm=sc.get("bite_mm", base_inp.bite_mm),
                velocity=sc.get("velocity", base_inp.velocity),
                treat_day=sc.get("treat_day", base_inp.treat_day),
                conc_nm=base_inp.conc_nm,
                viral_burden=base_inp.viral_burden,
                split_system=base_inp.split_system,
                f1_mw=base_inp.f1_mw, f1_rmt=base_inp.f1_rmt, f1_rvg=base_inp.f1_rvg,
                f2_mw=base_inp.f2_mw, f2_rmt=base_inp.f2_rmt, f2_rvg=base_inp.f2_rvg,
                immunocompromised=sc.get("immunocompromised", base_inp.immunocompromised),
                high_inflammation=sc.get("high_inflammation", base_inp.high_inflammation),
                brainstem_dominant=sc.get("brainstem_dominant", base_inp.brainstem_dominant),
                prior_vaccination=sc.get("prior_vaccination", base_inp.prior_vaccination),
                viral_load=sc.get("viral_load", base_inp.viral_load),
            )
            r = score_architecture(merged)
            scores[aid]        = r["finalScore"]
            window_phases[aid] = r["window"]["phaseLabel"]

        sorted_archs = sorted(scores, key=lambda a: -scores[a])
        comparison.append(dict(
            scenario=label, scores=scores,
            window_phases=window_phases,
            best_arch=sorted_archs[0],
            ranking=sorted_archs,
        ))

    return {
        "comparison": comparison,
        "elapsedMs" : round((time.perf_counter() - t0) * 1000, 2),
    }


@app.post("/explain")
def explain(req: ScoreRequest):
    try:
        inp  = _to_input(req)
        r    = score_architecture(inp)
        arch = ARCHITECTURES[req.arch_id]
        mfg  = compute_manufacturing(
            req.arch_id,
            r["f1Bbb"] * 100, r["f2Bbb"] * 100,
            req.f1_rmt, req.f2_rvg,
            arch.get("tripartite", False),
            inp.split_system,
        )
        saf = compute_safety(
            req.arch_id,
            r["f1Bbb"] * 100, r["f2Bbb"] * 100,
            req.f1_rmt, req.f2_rvg,
            dict(
                immunocompromised=req.immunocompromised,
                high_inflammation=req.high_inflammation,
                brainstem_dominant=req.brainstem_dominant,
                prior_vaccination=req.prior_vaccination,
                viral_load=req.viral_load,
            ),
        )
        return explain_score(r, mfg, saf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── DEV ENTRY ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("split_main:app", host="0.0.0.0", port=8002, reload=True)
