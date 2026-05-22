"""
splitter_main.py — NeuroViral Lab Protein Splitter Engine API  v1.0
Local FastAPI server: uvicorn splitter_main:app --reload --port 8003

Mirrors main.py (BBB, port 8000) and rabv_main.py (RABV, port 8001) in structure.

Endpoints:
  POST /analyze              — full split-site analysis for one protein
  POST /analyze/batch        — run ≤10 proteins in one request
  POST /analyze/site         — score a single user-specified cut position
  POST /analyze/target-compare — compare same design across all RABV targets
  GET  /health               — server health check
  GET  /presets              — RABV target presets
  GET  /split-systems        — available split system parameters
  GET  /benchmark            — experimental benchmark calibration set

Usage:
  pip install fastapi uvicorn pydantic --break-system-packages
  uvicorn splitter_main:app --reload --port 8003
"""

import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from splitter_core import (
    run_full_analysis,
    score_split_delivery_total,
    RABV_TARGET_MAPS,
    SPLIT_SYSTEM_REACH,
    find_split_site_candidates,
    estimate_residue_properties,
    detect_functional_motifs,
    parse_pdb,
    build_contact_map,
    apply_structure_to_props,
    rank_delivery_strategies,
    parse_fasta,
)

app = FastAPI(
    title="NeuroViral Lab — Protein Splitter Engine API",
    description="v1.0 structure-aware split-site prediction. Port 8002.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REQUEST MODELS ────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    sequence:         str   = ""
    target_id:        str   = ""
    protein_name:     str   = "Protein"
    pdb_text:         str   = ""
    af_pdb_text:      str   = ""
    split_system:     str   = "split_intein_npu"
    bbb_strategy:     str   = "tfr_rmt"
    delivery_mode:    str   = "protein_biologic"
    conc_nm:          float = Field(5.0,  ge=0.01, le=1000.0)
    target_tissue:    str   = "cns"
    min_fragment_aa:  int   = Field(50,   ge=20,   le=200)
    top_n:            int   = Field(10,   ge=1,    le=30)
    run_mc:           bool  = True
    mc_n:             int   = Field(500,  ge=50,   le=2000)


class SingleSiteRequest(BaseModel):
    sequence:      str
    target_id:     str   = ""
    cut_position:  int
    split_system:  str   = "split_intein_npu"
    bbb_strategy:  str   = "tfr_rmt"
    delivery_mode: str   = "protein_biologic"
    conc_nm:       float = 5.0
    pdb_text:      str   = ""


class BatchRequest(BaseModel):
    proteins: List[AnalyzeRequest]


class TargetCompareRequest(BaseModel):
    cut_positions: Dict[str, int] = Field(
        default={'P': 140, 'N': 230, 'L': 550},
    )
    split_system:  str   = "split_intein_npu"
    bbb_strategy:  str   = "tfr_rmt"
    delivery_mode: str   = "protein_biologic"
    conc_nm:       float = 5.0


# ── STATIC DATA ───────────────────────────────────────────────────────────────

PRESETS = {
    tid: {
        "name"         : m.get("name", tid),
        "length"       : m.get("length", 0),
        "pdb"          : m.get("pdb", "—"),
        "known_good_splits": m.get("annotations", {}).get("known_good_splits", []),
        "note"         : m.get("note", ""),
        "default_params": {
            "split_system" : "split_intein_npu" if tid != "L" else "split_intein_npu",
            "bbb_strategy" : "none"    if tid == "L" else "tfr_rmt",
            "delivery_mode": "dual_aav" if tid == "L" else "protein_biologic",
            "conc_nm"      : 5.0,
        },
    }
    for tid, m in RABV_TARGET_MAPS.items()
}

BENCHMARK_SPLITS = [
    {"protein":"GFP",               "position":214, "outcome":"success",
     "note":"Split-GFP 1-10/11 (Cabantous 2005) — canonical success"},
    {"protein":"GFP",               "position":157, "outcome":"success",
     "note":"Alternative split-GFP site — moderate efficiency"},
    {"protein":"Firefly luciferase","position":437, "outcome":"success",
     "note":"Split-Luc at 437 — high complementation (Dixon 2016)"},
    {"protein":"Cas9",              "position":573, "outcome":"success",
     "note":"Split-Cas9 at 573 — dual AAV (Truong 2015)"},
    {"protein":"Cas9",              "position":637, "outcome":"success",
     "note":"Split-Cas9 at 637 — alternative site (Chew 2016)"},
    {"protein":"TEV protease",      "position":118, "outcome":"success",
     "note":"Reconstituted TEV — intein-mediated (Mootz 2003)"},
    {"protein":"GFP",               "position":100, "outcome":"failure",
     "note":"Core β-barrel — destroys fold"},
    {"protein":"GFP",               "position":180, "outcome":"failure",
     "note":"Sheet interior — no complementation"},
    {"protein":"DHFR",              "position":1,   "outcome":"failure",
     "note":"N-terminus — F1 fragment too short"},
]


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status" : "ok",
        "engine" : "Protein Splitter v1.0",
        "version": "1.0.0",
        "port"   : 8003,
        "targets": list(PRESETS.keys()),
    }


@app.get("/presets")
def get_presets():
    return {"presets": PRESETS}


@app.get("/split-systems")
def get_split_systems():
    return {
        "systems": {
            k: {"reach_angstrom": v} for k, v in SPLIT_SYSTEM_REACH.items()
        }
    }


@app.get("/benchmark")
def get_benchmark():
    return {"benchmark_splits": BENCHMARK_SPLITS, "count": len(BENCHMARK_SPLITS)}


@app.post("/analyze")
def analyze_protein(req: AnalyzeRequest):
    t0 = time.perf_counter()
    try:
        result = run_full_analysis(
            sequence         = req.sequence     or None,
            target_id        = req.target_id    or None,
            pdb_text         = req.pdb_text     or None,
            af_pdb_text      = req.af_pdb_text  or None,
            split_system     = req.split_system,
            bbb_strategy     = req.bbb_strategy,
            delivery_mode    = req.delivery_mode,
            conc_nm          = req.conc_nm,
            target_tissue    = req.target_tissue,
            top_n            = req.top_n,
            run_mc           = req.run_mc,
            mc_n             = req.mc_n,
            constraints      = {"min_fragment_aa": req.min_fragment_aa},
        )
        if result.get("error"):
            raise HTTPException(status_code=422, detail=result["error"])

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        result["elapsed_ms"] = elapsed
        result["_source"]    = "python_ode"
        return result

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch")
def analyze_batch(req: BatchRequest):
    if len(req.proteins) > 10:
        raise HTTPException(status_code=400, detail="Batch limit: 10 proteins")
    t0 = time.perf_counter()
    results = []
    for pr in req.proteins:
        try:
            r = run_full_analysis(
                sequence      = pr.sequence  or None,
                target_id     = pr.target_id or None,
                split_system  = pr.split_system,
                bbb_strategy  = pr.bbb_strategy,
                delivery_mode = pr.delivery_mode,
                conc_nm       = pr.conc_nm,
                top_n         = min(pr.top_n, 5),
                run_mc        = False,
            )
            s = r.get("summary", {})
            results.append({
                "protein_name"     : r.get("target", pr.protein_name),
                "target_id"        : pr.target_id,
                "sequence_length"  : r.get("N", 0),
                "top_position"     : s.get("top_position"),
                "top_score"        : s.get("top_score"),
                "top_verdict"      : s.get("top_verdict"),
                "passed_candidates": s.get("passed_candidates"),
                "rejected_count"   : s.get("rejected_count"),
                "has_structural"   : s.get("has_structural_data", False),
                "calibration_passed": s.get("calibration_passed"),
                "delivery_top"     : r["deliveryRanking"][0]["label"] if r.get("deliveryRanking") else None,
            })
        except Exception as e:
            results.append({"protein_name": pr.protein_name or pr.target_id,
                             "error": str(e)})

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    results.sort(key=lambda r: r.get("top_score") or 0, reverse=True)
    return {"results": results, "count": len(results), "elapsed_ms": elapsed}


@app.post("/analyze/site")
def analyze_single_site(req: SingleSiteRequest):
    """Score one user-specified cut position in full detail."""
    t0 = time.perf_counter()
    try:
        # Resolve sequence
        if req.target_id and req.target_id in RABV_TARGET_MAPS:
            preset      = RABV_TARGET_MAPS[req.target_id]
            seq         = preset.get("sequence") or req.sequence or ""
            annotations = preset.get("annotations", {})
            domains     = preset.get("domains", [])
        else:
            fasta_r     = parse_fasta(req.sequence or "")
            seq         = fasta_r.get("sequence", "")
            annotations = {}
            domains     = []

        seq = "".join(c for c in seq.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
        cut = req.cut_position
        if cut < 1 or cut >= len(seq):
            raise HTTPException(status_code=422,
                detail=f"cut_position {cut} out of range (sequence length {len(seq)})")

        pdb_result  = parse_pdb(req.pdb_text)  if req.pdb_text  else None
        if pdb_result and getattr(pdb_result, 'error', None):
            pdb_result = None
        contact_map = build_contact_map(pdb_result) if pdb_result else None

        props = estimate_residue_properties(seq)
        if pdb_result:
            props = apply_structure_to_props(props, pdb_result, None)

        motif_hits = detect_functional_motifs(seq)

        site = {
            "position"     : cut,
            "aa"           : seq[cut - 1],
            "ss"           : "loop",
            "f1_length"    : cut,
            "f2_length"    : len(seq) - cut,
            "spell_score"  : 0.50,
            "has_structure": bool(pdb_result),
            "plddt"        : None,
            "b_factor"     : None,
            "subscores"    : {"loop":1.0, "accessibility":0.5, "conservation":0.5,
                              "func_distance":0.5, "balance":0.5, "disorder":0.5},
            "min_dist_to_site": None,
        }

        result = score_split_delivery_total(
            site, seq,
            split_system  = req.split_system,
            conc_nm       = req.conc_nm,
            delivery_mode = req.delivery_mode,
            bbb_strategy  = req.bbb_strategy,
            annotations   = annotations,
            domains       = domains,
            target_id     = req.target_id or None,
            contact_map   = contact_map,
            motif_hits    = motif_hits,
        )

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        result["elapsed_ms"] = elapsed
        result["_source"]    = "python_ode"
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/target-compare")
def target_compare(req: TargetCompareRequest):
    """Score a cut position on each RABV target — equivalent to BBB disease-compare."""
    t0 = time.perf_counter()
    comparison = []

    for target_id, cut_pos in req.cut_positions.items():
        if target_id not in RABV_TARGET_MAPS:
            comparison.append({"target_id": target_id, "error": "Unknown target"})
            continue
        try:
            preset      = RABV_TARGET_MAPS[target_id]
            seq         = preset.get("sequence", "")
            seq         = "".join(c for c in seq.upper() if c in "ACDEFGHIKLMNPQRSTVWY")
            if not seq or cut_pos < 1 or cut_pos >= len(seq):
                comparison.append({"target_id": target_id,
                                    "error": f"No sequence or cut {cut_pos} out of range"})
                continue

            annotations = preset.get("annotations", {})
            domains     = preset.get("domains", [])
            motif_hits  = detect_functional_motifs(seq)

            site = {
                "position": cut_pos, "aa": seq[cut_pos - 1],
                "ss": "loop", "f1_length": cut_pos, "f2_length": len(seq) - cut_pos,
                "spell_score": 0.5, "has_structure": False, "plddt": None, "b_factor": None,
                "subscores": {"loop":1,"accessibility":0.5,"conservation":0.5,
                              "func_distance":0.5,"balance":0.5,"disorder":0.5},
                "min_dist_to_site": None,
            }

            r = score_split_delivery_total(
                site, seq,
                split_system  = req.split_system,
                conc_nm       = req.conc_nm,
                delivery_mode = req.delivery_mode,
                bbb_strategy  = req.bbb_strategy,
                annotations   = annotations,
                domains       = domains,
                target_id     = target_id,
                motif_hits    = motif_hits,
            )

            comparison.append({
                "target_id"       : target_id,
                "protein_name"    : preset.get("name", target_id),
                "cut_position"    : cut_pos,
                "final_score"     : r.get("final_score", 0),
                "verdict"         : r.get("verdict", "Poor"),
                "is_rejected"     : r.get("is_rejected", False),
                "rejection_reason": r.get("rejection_reason", ""),
                "domain_class"    : (r.get("domain_score") or {}).get("classification"),
                "assembly_class"  : (r.get("assembly_score") or {}).get("assembly_class"),
                "assembly_risk"   : (r.get("assembly_score") or {}).get("assembly_risk"),
                "f1_grade"        : (r.get("f1_foldability") or {}).get("grade"),
                "f2_grade"        : (r.get("f2_foldability") or {}).get("grade"),
                "mfg_grade"       : (r.get("manufacturability") or {}).get("grade"),
                "subscores"       : r.get("subscores", {}),
            })
        except Exception as e:
            comparison.append({"target_id": target_id, "error": str(e)})

    comparison.sort(key=lambda x: x.get("final_score", -1), reverse=True)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "comparison" : comparison,
        "count"      : len(comparison),
        "best_target": comparison[0].get("target_id") if comparison else None,
        "elapsed_ms" : elapsed,
    }
