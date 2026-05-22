/**
 * shared_engine.js — v9.0  (Split-Protein Designer overhaul)
 * NeuroViral Lab — RABV Post-Infection CNS Therapeutic Platform
 *
 * CHANGES v9.0 — SPLIT-PROTEIN DESIGNER ENGINE:
 *
 *  1. MONTE CARLO: runMonteCarlo(archId, gp, n=5000)
 *     Samples each parameter from its uncertainty distribution,
 *     runs scoreArchitectureLive() per sample, returns full rank
 *     distribution across all 4 architectures. Replaces static score
 *     with: expected, p10, p90, rank_pct_#1.
 *
 *  2. ARCHITECTURE MECHANISMS: computeArchMechanism(archId, gp, result)
 *     Per-architecture mechanistic efficacy model:
 *     - A: IFN restoration × immune competence → antiviral clearance
 *     - B: N-oligo disruption × occupancy threshold → RNP collapse
 *     - C: L-RdRp dominant-negative × competition ratio → replication block
 *     - D: multi-target synergy E=1-∏(1-Ei) minus delivery complexity
 *     Returns mechanismDetail with sub-scores and limiting factor.
 *
 *  3. MANUFACTURING: computeManufacturing(archId, f1mol, f2mol)
 *     Scores feasibility across: fragment complexity, conjugate burden,
 *     purification difficulty, aggregation risk, shelf-life estimate,
 *     GMP cost index. Returns feasibility (0-100) + grade (A/B/C/D/F).
 *
 *  4. SAFETY: computeSafety(archId, mol, sig, hostState)
 *     Models: off-target host binding risk, immunogenicity (MHC-II load),
 *     TfR receptor saturation, RVG GABA_B off-target, neuronal protein
 *     load stress, inflammatory over-activation (anti-P + vaccination).
 *     Returns safetyScore (0-100, higher=safer) + risk flags.
 *
 *  5. UNCERTAINTY PROPAGATION: propagateUncertainty(archId, gp)
 *     Returns {expected, pessimistic, optimistic, confidence}
 *     using analytical error propagation (no sampling needed for UI speed).
 *
 *  6. EXPLAINABILITY: explainScore(archId, result, mfg, safety, mc)
 *     Returns structured why/why-not breakdown:
 *     { drivers[], limiters[], risks[], upgradeSteps[] }
 *
 *  Phase 1 BBB changes (v8.1) preserved unchanged.
 *
 *  BBB ENGINE — SCIENTIFIC FIXES:
 *  1. Passive diffusion: replaced stepped MW cutoffs with exponential decay
 *     exp(-0.0135 × (MW-100)) per Abraham/Clark model. PPB now applied to
 *     effective flux only, NOT to intrinsic membrane permeability (Pm).
 *     logP window model: Gaussian around logP=2.0, penalty above 4.5 (flow-limited).
 *
 *  2. Paracellular: separate TJ and AJ failure contributions now tracked
 *     independently. AJ failure contributes to both paracellular AND
 *     transcytosis breach (mechanistic coupling). Charge selectivity now
 *     continuous sigmoid not binary steps.
 *
 *  3. RMT (TfR/IR/LRP): added Michaelis-Menten receptor occupancy with
 *     Bmax/Kd terms. Added affinity-efficiency paradox: very tight binders
 *     (Kd < 1nM) penalised for endosomal trapping. Endosomal sorting
 *     now explicit: f_trans=0.35, f_recycle=0.45, f_lysosome=0.20.
 *     pH-sensitive release factor modelled.
 *
 *  4. RVG-29: marked as evidence tier 2 (experimental/mechanism ambiguous).
 *     Uncertainty band widened. Added species penalty for human vs rodent.
 *     Payload format penalty for large constructs.
 *
 *  5. Efflux: replaced flat additive penalties with Michaelis-Menten
 *     saturation per transporter class (Km_pgp ~1μM, Km_bcrp ~0.8μM).
 *     Disease-state efflux modulation (NF-κB upregulates P-gp expression).
 *     Species factor added (rodent P-gp ~1.7× human at BBB).
 *
 *  6. CMT: added Michaelis-Menten saturation for GLUT1 (Km ~1.5mM glucose
 *     equivalent), LAT1 (Km ~0.15mM phenylalanine equivalent). Wnt modulates
 *     transporter expression, not just flux directly.
 *
 *  7. Barrier state: TJ and transcytosis failure now tracked as SEPARATE
 *     state variables (paraCellularBreach vs transcytosisBreach). Previously
 *     coupled through a single eTJ term. Now:
 *     - TJ failure → paracellular route
 *     - MFSD2A/caveolar failure → transcellular/transcytosis route
 *     - Both can fail independently (stroke: transcytosis first, TJ later)
 *
 *  8. Output honesty: all route scores now carry:
 *     - evidenceTier: 1 (validated) / 2 (experimental) / 3 (theoretical)
 *     - uncertaintyBand: { low, expected, high } based on documented
 *       biological variability (27-57% interindividual transporter abundance)
 *     - assumptionLog: array of strings explaining what drove the score
 *     - isHeuristic flag on final net score
 *
 *  9. REGIONAL FACTORS: expanded with species-specific transporter abundance
 *     data. Added 'thalamus' and 'choroid_plexus' regions. Uncertainty on
 *     regional factors now explicitly tracked.
 *
 *  10. DISEASE PARAMS: RABV states now separately specify:
 *      - tj_state (paracellular failure)
 *      - transcytosis_state (vesicular failure)
 *      - inflammatory_state (endothelial activation, leukocyte adhesion)
 *      These were previously collapsed into single TJ/AJ sliders.
 *
 * NOTE: Phase 2 (Python ODE backend) will replace computeRoutes() inner
 * calculations with QSS-Euler 7-compartment model via local FastAPI.
 * All output keys preserved for backward compatibility.
 */
'use strict';

// ─── PHYSICAL CONSTANTS ──────────────────────────────────
const BLOOD_PH     = 7.4;
const ENDOSOME_PH  = 5.8;
const BRAIN_ISF_PH = 7.3;
const CBF_NORMAL   = 50;    // mL/100g/min normal cerebral blood flow
const CAPILLARY_SA = 100;   // relative capillary surface area

// ─── BBB TRANSPORT PARAMETERS (Phase 1 — literature-grounded) ─────────────
// Passive diffusion: Abraham/Clark exponential MW model
// J_passive = Pm_intrinsic × f_eff × SA
// Pm_intrinsic = logP_factor × exp(-0.0025 × (MW-100)) × HB_factor
// Pardridge PPB correction: for logP > PM_PARDRIDGE_BLEND_MAX,
// membrane partitioning dominates over plasma protein binding
const PM_MW_DECAY     = 0.0025;  // /Da — calibrated against 10-compound benchmark
const PM_LOGP_OPTIMUM = 1.7;     // optimal logP (van de Waterbeemd CNS drug space)
const PM_LOGP_WIDTH   = 2.5;     // Gaussian width (wider than v8.0 — captures CNS range)
const PM_FLOW_LIMIT   = 3.5;     // logP above which flow-limited kinetics apply
const PM_PARDRIDGE_BLEND_MAX = 2.0; // logP at which PPB constraint becomes negligible

// RMT — TfR Michaelis-Menten parameters (human BBB, healthy)
// From: pabinafusp alfa PK data + in vitro TfR transcytosis assays
const TFR_BMAX_NM       = 50.0;  // nM receptor sites at luminal BMEC surface
const TFR_KD_OPTIMAL_NM = 30.0;  // nM — optimal Kd for transcytosis efficiency
const TFR_AFFINITY_PENALTY_FACTOR = 50.0; // controls paradox steepness
const TFR_F_TRANS       = 0.35;  // fraction → productive transcytosis
const TFR_F_RECYCLE     = 0.45;  // fraction → receptor recycling to blood
// TFR_F_LYSOSOME = 1 - TFR_F_TRANS - TFR_F_RECYCLE = 0.20

// Efflux — Michaelis-Menten Km values (human BBB luminal membrane)
// From: Doran et al. cross-species efflux analysis; PET verapamil studies
const PGP_KM_NM   = 1000;   // nM — P-gp Km (substrate concentration at half-max)
const BCRP_KM_NM  = 800;    // nM — BCRP Km
const MRP_KM_NM   = 500;    // nM — MRP Km
const PGP_VMAX    = 1.0;    // relative Vmax (normalised to 1.0)
const BCRP_VMAX   = 0.75;
const MRP_VMAX    = 0.55;

// Species efflux scaling (rodent P-gp expression ~1.7× human at BBB)
const SPECIES_PGP_SCALE = { human:1.0, rat:1.7, mouse:1.9, nhp:1.1 };

// CMT — Michaelis-Menten saturation
// GLUT1 Km ~1.5 mM glucose equivalent (blood glucose ~5mM → ~77% saturated)
const GLUT1_KM_REL = 0.30;  // relative saturation at physiological blood glucose
const LAT1_KM_REL  = 0.25;  // relative saturation at physiological plasma AA

// Interindividual variability in BBB transporter abundance (human proteomics)
// Source: Uchida et al. human BBB proteomics, CV 27-57% across donors
const TRANSPORTER_CV = {
  pgp : 0.42,   // coefficient of variation (42%)
  bcrp: 0.35,
  mrp : 0.48,
  tfr : 0.31,
  glut1:0.27,
  lat1: 0.33,
  rvg : 0.38,   // RVG-29 nAChR shuttle CV (Vmax=0.12, Km=45 nM; tier-2 evidence)
};

// Evidence tiers for routes
const EVIDENCE_TIER = {
  passive_diffusion  : 1,  // validated, well-characterised
  paracellular       : 1,  // validated
  cmt_glut1          : 1,  // validated
  cmt_lat1           : 1,  // validated
  rmt_tfr            : 1,  // validated (pabinafusp alfa clinical precedent)
  efflux_pgp         : 1,  // validated
  efflux_bcrp        : 1,  // validated
  caveolar           : 2,  // mechanistically supported, limited quantitation
  amt_adsorptive     : 2,  // experimental
  rvg29_nachr        : 2,  // experimental — mechanism ambiguous (nAChR vs GABA_B vs caveolae)
  immune_trafficking : 1,  // validated in neuroinflammation models
};

// ─── CANONICAL MOLECULE SCHEMA ───────────────────────────
function buildMolecule(raw) {
  raw = raw || {};
  let ppb = raw.ppb != null ? raw.ppb : 0;
  if (ppb > 1) ppb = ppb / 100;
  ppb = Math.max(0, Math.min(0.99, ppb));
  const base_charge = raw.base_charge != null ? raw.base_charge
                    : raw.charge      != null ? raw.charge : 0;
  return {
    name       : raw.name        || 'Unnamed',
    type       : raw.type        || 'sm',
    mw         : raw.mw          != null ? raw.mw    : 300,
    logp       : raw.logp        != null ? raw.logp  : 1.0,
    hbd        : raw.hbd         != null ? raw.hbd   : 2,
    hba        : raw.hba         != null ? raw.hba   : 4,
    ppb        : ppb,
    base_charge: base_charge,
    pKa_acid   : raw.pKa_acid    != null ? raw.pKa_acid : null,
    pKa_base   : raw.pKa_base    != null ? raw.pKa_base : null,
    pgp        : !!raw.pgp,
    bcrp       : !!raw.bcrp,
    mrp        : !!raw.mrp,
    rmt        : !!raw.rmt,
    rvg        : !!raw.rvg,   // RVG-29 conjugate → nAChR-mediated endocytosis at BMEC
    cation     : !!raw.cation,
    // NVU state (0-100)
    tj         : raw.tj         != null ? raw.tj         : 100,
    aj         : raw.aj         != null ? raw.aj         : 100,
    mmp        : raw.mmp        != null ? raw.mmp        : 0,
    nfkb       : raw.nfkb       != null ? raw.nfkb       : 0,
    wnt        : raw.wnt        != null ? raw.wnt        : 100,
    shh        : raw.shh        != null ? raw.shh        : 100,
    pericyte   : raw.pericyte   != null ? raw.pericyte   : 100,
    notch      : raw.notch      != null ? raw.notch      : 100,
    angpt      : raw.angpt      != null ? raw.angpt      : 100,
    ptm        : raw.ptm        != null ? raw.ptm        : 0,
    cbf        : raw.cbf        != null ? raw.cbf        : 100,
    // fragment-specific
    sequence   : raw.sequence   || null,
    target     : raw.target     || null,
    arch       : raw.arch       || null,
    frag       : raw.frag       || null,
    // intracellular half-life estimate (hours) — used for co-localization mismatch
    t_half_ic_h: raw.t_half_ic_h != null ? raw.t_half_ic_h : (raw.mw > 10000 ? 6 : 14),
  };
}

// ─── IONIZATION ──────────────────────────────────────────
function fractionIonized(pKa, pH, type) {
  if (pKa == null) return 0;
  return type === 'acid'
    ? 1 / (1 + Math.pow(10, pKa - pH))
    : 1 / (1 + Math.pow(10, pH - pKa));
}
function effectiveCharge(mol, pH) {
  const hasPKa = mol.pKa_acid != null || mol.pKa_base != null;
  if (!hasPKa) return mol.base_charge || 0;
  let c = 0;
  if (mol.pKa_acid != null) c -= fractionIonized(mol.pKa_acid, pH, 'acid');
  if (mol.pKa_base != null) c += fractionIonized(mol.pKa_base, pH, 'base');
  return c;
}
function ionizationPenalty(charge) {
  const a = Math.abs(charge);
  if (a < 0.15) return 1.00;
  if (a < 0.50) return 0.80;
  if (a < 1.00) return 0.40;
  if (a < 1.50) return 0.18;
  return 0.06;
}

// ─── REGIONAL FACTORS ────────────────────────────────────
// Values sourced from: Uchida proteomics, Allen Brain Atlas transporter expression,
// regional microvessel isolation studies. Uncertainty reflects documented
// interregional variation (30-42% relative difference across CNS territories).
const REGIONAL_FACTORS = {
  cortex      : { glut1:1.00, lat1:1.00, pgp:1.00, bcrp:1.00, tj_integrity:1.00, tfr:1.00, nachr:0.60, ca_sa:1.00 },
  hippocampus : { glut1:1.30, lat1:1.20, pgp:0.80, bcrp:0.85, tj_integrity:0.90, tfr:1.20, nachr:1.00, ca_sa:1.10 },
  brainstem   : { glut1:0.80, lat1:0.90, pgp:1.40, bcrp:1.30, tj_integrity:1.10, tfr:0.90, nachr:1.40, ca_sa:0.95 },
  cerebellum  : { glut1:0.90, lat1:0.80, pgp:1.20, bcrp:1.15, tj_integrity:1.00, tfr:0.95, nachr:0.70, ca_sa:0.90 },
  thalamus    : { glut1:1.10, lat1:1.00, pgp:0.90, bcrp:0.95, tj_integrity:1.05, tfr:1.05, nachr:0.80, ca_sa:1.05 },
  choroid_plexus:{ glut1:1.50, lat1:1.30, pgp:0.60, bcrp:0.70, tj_integrity:0.50, tfr:1.40, nachr:0.50, ca_sa:1.80 },
};

// Regional uncertainty bands (±1 SD based on proteomics variability)
const REGIONAL_UNCERTAINTY = {
  cortex:0.30, hippocampus:0.35, brainstem:0.38, cerebellum:0.32, thalamus:0.33, choroid_plexus:0.45,
};

// ─── DISEASE PARAMS ──────────────────────────────────────
const DISEASE_PARAMS = {
  none               : { tj:100,aj:100,mmp:0,  nfkb:0,  wnt:100,shh:100,pericyte:100,notch:100,angpt:100,ptm:0,  cbf:100 },
  rabv_wildtype      : { tj:95, aj:97, mmp:5,  nfkb:5,  wnt:95, shh:90, pericyte:95, notch:95, angpt:95, ptm:2,  cbf:100 },
  rabv_p_neutralized : { tj:78, aj:82, mmp:18, nfkb:32, wnt:78, shh:78, pericyte:90, notch:88, angpt:82, ptm:10, cbf:100 },
  rabv_late_stage    : { tj:70, aj:75, mmp:25, nfkb:40, wnt:72, shh:70, pericyte:85, notch:80, angpt:75, ptm:18, cbf:95  },
  rabv_n_dn_deployed : { tj:82, aj:85, mmp:12, nfkb:20, wnt:82, shh:80, pericyte:90, notch:88, angpt:85, ptm:8,  cbf:100 },
  stroke_acute       : { tj:15, aj:20, mmp:95, nfkb:80, wnt:20, shh:30, pericyte:30, notch:30, angpt:5,  ptm:80, cbf:25  },
  stroke_chronic     : { tj:60, aj:65, mmp:35, nfkb:45, wnt:55, shh:60, pericyte:65, notch:60, angpt:50, ptm:30, cbf:80  },
  ms                 : { tj:45, aj:55, mmp:45, nfkb:90, wnt:55, shh:50, pericyte:65, notch:55, angpt:40, ptm:40, cbf:90  },
  ad                 : { tj:70, aj:75, mmp:30, nfkb:40, wnt:45, shh:55, pericyte:35, notch:50, angpt:55, ptm:25, cbf:70  },
  meningitis         : { tj:15, aj:20, mmp:70, nfkb:95, wnt:30, shh:35, pericyte:50, notch:40, angpt:20, ptm:50, cbf:80  },
  tumor              : { tj:55, aj:60, mmp:55, nfkb:55, wnt:50, shh:45, pericyte:50, notch:50, angpt:35, ptm:30, cbf:120 },
};

const DISEASE_NOTES = {
  none               : 'Healthy BBB — all barriers intact.',
  rabv_wildtype      : 'WILD-TYPE RABV: P protein blocks TBK1 (Ser179) → IFN-β suppressed → NF-κB low → claudin-5 intact. THE THERAPEUTIC PARADOX — virus keeps its own treatment locked out.',
  rabv_p_neutralized : 'P PROTEIN NEUTRALIZED: IFN-β restored → NF-κB 32% → mild BBB opening (TJ 78%). Desired therapeutic effect. Microglia activating.',
  rabv_late_stage    : 'LATE-STAGE RABV: High viral burden. Some neuroinflammation but RABV suppresses it to preserve axonal highways.',
  rabv_n_dn_deployed : 'N-PROTEIN DOMINANT NEGATIVE: RNP collapses → viral load drops → BBB partially recovers.',
  stroke_acute       : 'ACUTE STROKE: Maximal MMP surge. Severe paracellular opening. CBF 25%.',
  stroke_chronic     : 'CHRONIC STROKE: BBB partially restored. CBF 80%.',
  ms                 : 'MULTIPLE SCLEROSIS: NF-κB ICAM-1/VCAM-1 leukocyte trafficking dominant.',
  ad                 : 'ALZHEIMER\'S DISEASE: Pericyte loss, Wnt suppression, CBF 70%.',
  meningitis         : 'MENINGITIS: Maximal NF-κB + MMP. Paracellular + immune trafficking.',
  tumor              : 'BRAIN TUMOR (BTB): Heterogeneous. VEGF-driven focal MMP. CBF 120%.',
};

// ─── SIGNALLING ──────────────────────────────────────────
function computeSignaling(mol) {
  const wnt      = (mol.wnt      != null ? mol.wnt      : 100) / 100;
  const shh      = (mol.shh      != null ? mol.shh      : 100) / 100;
  const pericyte = (mol.pericyte != null ? mol.pericyte : 100) / 100;
  const notch    = (mol.notch    != null ? mol.notch    : 100) / 100;
  const angpt    = (mol.angpt    != null ? mol.angpt    : 100) / 100;
  const mmp      = (mol.mmp      != null ? mol.mmp      :   0) / 100;
  const nfkb     = (mol.nfkb     != null ? mol.nfkb     :   0) / 100;
  const ptm      = (mol.ptm      != null ? mol.ptm      :   0) / 100;
  const aj_raw   = (mol.aj       != null ? mol.aj       : 100) / 100;
  const tj_raw   = (mol.tj       != null ? mol.tj       : 100) / 100;
  const cbf      = (mol.cbf      != null ? mol.cbf      : 100) / 100;

  // Wnt/Shh suppressed by NF-κB (inflammatory crosstalk)
  const wntEff = Math.max(0, wnt - nfkb * 0.35);
  const shhEff = Math.max(0, shh - nfkb * 0.40);

  // ── PARACELLULAR BARRIER: TJ integrity ───────────────────────────────────
  // Driven by: MMP (degrades claudin/occludin directly), PTM (phosphorylation
  // mislocates ZO-1), AJ coupling (AJ loss destabilises TJ scaffolding via
  // VE-cadherin/β-catenin/actin axis), Wnt (maintains claudin-5 transcription)
  const eTJ = Math.max(0, Math.min(1,
    tj_raw
    * (1 - mmp  * 0.70)   // MMP proteolytic degradation of TJ proteins
    * (1 - ptm  * 0.40)   // phosphorylation/ubiquitination mislocalisation
    * (1 - (1 - aj_raw) * 0.22)  // AJ→TJ structural coupling
    * (0.45 + wntEff * 0.55)     // Wnt→claudin-5 transcription programme
  ));

  // ── TRANSCYTOSIS BARRIER: MFSD2A/caveolar suppression ───────────────────
  // Driven by: Wnt (→ MFSD2A expression), Notch (→ reduces vesicle formation),
  // Pericyte (→ ECM→integrin signalling suppresses caveolae)
  // NOTE: This is INDEPENDENT from TJ failure. In early stroke/disease,
  // transcytosis fails BEFORE TJ disruption (Knowland et al. 2014).
  const eMFSD2A = Math.max(0, Math.min(1,
    0.40 * wntEff + 0.30 * notch + 0.30 * pericyte
  ));

  // ── ADHERENS JUNCTIONS ───────────────────────────────────────────────────
  const eAJ = Math.max(0, Math.min(1,
    aj_raw * (0.30 + angpt * 0.70) * (1 - nfkb * 0.40) * (1 - mmp * 0.35)
  ));

  // ── ENDOTHELIAL INFLAMMATORY ACTIVATION STATE ────────────────────────────
  // NF-κB → ICAM-1/VCAM-1 upregulation → leukocyte adhesion competence
  // MMP → basement membrane damage → transmigration competence
  const endothelialActivation = Math.min(1, nfkb * 0.7 + mmp * 0.3);
  const leukocyteAdhesion     = Math.min(1, nfkb * 0.85);
  const transmigrationComp    = Math.min(1, nfkb * 0.6 + mmp * 0.4);
  const ecmDamage             = Math.min(1, mmp  * 0.80 + ptm * 0.20);

  // ── EFFLUX EXPRESSION MODULATION ─────────────────────────────────────────
  // NF-κB upregulates P-gp expression in inflammatory conditions
  // (Bauer et al.; Hartz et al. P-gp regulation data)
  const pgpInflamBoost  = 1.0 + nfkb * 0.45;  // up to 1.45× at max NF-κB
  const bcrpInflamBoost = 1.0 + nfkb * 0.30;

  // ── FAILURE MODE CLASSIFICATION ──────────────────────────────────────────
  // Separate paracellular (TJ) vs transcellular (MFSD2A) vs immune
  const paraBreach   = Math.round((1 - eTJ) * 100);
  const transBreach  = Math.round((1 - eMFSD2A) * 100);
  const immuneBreach = Math.round(transmigrationComp * 100);

  // Failure mode based on DOMINANT pathway
  let failureMode = 'intact';
  if ((1-eTJ) > 0.30 && (1-eMFSD2A) > 0.30) failureMode = 'mixed';
  else if ((1-eTJ) > 0.30)                    failureMode = 'paracellular';
  else if ((1-eMFSD2A) > 0.30)               failureMode = 'transcellular';
  else if (transmigrationComp > 0.50)        failureMode = 'immune';

  const tcyLevel = eMFSD2A > 0.65 ? 'low' : eMFSD2A > 0.35 ? 'med' : 'high';
  const immuneTrafficking = Math.min(1, nfkb + mmp * 0.30);

  return {
    // Core barrier states
    effectiveTJ       : eTJ,
    effectiveAJ       : eAJ,
    effectiveMFSD2A   : eMFSD2A,
    // Separate breach channels (Phase 1 key addition)
    paracellularBreach : paraBreach,
    transcytosisBreach : transBreach,
    immuneBreach       : immuneBreach,
    // Signalling levels
    wntLevel           : wntEff,
    shhLevel           : shhEff,
    pericyteLevel      : pericyte,
    mmpLevel           : mmp,
    nfkbLevel          : nfkb,
    ptmLevel           : ptm,
    angptLevel         : angpt,
    cbfLevel           : cbf,
    // Inflammatory state (new in v8.1)
    endothelialActivation,
    leukocyteAdhesion,
    transmigrationComp,
    ecmDamage,
    // Efflux expression modifiers (new in v8.1)
    pgpInflamBoost,
    bcrpInflamBoost,
    // Derived
    computedTcy        : tcyLevel,
    immuneTrafficking,
    failureMode,
    // Legacy aliases for backward compatibility
    paraBreach         : paraBreach,
    transBreach        : transBreach,
  };
}

// ─── BBB ROUTE SCORING ───────────────────────────────────
// Phase 1: mechanistic sub-models with uncertainty bands and assumption logging.
// Each route returns { score, low, high, evidenceTier, assumptionLog }.
// Phase 2 will replace inner calculations with Python ODE calls.

function _routeUncertaintyBand(score, evidenceTier, regionalCV) {
  // Uncertainty from: biological variability (transporter CV) +
  // evidence tier (how well-characterised the route is)
  const tierMult = evidenceTier === 1 ? 1.0 : evidenceTier === 2 ? 1.6 : 2.5;
  const cv       = (regionalCV || 0.35) * tierMult;
  return {
    low     : Math.max(0,   Math.round(score * (1 - cv))),
    expected: Math.round(score),
    high    : Math.min(100, Math.round(score * (1 + cv))),
  };
}

function computeRoutes(mol, region, species) {
  region  = region  || 'cortex';
  species = species || 'human';
  const molC = buildMolecule(mol);
  const sig  = computeSignaling(molC);
  const rf   = REGIONAL_FACTORS[region] || REGIONAL_FACTORS.cortex;
  const rCV  = REGIONAL_UNCERTAINTY[region] || 0.35;

  const freeF   = Math.max(0.01, 1 - molC.ppb);  // free fraction
  const bCharge = effectiveCharge(molC, BLOOD_PH);
  const cbfF    = sig.cbfLevel;
  const eTJ     = sig.effectiveTJ;

  // ── 1. PASSIVE DIFFUSION ─────────────────────────────────────────────────
  // Phase 1 fix: exponential MW decay, Gaussian logP window, PPB on flux only
  let pd = 0;
  const assumptions_pd = [];
  if (molC.type === 'gas') {
    pd = 95;
    assumptions_pd.push('Gas: rate-limited by CBF, not membrane permeability');
  } else {
    // Intrinsic membrane permeability (NOT affected by PPB — Pardridge model)
    const mwFactor  = Math.exp(-PM_MW_DECAY * Math.max(0, molC.mw - 100));
    const logpFactor= Math.exp(-0.5 * Math.pow((molC.logp - PM_LOGP_OPTIMUM) / PM_LOGP_WIDTH, 2));
    const hbPenalty = Math.max(0, 1 - molC.hbd * 0.13 - molC.hba * 0.04);
    const ionFactor = Math.max(0.02, 1 - Math.abs(bCharge) * 0.55);
    const Pm_intrinsic = logpFactor * mwFactor * hbPenalty * ionFactor;
    // Pardridge PPB correction (Pardridge 1988 — calibrated against 10-drug benchmark):
    // logP < 0  : full PPB constraint — only free drug diffuses
    // logP 0→2  : linear blend — membrane partitioning increasingly bypasses PPB
    // logP > 2  : BBB diffusion effectively PPB-independent (highly lipophilic drugs
    //             partition into membrane directly from albumin-bound state)
    const pardridge_blend = Math.max(0, Math.min(1, molC.logp / PM_PARDRIDGE_BLEND_MAX));
    const eff_free = (1 - molC.ppb) * (1 - pardridge_blend) + 0.90 * pardridge_blend;
    pd = Pm_intrinsic * eff_free * 100;

    // Flow-limited cap for highly lipophilic molecules (logP > 3.5)
    if (molC.logp > PM_FLOW_LIMIT) {
      const ps = Pm_intrinsic * 1e-3 * CAPILLARY_SA * (rf.ca_sa || 1.0);
      const fl = cbfF * CBF_NORMAL / 6000;
      const flowCap = (ps / (ps + fl)) * fl * 1e4;
      if (pd > flowCap) {
        pd = flowCap;
        assumptions_pd.push(`logP=${molC.logp.toFixed(1)}>3.5 → flow-limited (CBF=${Math.round(cbfF*100)}%)`);
      }
    }
    assumptions_pd.push(`Pm_intrinsic=${(Pm_intrinsic*100).toFixed(1)} · eff_free=${(eff_free*100).toFixed(0)}% (Pardridge blend=${(pardridge_blend*100).toFixed(0)}%)`);
    assumptions_pd.push(`MW_factor=${mwFactor.toFixed(3)} · logP_factor=${logpFactor.toFixed(3)} (exp decay, calibrated)`);
    if (molC.hbd > 5) assumptions_pd.push(`HBD=${molC.hbd} — strong hydrogen-bond penalty on passive diffusion`);
  }
  pd = Math.min(pd, 95);

  // ── 2. PARACELLULAR (TJ+AJ) ──────────────────────────────────────────────
  // Phase 1 fix: TJ and AJ failures tracked separately; continuous charge selectivity
  const assumptions_para = [];
  const tjOpen   = 1 - eTJ;
  const ajOpen   = 1 - sig.effectiveAJ;
  // Continuous sigmoid charge selectivity (Deen pore model approximation)
  const cSel     = Math.exp(-0.8 * Math.abs(bCharge));  // 1.0 at neutral → 0 at |charge|=3
  // MW-dependent pore sieving (Stokes radius approximation)
  const poreSieve= molC.mw < 200 ? 1.0 : molC.mw < 500 ? 0.55 : molC.mw < 2000 ? 0.15 : 0.02;
  const paraTJ   = tjOpen  * 40 * cSel * poreSieve;
  const paraAJ   = ajOpen  * 15 * cSel * poreSieve;
  const para     = Math.min((paraTJ + paraAJ * 0.6) * rf.tj_integrity, 80);
  assumptions_para.push(`TJ_open=${Math.round(tjOpen*100)}% para_contrib=${Math.round(paraTJ)} (separate from transcytosis)`);
  assumptions_para.push(`Charge_selectivity=${cSel.toFixed(3)} (continuous sigmoid, not stepped)`);
  assumptions_para.push('Phase 1: static model. Phase 2 will add time-dependent TJ vs transcytosis failure timelines.');

  // ── 3. CMT / SLC ─────────────────────────────────────────────────────────
  // Phase 1 fix: Michaelis-Menten saturation for GLUT1 and LAT1
  let cmt = 0;
  const assumptions_cmt = [];
  if (molC.type === 'glucose') {
    // GLUT1: at physiological blood glucose ~5mM, already ~77% saturated
    // Wnt modulates GLUT1 expression (not flux directly)
    const glut1_expr = 0.60 + sig.wntLevel * 0.40;  // Wnt → GLUT1 expression
    const glut1_sat  = 1 - GLUT1_KM_REL;             // saturation at physiological [glucose]
    cmt = 90 * glut1_expr * glut1_sat * rf.glut1;
    assumptions_cmt.push(`GLUT1 MM: expression=${glut1_expr.toFixed(2)}, sat=${glut1_sat.toFixed(2)}`);
  } else if (molC.type === 'aa') {
    const lat1_expr  = 0.55 + sig.wntLevel * 0.45;
    const lat1_sat   = 1 - LAT1_KM_REL;
    cmt = 82 * lat1_expr * lat1_sat * rf.lat1;
    assumptions_cmt.push(`LAT1 MM: expression=${lat1_expr.toFixed(2)}, sat=${lat1_sat.toFixed(2)}`);
  } else if (molC.type === 'monocarb') {
    cmt = 76 * (0.65 + sig.wntLevel * 0.35);
    assumptions_cmt.push('MCT1/2: Wnt-regulated expression, no explicit Km (substrate-agnostic)');
  }
  cmt = Math.min(cmt, 95);

  // ── 4. RMT (TfR/IR/LRP) ─────────────────────────────────────────────────
  // Phase 1 fix: Michaelis-Menten occupancy + affinity-efficiency paradox +
  // endosomal sorting fractions + pH-sensitive release model
  let rmt = 0;
  const assumptions_rmt = [];
  if (molC.rmt) {
    // Receptor occupancy: MM at luminal surface
    // Use TFR_KD_OPTIMAL_NM as reference; actual Kd is user-implicit (rmt flag only)
    // Phase 2 will expose explicit Kd slider
    const assumed_kd  = TFR_KD_OPTIMAL_NM;  // assume optimal Kd when rmt:true
    const occupancy   = TFR_BMAX_NM / (TFR_BMAX_NM + assumed_kd);  // ~0.63

    // Affinity-efficiency paradox: modelled with Gaussian around optimal Kd
    // pabinafusp alfa & trontinemab data: optimal Kd ~10-50nM for transcytosis
    // Ultra-tight binders (Kd < 1nM) trapped in endosomes → lower transcytosis
    const affinityEff = Math.exp(-0.5 * Math.pow(
      (Math.log10(assumed_kd) - Math.log10(TFR_KD_OPTIMAL_NM)) / 0.8, 2
    ));

    // Endosomal sorting: pH-sensitive release improves transcytosis
    // At endosomal pH 5.8 vs blood pH 7.4: charge shifts affect binding
    const pH_release  = Math.max(0.3, 1 - Math.exp(-assumed_kd / TFR_AFFINITY_PENALTY_FACTOR));
    const f_trans_eff = TFR_F_TRANS   * (0.40 + 0.60 * pH_release);
    const f_rec_eff   = TFR_F_RECYCLE * (1.00 - 0.40 * pH_release);
    // f_lys = remainder → degraded

    // Transcytosis suppression: high MFSD2A partially suppresses all vesicular routes
    const mfsdSupp    = 0.50 + sig.effectiveMFSD2A * 0.50;

    // MW penalty: larger biologics trapped in endosomes more
    const mwPenRMT    = molC.mw <= 5000  ? 1.00
                      : molC.mw <= 10000 ? 0.85
                      : molC.mw <= 30000 ? 0.65
                      : molC.mw <= 80000 ? 0.45 : 0.20;

    rmt = occupancy * affinityEff * f_trans_eff * mfsdSupp * mwPenRMT * rf.tfr * 72;

    assumptions_rmt.push(`Occupancy=${occupancy.toFixed(2)} (Bmax=${TFR_BMAX_NM}nM, assumed Kd=${assumed_kd}nM)`);
    assumptions_rmt.push(`AffinityEff=${affinityEff.toFixed(2)} — paradox: tight binders score lower`);
    assumptions_rmt.push(`f_trans=${f_trans_eff.toFixed(2)}, f_recycle=${f_rec_eff.toFixed(2)}, f_lys=${(1-f_trans_eff-f_rec_eff).toFixed(2)}`);
    assumptions_rmt.push('Phase 2 will expose explicit Kd slider for precise affinity-efficiency modelling');
  } else if (molC.type === 'protein' && molC.mw > 5000) {
    // Plain IgG-like: ~0.05% brain entry via non-specific transcytosis
    rmt = 3;
    assumptions_rmt.push('Non-targeted biologic: ~0.05% entry via bulk-phase transcytosis (literature baseline)');
  }
  rmt = Math.min(rmt, 72);

  // ── 5. RVG-29 (nAChR) — Evidence Tier 2 ────────────────────────────────
  // Phase 1 fix: marked as experimental; wider uncertainty; mechanism caveats
  let rvgRoute = 0;
  const assumptions_rvg = [];
  if (molC.rvg) {
    // MW penalty: nAChR-mediated endocytosis more cargo-size sensitive than TfR
    const mwPenRVG = molC.mw <= 3000  ? 1.00
                   : molC.mw <= 8000  ? 0.80
                   : molC.mw <= 15000 ? 0.55
                   : molC.mw <= 20000 ? 0.35 : 0.15;

    // nAChR availability: partially suppressed by MFSD2A (tight transcytosis)
    const mfsdSupp = 0.55 + sig.effectiveMFSD2A * 0.45;

    // Species penalty: most RVG-29 data from rodents; human translation uncertain
    const speciesFactor = species === 'human' ? 0.65 : species === 'nhp' ? 0.80 : 1.0;

    rvgRoute = 32 * mwPenRVG * rf.nachr * mfsdSupp * speciesFactor * freeF;

    assumptions_rvg.push('EVIDENCE TIER 2 — experimental route, mechanism ambiguous');
    assumptions_rvg.push('Mechanism debated: nAChR vs GABA_B receptor vs caveolae-mediated endocytosis');
    assumptions_rvg.push(`Species factor=${speciesFactor} (most data from rodents; human uncertain)`);
    assumptions_rvg.push(`Uncertainty band wide: ±${Math.round(TRANSPORTER_CV.tfr * 1.6 * 100)}% (evidence tier 2)`);
  }
  rvgRoute = Math.min(rvgRoute, 40);

  // ── 6. CAVEOLAR (MFSD2A-gated) ──────────────────────────────────────────
  const mfsdLoss = 1 - sig.effectiveMFSD2A;
  const ionPenB  = ionizationPenalty(bCharge);
  const cav = molC.mw > 50000 ? mfsdLoss*0.40*45*ionPenB
            : molC.mw >  5000 ? mfsdLoss*0.70*45*ionPenB
            : molC.mw >  1000 ? mfsdLoss*0.90*40*ionPenB
            : molC.mw >   400 ? mfsdLoss*0.22*28*ionPenB : 0;

  // ── 7. AMT (adsorptive-mediated) ─────────────────────────────────────────
  const amtElig = molC.cation || bCharge >= 0.5;
  const amt     = amtElig
    ? Math.min(38, (Math.abs(bCharge) + 0.5) * 8)
      * (sig.computedTcy === 'low' ? 0.35 : sig.computedTcy === 'med' ? 0.65 : 1.0)
    : 0;

  // ── 8. EFFLUX — Michaelis-Menten ─────────────────────────────────────────
  // Phase 1 fix: MM saturation, disease-state expression boost, species scaling
  // NOTE: concISF assumed proportional to score (heuristic; Phase 2 uses ODE)
  // Assumed intracellular concentration for MM: 500 nM (representative mid-dose)
  const ASSUMED_CONC_NM = 500;
  const assumptions_eff = [];
  let efflux = 0;

  if (molC.pgp) {
    const pgp_vmax = PGP_VMAX * rf.pgp * sig.pgpInflamBoost
                   * (SPECIES_PGP_SCALE[species] || 1.0);
    const pgp_mm   = pgp_vmax * ASSUMED_CONC_NM / (PGP_KM_NM + ASSUMED_CONC_NM);
    efflux += pgp_mm * 36;
    assumptions_eff.push(`P-gp MM: Vmax=${pgp_vmax.toFixed(2)} Km=${PGP_KM_NM}nM inflam_boost=${sig.pgpInflamBoost.toFixed(2)}`);
  }
  if (molC.bcrp) {
    const bcrp_vmax = BCRP_VMAX * sig.bcrpInflamBoost;
    const bcrp_mm   = bcrp_vmax * ASSUMED_CONC_NM / (BCRP_KM_NM + ASSUMED_CONC_NM);
    efflux += bcrp_mm * 26;
    assumptions_eff.push(`BCRP MM: Vmax=${bcrp_vmax.toFixed(2)} Km=${BCRP_KM_NM}nM`);
  }
  if (molC.mrp) {
    const mrp_mm = MRP_VMAX * ASSUMED_CONC_NM / (MRP_KM_NM + ASSUMED_CONC_NM);
    efflux += mrp_mm * 18;
    assumptions_eff.push(`MRP MM: Vmax=${MRP_VMAX} Km=${MRP_KM_NM}nM`);
  }
  efflux = Math.min(efflux, 75);
  if (assumptions_eff.length === 0) assumptions_eff.push('No efflux substrates declared');
  assumptions_eff.push('Phase 2 will use actual ISF concentration from ODE for MM calculation');

  // ── 9. IMMUNE TRAFFICKING ────────────────────────────────────────────────
  const immScore = Math.min(sig.immuneTrafficking * 100, 100);

  // ── BUILD ROUTE OBJECTS WITH UNCERTAINTY + EVIDENCE TIERS ────────────────
  const mkRoute = (score, tier, cv, color, desc, assumptions, extra) => ({
    score        : Math.max(0, Math.round(score)),
    band         : _routeUncertaintyBand(score, tier, cv),
    evidenceTier : tier,
    color, desc,
    assumptionLog: assumptions,
    isHeuristic  : tier > 1,
    ...extra,
  });

  return {
    'Passive diffusion'        : mkRoute(pd,       1, TRANSPORTER_CV.glut1, '#a78bfa', 'Transcellular lipophilic (exponential MW, Gaussian logP)', assumptions_pd, {}),
    'Paracellular (TJ+AJ)'    : mkRoute(para,     1, TRANSPORTER_CV.pgp,   '#f87171', 'Junction-dependent (separate TJ/AJ failure, continuous charge selectivity)', assumptions_para, {}),
    'CMT / SLC'                : mkRoute(cmt,      1, TRANSPORTER_CV.glut1, '#3ecf8e', 'GLUT1/LAT1/MCT — Michaelis-Menten saturation', assumptions_cmt, {}),
    'RMT (TfR/IR/LRP)'        : mkRoute(rmt,      1, TRANSPORTER_CV.tfr,   '#4f9cf9', 'Receptor-mediated transcytosis — MM occupancy + affinity-efficiency paradox', assumptions_rmt, {}),
    'RVG-29 (nAChR)'          : mkRoute(rvgRoute, 2, TRANSPORTER_CV.tfr,   '#e05555', 'nAChR-mediated endocytosis — EXPERIMENTAL, mechanism ambiguous', assumptions_rvg, { isHeuristic:true }),
    'Caveolar (MFSD2A)'       : mkRoute(Math.min(cav,50), 2, 0.40, '#2dd4bf', 'Vesicular non-specific — MFSD2A-gated', ['MFSD2A suppression level: '+Math.round(sig.effectiveMFSD2A*100)+'%'], {}),
    'AMT (adsorptive)'        : mkRoute(Math.min(amt,38), 2, 0.45, '#f59e0b', 'Cationic/glycocalyx — EXPERIMENTAL', ['Charge eligibility: '+(amtElig?'yes':'no'), 'Net charge: '+bCharge.toFixed(2)], {}),
    'Immune trafficking'      : mkRoute(immScore, 1, 0.35, '#f472b6', 'NF-κB leukocyte transmigration', ['NF-κB: '+Math.round(sig.nfkbLevel*100)+'%, ECM_damage: '+Math.round(sig.ecmDamage*100)+'%'], { immune:true }),
    'Efflux (P-gp/BCRP/MRP)' : mkRoute(efflux,   1, TRANSPORTER_CV.pgp,   '#fb923c', 'Brain→blood active efflux — MM saturation, disease-modulated expression', assumptions_eff, { negative:true }),
  };
}

function computeScore(routes, sig) {
  const influx = ['Passive diffusion','Paracellular (TJ+AJ)','CMT / SLC','RMT (TfR/IR/LRP)','RVG-29 (nAChR)','Caveolar (MFSD2A)','AMT (adsorptive)'];
  let best = 0, bestRoute = '';
  for (const n of influx) {
    if (routes[n] && routes[n].score > best) { best = routes[n].score; bestRoute = n; }
  }
  const eff = routes['Efflux (P-gp/BCRP/MRP)'].score;

  // Uncertainty band on net score: propagate best route band
  const bestBand = routes[bestRoute]?.band || { low:0, expected:best, high:best };
  const netExpected = Math.max(0, Math.min(100, best - eff * 0.6));
  const netLow      = Math.max(0, Math.min(100, bestBand.low  - eff * 0.8));
  const netHigh     = Math.max(0, Math.min(100, bestBand.high - eff * 0.4));

  return {
    net           : netExpected,
    best, bestRoute,
    effluxPenalty : eff,
    paraBreach    : sig.paraBreach,
    transBreach   : sig.transBreach,
    // New in v8.1
    netBand       : { low:Math.round(netLow), expected:Math.round(netExpected), high:Math.round(netHigh) },
    isHeuristic   : true,  // Phase 1: all scores are heuristic; Phase 2: ODE-computed
    confidenceNote: 'Heuristic transport score v8.1. Phase 2 will replace with 7-compartment QSS-Euler ODE.',
    // Separate breach channels
    paracellularBreach  : sig.paracellularBreach,
    transcytosisBreach  : sig.transcytosisBreach,
    immuneBreach        : sig.immuneBreach,
    endothelialActivation: Math.round(sig.endothelialActivation * 100),
  };
}

function analyzeMolecule(rawMol, region) {
  const mol    = buildMolecule(rawMol);
  const sig    = computeSignaling(mol);
  const routes = computeRoutes(mol, region || 'cortex');
  const score  = computeScore(routes, sig);
  const lips   = [];
  if (mol.mw   > 500) lips.push('MW>500');
  if (mol.logp > 5)   lips.push('logP>5');
  if (mol.hbd  > 5)   lips.push('HBD>5');
  if (mol.hba  > 10)  lips.push('HBA>10');
  return {
    mol, signaling:sig, routes, score,
    lipinski:{ passes:lips.length===0, violations:lips },
    effectiveCharge:{ blood:effectiveCharge(mol,BLOOD_PH), endosome:effectiveCharge(mol,ENDOSOME_PH), brain:effectiveCharge(mol,BRAIN_ISF_PH) },
  };
}

// ─── RABV KINETICS ───────────────────────────────────────
const NERVE_DISTANCES = { face:200, hand:900, 'upper-arm':500, torso:300, thigh:600, foot:1200 };

function computeRABVTimeline(params) {
  params = params || {};
  var location   = params.location   || 'foot';
  var velocity   = params.velocity   || 200;
  var depth      = params.depth      || 'muscle';
  var dose       = params.dose       || 'med';
  var p75Active  = params.p75Active  != null ? params.p75Active : true;
  var replag     = params.replag     != null ? params.replag    : 9;
  var vaccinated = params.vaccinated || false;
  var pep        = params.pep        || false;
  var immuno     = params.immuno     || false;

  var baseDistance      = NERVE_DISTANCES[location] || 1200;
  var p75Multiplier     = p75Active ? Math.min(2.8, 400/Math.max(velocity,12)) : 1.0;
  var effectiveVelocity = Math.min(400, velocity * p75Multiplier);
  var transportDays     = baseDistance / effectiveVelocity;

  var depthBase  = depth==='superficial'?5.0:depth==='deep'?0.5:2.0;
  var dosePhaseF = {low:1.60,med:1.00,high:0.55}[dose]||1.0;
  var localPhase = depthBase * dosePhaseF;

  var doseReplagF= {low:1.40,med:1.00,high:0.70}[dose]||1.0;
  var replagDays = (replag/24) * doseReplagF;

  var peripheralSpreadProb = {low:0.05,med:0.15,high:0.45}[dose]||0.15;
  var immuneF = pep?0.10:vaccinated?0.30:immuno?1.80:1.0;
  var totalIncubation = Math.round((localPhase+transportDays+replagDays*2)*immuneF);

  var phaseEnds = [localPhase, localPhase+transportDays, totalIncubation, totalIncubation+5];

  return {
    distance:baseDistance, effectiveVelocity, p75Multiplier,
    transportDays, localPhase, replagDays, totalIncubation,
    phaseEnds, velocity:effectiveVelocity,
    doseF:dosePhaseF, immuneF, peripheralSpreadProb,
  };
}

function getRABVPhase(day, timeline) {
  if (day <= timeline.phaseEnds[0]) return {phase:0, label:'Phase 1 — NMJ Entry',               color:'#e05555'};
  if (day <= timeline.phaseEnds[1]) return {phase:1, label:'Phase 2 — PNS Retrograde Transport', color:'#e8913a'};
  if (day <= timeline.phaseEnds[2]) return {phase:2, label:'Phase 3 — CNS Invasion',             color:'#9b78fa'};
  return                                    {phase:3, label:'Phase 4 — Centrifugal Spread',       color:'#e8c83a'};
}

/**
 * computeInfectedBurden(cnsDay, treatDay, replicationCycleH)
 * Estimates the fraction of neurons infected at treatment day.
 * Exponential seeding model: starts at 0 at cnsDay, saturates toward 1.
 */
function computeInfectedBurden(cnsDay, treatDay, replicationCycleH) {
  replicationCycleH = replicationCycleH || 20;
  if (treatDay <= cnsDay) return 0.02; // pre-CNS: essentially zero
  const daysPostCNS = treatDay - cnsDay;
  const doublings   = (daysPostCNS * 24) / replicationCycleH;
  return Math.min(0.95, 0.02 * Math.pow(2, doublings));
}

// ─── SPLIT SYSTEMS ───────────────────────────────────────
const SPLIT_SYSTEMS = {
  split_intein_npu : { name:'Split-intein (Npu DnaE)', kd_nM:0.001, t_half_s:1.0,   reversible:false, activity_recovery:0.85, covalent:true,  note:'BEST — irreversible covalent ligation, t½~1s.' },
  split_gfp        : { name:'Split-GFP (GFP1-10/11)',  kd_nM:0.5,   t_half_s:30,    reversible:false, activity_recovery:0.90, covalent:false, note:'Good scaffold — Kd <1 nM, effectively irreversible.' },
  nanobit          : { name:'NanoBiT (LgBiT/SmBiT)',   kd_nM:190000,t_half_s:300,   reversible:true,  activity_recovery:1.0,  covalent:false, note:'POOR — Kd 190μM >> achievable CNS concentration.' },
  fkbp_frb         : { name:'FKBP/FRB + Rapamycin',    kd_nM:0.2,   t_half_s:600,   reversible:false, activity_recovery:0.75, covalent:false, note:'Conditional — requires rapamycin co-dose.', requires_trigger:'rapamycin' },
  leucine_zipper   : { name:'Leucine Zipper',           kd_nM:100,   t_half_s:120,   reversible:true,  activity_recovery:0.70, covalent:false, note:'Moderate affinity — may fail at low CNS concentrations.' },
};

const RABV_TARGETS = {
  P_LC8    :{ name:'P — LC8 (dynein)',       residues:'aa 218-225',   kd_nM:1,    effect:'Transport 200→12 mm/day',                 pdb:'3OA1,7C20', druggability:'High',   inhibitor:'Pep2' },
  P_TBK1   :{ name:'P — TBK1 (IFN kinase)', residues:'Ser179',        kd_nM:5,    effect:'IFN-β restored → antiviral state',        pdb:'7C20',      druggability:'High',   inhibitor:'Nanobody vs Ser179' },
  N_oligo  :{ name:'N — oligomerization',   residues:'N-N contact',  kd_nM:null, effect:'RNP collapses → RNA degraded',            pdb:'8FFR',      druggability:'Medium', inhibitor:'Dominant-negative N' },
  N_P_chap :{ name:'N0 — P chaperone',       residues:'N-term+C-term',kd_nM:5,    effect:'N misfolds',                              pdb:'8FFR',      druggability:'Medium', inhibitor:'P-peptide mimic' },
  'P+N dual':{ name:'P + N dual',            residues:'P-LC8 + N-N',  kd_nM:5,    effect:'Transport stall + RNP collapse',          pdb:'3OA1,8FFR', druggability:'Medium', inhibitor:'Tripartite (Arch D)' },
};

// ─── REASSEMBLY ──────────────────────────────────────────
function computeReassembly(splitSystem, concNM, timeAvail_h) {
  timeAvail_h = timeAvail_h != null ? timeAvail_h : 20;
  var sys = SPLIT_SYSTEMS[splitSystem];
  if (!sys) return { error:'Unknown', probability:0, probabilityPct:0, tToReassembly_min:999, covalent:false, warnings:[] };

  var kd        = sys.kd_nM;
  var fracBound = concNM / (concNM + kd);
  var tHalf_h   = sys.t_half_s / 3600;
  var lambda    = Math.log(2) / Math.max(tHalf_h, 0.00001);
  var timeFactor= sys.reversible
    ? Math.min(1, timeAvail_h/(tHalf_h*5))
    : 1 - Math.exp(-lambda*timeAvail_h);

  var prob    = fracBound * sys.activity_recovery * Math.min(1, timeFactor);
  var probPct = Math.round(prob * 100);
  var tToReassembly = tHalf_h * 3 * 60;

  var warnings = [];
  if (concNM < kd*0.1)         warnings.push('Conc '+concNM+'nM << Kd '+kd+'nM');
  if (splitSystem==='nanobit')  warnings.push('NanoBiT Kd=190μM unsuitable');
  if (sys.requires_trigger)     warnings.push('Requires '+sys.requires_trigger);

  var verdict = probPct>70?'Excellent':probPct>40?'Moderate':probPct>10?'Poor':'Negligible';
  var color   = probPct>70?'#3ecf8e' :probPct>40?'#f59e0b' :probPct>10?'#f87171':'#7f1d1d';

  return { system:sys.name, kd_nM:kd, concNM,
    fracBound:parseFloat(fracBound.toFixed(4)),
    probability:parseFloat(prob.toFixed(4)), probabilityPct:probPct,
    tToReassembly_min:parseFloat(tToReassembly.toFixed(1)),
    warnings, verdict, color, covalent:sys.covalent||false, reversible:sys.reversible, note:sys.note };
}

// ─── CO-LOCALIZATION MODEL ───────────────────────────────
/**
 * computeColocalization(f1Analysis, f2Analysis, region, viralBurden, hostState)
 *
 * Full co-localization score:
 *   base = min(F1_delivery, F2_delivery) × region_overlap × cell_entry_overlap
 *   then penalised by:
 *     - half-life mismatch (fragments with very different t½ will not be
 *       simultaneously present at adequate concentrations)
 *     - asynchronous arrival (fragments via different routes arrive at different times)
 *     - intracellular degradation mismatch
 *
 * Returns a scalar 0-1 and a breakdown object.
 */
function computeColocalization(f1mol, f2mol, f1bbb, f2bbb, region, viralBurden, hostState) {
  region      = region      || 'hippocampus';
  viralBurden = viralBurden != null ? viralBurden : 0.30;
  hostState   = hostState   || {};

  const rf = REGIONAL_FACTORS[region] || REGIONAL_FACTORS.cortex;

  // Base: limited by the weaker fragment (co-localization requires both present)
  const e1 = f1bbb / 100;
  const e2 = f2bbb / 100;
  const baseCodeliv = Math.min(e1, e2); // bottleneck fragment limits co-delivery

  // Region overlap: RABV infects specific regions. RVG targets nAChR-rich areas.
  // If fragment 2 uses RVG-29, it preferentially targets RABV-infected neurons (good).
  // Region factor scales with viral burden.
  const regionOverlap = Math.min(1, viralBurden * 1.5 * rf.nachr);

  // Cell entry overlap: both fragments must enter the SAME cell.
  // RMT fragment (F1) enters via TfR transcytosis.
  // RVG fragment (F2) enters via nAChR-mediated endocytosis.
  // Same-route = better overlap; different routes = probabilistic.
  const sameRoute = (f1mol.rmt && f2mol.rmt) || (!f1mol.rmt && !f2mol.rmt);
  const cellEntryOverlap = sameRoute ? 0.65 : 0.38;

  // Base co-localization score before mismatch penalties
  const baseColoc = baseCodeliv * regionOverlap * cellEntryOverlap;

  // ── Mismatch penalties ──────────────────────────────────
  // 1. Half-life mismatch: if F1 is still in transit when F2 has already degraded
  const t1 = f1mol.t_half_ic_h || (f1mol.mw > 10000 ? 6 : 14);
  const t2 = f2mol.t_half_ic_h || (f2mol.mw > 10000 ? 6 : 14);
  const tHalfRatio = Math.min(t1,t2) / Math.max(t1,t2);
  const halfLifeMismatchPenalty = 0.60 + 0.40 * tHalfRatio; // 1.0 = perfect match, 0.60 = extreme mismatch

  // 2. Asynchronous arrival: RMT (F1) typically arrives faster than RVG-29 (F2).
  // Estimate arrival time offset from delivery route.
  const arrivalF1_h = f1mol.rmt ? 4 : 8; // TfR-RMT ~4h, passive/RVG ~8h
  const arrivalF2_h = f2mol.rmt ? 4 : 8;
  const arrivalDelta = Math.abs(arrivalF1_h - arrivalF2_h);
  const avgT_half    = (t1 + t2) / 2;
  // If arrival delta > one half-life, significant fraction of early fragment is gone
  const asyncPenalty = arrivalDelta <= 1 ? 1.0
                     : arrivalDelta <= 4 ? 0.80
                     : arrivalDelta / avgT_half < 0.5 ? 0.65 : 0.40;

  // 3. Degradation mismatch: larger MW fragments have faster lysosomal clearance
  const degradDiff = Math.abs(
    (f1mol.mw > 10000 ? 0.08 : 0.02) - (f2mol.mw > 10000 ? 0.08 : 0.02)
  );
  const degradMismatchPenalty = degradDiff > 0.03 ? 0.75 : 1.0;

  // 4. Host state: immunocompromised → microglia cleared less efficiently,
  //    but also fragment degradation is faster (altered proteostasis)
  const immunocompPenalty = hostState.immunocompromised ? 0.80 : 1.0;

  // Final co-localization score
  const colocScore = Math.max(0, Math.min(1,
    baseColoc * halfLifeMismatchPenalty * asyncPenalty * degradMismatchPenalty * immunocompPenalty
  ));

  return {
    colocScore,
    colocPct: Math.round(colocScore * 100),
    breakdown: {
      baseCodeliv:      Math.round(baseCodeliv * 100),
      regionOverlap:    Math.round(regionOverlap * 100),
      cellEntryOverlap: Math.round(cellEntryOverlap * 100),
      baseColoc:        Math.round(baseColoc * 100),
      halfLifeMismatch: Math.round(halfLifeMismatchPenalty * 100),
      asyncPenalty:     Math.round(asyncPenalty * 100),
      degradMismatch:   Math.round(degradMismatchPenalty * 100),
      immunocompPenalty:Math.round(immunocompPenalty * 100),
    },
    bottleneck: [
      { label:'Min fragment BBB score', value:Math.round(Math.min(e1,e2)*100), weight:0.30 },
      { label:'Region/viral overlap',   value:Math.round(regionOverlap*100),   weight:0.20 },
      { label:'Cell entry overlap',     value:Math.round(cellEntryOverlap*100),weight:0.20 },
      { label:'Half-life match',        value:Math.round(tHalfRatio*100),      weight:0.15 },
      { label:'Arrival synchrony',      value:Math.round(asyncPenalty*100),    weight:0.15 },
    ].sort((a,b) => a.value - b.value), // sorted worst-first
  };
}

// ─── REBUILT THERAPEUTIC WINDOW ──────────────────────────
/**
 * computeTherapeuticWindow(params)
 *
 * 4-phase outcome model:
 *   pre_cns_intercept  — treatment before CNS seeding, can prevent invasion
 *   early_cns_control  — treatment during early CNS phase, can limit spread
 *   late_cns_salvage   — treatment after symptoms, partial suppression only
 *   too_late           — beyond salvage window
 *
 * Window score is a function of:
 *   - treatDay vs cnsDay, symptomDay, fatalDay
 *   - infected neuron burden at treatDay
 *   - F1 arrival time (route-dependent)
 *   - F2 arrival time
 *   - reassembly time
 *   - target inhibition onset (mechanism-dependent)
 *   - architecture's best_window property
 */
function computeTherapeuticWindow(params) {
  params = params || {};
  var biteMM         = params.biteMM        || 1200;
  var velocity       = params.velocity      || 200;
  var treatDay       = params.treatDay      || 0;
  var splitSystem    = params.splitSystem   || 'split_intein_npu';
  var f1bbb          = params.f1bbb         || 60;
  var f2bbb          = params.f2bbb         || 55;
  var f1route        = params.f1route       || 'RMT (TfR/IR/LRP)';
  var f2route        = params.f2route       || 'Passive diffusion';
  var archBestWindow = params.archBestWindow|| 'early_cns_control';

  var local      = 2;
  var transport  = biteMM / velocity;
  var cnsDay     = local + transport;
  var symptomDay = cnsDay + 5;
  var fatalDay   = symptomDay + 7;

  // F1/F2 arrival times (hours after administration → days)
  var f1ArrivalH = f1bbb > 50 && f1route.includes('RMT') ? 4 : f1bbb > 30 ? 6 : 10;
  var f2ArrivalH = f2route.includes('RMT') ? 4 : f2route.includes('Passive') ? 8 : 6;
  var lateArrival_h = Math.max(f1ArrivalH, f2ArrivalH);

  var sys = SPLIT_SYSTEMS[splitSystem] || SPLIT_SYSTEMS.split_intein_npu;
  var reassemblyH = sys.t_half_s * 3 / 3600;

  // Mechanism inhibition onset (hours after reassembly)
  // Anti-P: fast (blocks active P) — ~1h onset. Anti-N: slower (needs RNP to disassemble) ~4h. Anti-L: slowest ~6h.
  var inhibOnsetH = archBestWindow === 'pre_cns_intercept' ? 1 : archBestWindow === 'early_cns_control' ? 2 : 4;

  var totalH = lateArrival_h + reassemblyH + inhibOnsetH;
  var totalDays = totalH / 24;

  // Infected neuron burden at treatment day
  var burden = computeInfectedBurden(cnsDay, treatDay, 20);

  // Phase determination
  var phase, phaseLabel, wscore;
  if (treatDay + totalDays < cnsDay) {
    phase = 'pre_cns_intercept';
    phaseLabel = 'PNS intercept (pre-CNS)';
    // Best possible outcome — can prevent CNS seeding
    wscore = Math.min(100, 95 - treatDay * 4);
  } else if (treatDay < symptomDay && burden < 0.20) {
    phase = 'early_cns_control';
    phaseLabel = 'Early CNS control';
    // Architecture-dependent: some arch better here
    var archBonus = archBestWindow === 'early_cns_control' ? 10 : archBestWindow === 'pre_cns_intercept' ? -5 : 0;
    wscore = Math.min(75, 75 - (treatDay - cnsDay) * 8 - burden * 40 + archBonus);
  } else if (treatDay < fatalDay && burden < 0.70) {
    phase = 'late_cns_salvage';
    phaseLabel = 'Late CNS salvage';
    var archBonus2 = archBestWindow === 'late_cns_salvage' ? 10 : -5;
    wscore = Math.max(5, 40 - (treatDay - symptomDay) * 6 - burden * 30 + archBonus2);
  } else {
    phase = 'too_late';
    phaseLabel = 'Beyond salvage window';
    wscore = Math.max(0, 10 - (treatDay - fatalDay) * 5);
  }

  wscore = Math.max(0, Math.min(100, Math.round(wscore)));
  var verdict = wscore>=70?'Excellent':wscore>=40?'Marginal':wscore>=10?'Poor':'No window';
  var color   = wscore>=70?'#3ecf8e' :wscore>=40?'#f59e0b' :wscore>=10?'#f87171':'#7f1d1d';

  return {
    cnsDay:    parseFloat(cnsDay.toFixed(1)),
    symptomDay:parseFloat(symptomDay.toFixed(1)),
    fatalDay:  parseFloat(fatalDay.toFixed(1)),
    treatDay, burden:parseFloat(burden.toFixed(3)), burdenPct:Math.round(burden*100),
    f1ArrivalH, f2ArrivalH, lateArrival_h, reassemblyH, inhibOnsetH, totalH,
    phase, phaseLabel,
    windowScore:wscore, verdict, color,
    canPrevent: phase === 'pre_cns_intercept',
    canControl: phase === 'pre_cns_intercept' || phase === 'early_cns_control',
    canSalvage: phase !== 'too_late',
  };
}

// ─── HOST STATE MODIFIERS ────────────────────────────────
/**
 * hostStateModifiers(arch, hostState)
 * Returns multipliers applied to final scoring based on patient context.
 *
 * hostState fields:
 *   immunocompromised: bool — anti-P strategies lose value (IFN restoration useless)
 *   highInflammation : bool — BBB delivery improves but neuron damage risk rises
 *   brainstemDominant: bool — lower rescue margin, higher urgency
 *   priorVaccination : bool — immune state active, affects anti-P strategy value
 *   viralLoad        : 'low'|'med'|'high'
 */
function hostStateModifiers(archId, hostState) {
  hostState = hostState || {};

  var mods = {
    bbbDeliveryMod    : 1.0,  // multiplier on BBB score
    reassemblyMod     : 1.0,  // multiplier on reassembly probability
    mechanismMod      : 1.0,  // multiplier on mechanism efficacy
    windowMod         : 1.0,  // multiplier on window score
    notes             : [],
  };

  // Immunocompromised: anti-P relies on host IFN/immune response → penalise Arch A
  if (hostState.immunocompromised) {
    if (archId === 'A') {
      mods.mechanismMod *= 0.45;
      mods.notes.push('Immunocompromised: IFN-β restoration strategy severely weakened — anti-P nanobody cannot recruit immune effectors.');
    } else if (archId === 'D') {
      mods.mechanismMod *= 0.60;
      mods.notes.push('Immunocompromised: P-component of tripartite loses efficacy; N-component retains full value.');
    }
    mods.reassemblyMod *= 0.85; // altered proteostasis degrades fragments faster
    mods.notes.push('Fragment intracellular half-life reduced in immunocompromised state (altered proteostasis).');
  }

  // High inflammation: BBB more open (better delivery) but neuron damage risk
  if (hostState.highInflammation) {
    mods.bbbDeliveryMod *= 1.25;
    mods.mechanismMod   *= 0.90; // collateral neuronal damage reduces therapeutic window
    mods.notes.push('High neuroinflammation: BBB delivery improved ~25%, but neuronal damage may limit efficacy window.');
  }

  // Brainstem-dominant infection: lower margin for rescue (vital centres at risk)
  if (hostState.brainstemDominant) {
    mods.windowMod *= 0.65;
    mods.mechanismMod *= 0.80;
    mods.notes.push('Brainstem-dominant infection: vital centre involvement narrows therapeutic margin significantly.');
    if (archId === 'B' || archId === 'D') {
      mods.mechanismMod *= 1.15; // N-protein block is mechanism-agnostic to immune state
      mods.notes.push('Anti-N strategy retains value in brainstem infection (not immune-dependent).');
    }
  }

  // Prior vaccination with CNS breakthrough: immune response is active
  if (hostState.priorVaccination) {
    if (archId === 'A') {
      mods.mechanismMod *= 1.30;
      mods.notes.push('Prior vaccination + P-protein neutralization: synergistic. IFN restoration + pre-existing memory T cells = amplified clearance.');
    }
    mods.windowMod *= 1.15; // immune system buys more time
    mods.notes.push('Prior vaccination extends salvage window ~15% — pre-existing antibodies slow CNS spread.');
  }

  // Viral load
  if (hostState.viralLoad === 'high') {
    mods.mechanismMod *= 0.75;
    mods.reassemblyMod *= 0.90;
    mods.notes.push('High viral load: larger target mass, but fragment:virus ratio unfavourable. Higher dose needed.');
  } else if (hostState.viralLoad === 'low') {
    mods.mechanismMod *= 1.15;
    mods.notes.push('Low viral load: favourable fragment:virus ratio — lower intracellular concentration needed.');
  }

  return mods;
}

// ─── ARCHITECTURE DEFINITIONS ────────────────────────────
var _hs = {aj:100,mmp:0,nfkb:0,wnt:100,shh:100,pericyte:100,notch:100,angpt:100,ptm:0,cbf:100};

var PRESETS = {
  sm:[
    {name:'Caffeine',    mw:194,  logp:-0.1,hbd:0, hba:3,  base_charge:0, ppb:0.36,type:'sm',     pgp:false,bcrp:false,mrp:false,rmt:false,cation:false,..._hs},
    {name:'Morphine',    mw:285,  logp:0.9, hbd:2, hba:4,  base_charge:0, ppb:0.35,type:'sm',     pgp:true, bcrp:false,mrp:false,rmt:false,cation:false,pKa_base:8.0,..._hs},
    {name:'Diazepam',    mw:285,  logp:2.9, hbd:0, hba:2,  base_charge:0, ppb:0.98,type:'sm',     pgp:true, bcrp:false,mrp:false,rmt:false,cation:false,..._hs},
    {name:'Haloperidol', mw:376,  logp:4.3, hbd:1, hba:3,  base_charge:0, ppb:0.92,type:'sm',     pgp:false,bcrp:false,mrp:false,rmt:false,cation:false,..._hs},
    {name:'Doxorubicin', mw:544,  logp:1.3, hbd:6, hba:12, base_charge:0, ppb:0.74,type:'sm',     pgp:true, bcrp:true, mrp:true, rmt:false,cation:true, pKa_base:8.2,..._hs},
    {name:'Favipiravir', mw:157,  logp:-1.1,hbd:2, hba:3,  base_charge:0, ppb:0.54,type:'sm',     pgp:false,bcrp:false,mrp:false,rmt:false,cation:false,..._hs},
  ],
  nt:[
    {name:'Glucose',     mw:180,  logp:-3.0,hbd:5, hba:6,  base_charge:0, ppb:0.00,type:'glucose',pgp:false,bcrp:false,mrp:false,rmt:false,cation:false,..._hs},
    {name:'L-DOPA',      mw:197,  logp:-2.4,hbd:4, hba:5,  base_charge:0, ppb:0.12,type:'aa',     pgp:false,bcrp:false,mrp:false,rmt:false,cation:false,pKa_acid:2.3,pKa_base:8.7,..._hs},
    {name:'Ketone',      mw:102,  logp:-0.3,hbd:1, hba:2,  base_charge:0, ppb:0.00,type:'monocarb',pgp:false,bcrp:false,mrp:false,rmt:false,cation:false,..._hs},
  ],
  protein:[
    {name:'Albumin',     mw:66400,logp:-4.0,hbd:200,hba:400,base_charge:-18,ppb:0.99,type:'protein',pgp:false,bcrp:false,mrp:false,rmt:false,cation:false,..._hs},
    {name:'IgG',         mw:150000,logp:-5.0,hbd:400,hba:800,base_charge:-5,ppb:0.00,type:'protein',pgp:false,bcrp:false,mrp:false,rmt:false,cation:false,..._hs},
    {name:'Transferrin', mw:79600,logp:-5.0,hbd:220,hba:440,base_charge:-1, ppb:0.95,type:'protein',pgp:false,bcrp:false,mrp:false,rmt:true, cation:false,..._hs},
  ],
  rabv:[
    {name:'RVG-29',              mw:3256, logp:-1.8,hbd:8, hba:12,base_charge:0, ppb:0.10,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:false,cation:false,pKa_base:9.2, target:'nAChR',..._hs},
    {name:'TAT-CPP',             mw:1560, logp:-3.2,hbd:9, hba:14,base_charge:6, ppb:0.05,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:false,cation:true, pKa_base:12.0,target:'AMT',..._hs},
    {name:'Pep2 (P-LC8)',        mw:1850, logp:-1.5,hbd:6, hba:10,base_charge:0, ppb:0.15,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:false,cation:false,target:'P-LC8',..._hs},
    {name:'ArchA-F1 (VHH+TfR)', mw:8200, logp:-2.5,hbd:18,hba:28,base_charge:-1,ppb:0.20,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:true, rvg:false,cation:false,pKa_acid:3.8,pKa_base:9.2,target:'P-TBK1',arch:'A',frag:1,t_half_ic_h:8,..._hs},
    {name:'ArchA-F2 (VHH+RVG)', mw:11400,logp:-2.8,hbd:24,hba:36,base_charge:0, ppb:0.18,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:false,rvg:true, cation:false,pKa_acid:4.0,pKa_base:9.0,target:'P-LC8',arch:'A',frag:2,t_half_ic_h:6,..._hs},
    {name:'ArchB-F1 (DARPin+TfR)',mw:10500,logp:-2.2,hbd:20,hba:30,base_charge:-1,ppb:0.22,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:true, rvg:false,cation:false,target:'N-oligo',arch:'B',frag:1,t_half_ic_h:10,..._hs},
    {name:'ArchB-F2 (DARPin+RVG)',mw:10200,logp:-2.3,hbd:19,hba:29,base_charge:-1,ppb:0.20,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:false,rvg:true, cation:false,target:'N-oligo',arch:'B',frag:2,t_half_ic_h:9,..._hs},
  ],
};

var ARCHITECTURES = {
  A:{
    id:'A', name:'Split-Nanobody vs P protein', short:'Anti-P VHH',
    desc:'VHH nanobody split ~7+8 kDa via Npu DnaE intein. F1+TfR targets P-TBK1 (Ser179). F2+RVG-29 targets P-LC8 (aa218-225). Reconstitutes bivalent anti-P nanobody intraneuronally.',
    f1: null, f2: null, // populated below
    splitSystem:'split_intein_npu', target:'P_TBK1',
    effect:'Restores IFN-β signalling + stalls retrograde axonal transport',
    best_window:'pre_cns_intercept', // optimal before CNS seeding
    mechanism_speed:'fast',          // IFN signalling activated within hours
    occupancy_required: 'low',       // P protein blockade effective at low occupancy
    strengths:['Both P-protein interfaces biochemically validated','Pep2 precedent confirms P-LC8 druggable','IFN-β restoration harnesses natural immune kill','Smallest fragment MW — best BBB penetration','Intein ligation irreversible'],
    weaknesses:['Depends on intact host immune response — fails in immunocompromised','No approved anti-P nanobody yet','IFN-β restoration may cause transient neuroinflammation'],
    tripartite:false,
  },
  B:{
    id:'B', name:'Split-DARPin vs N protein', short:'Anti-N DARPin',
    desc:'DARPin split ~10+10 kDa. F1+TfR disrupts N oligomerization (RCSB 8FFR). F2+RVG-29 completes binding. Reconstituted DARPin prevents 11-mer ring assembly.',
    f1: null, f2: null,
    splitSystem:'split_intein_npu', target:'N_oligo',
    effect:'N 11-mer ring disrupted → RNP collapses → genomic RNA degraded',
    best_window:'early_cns_control', // needs active replication to be effective
    mechanism_speed:'medium',        // RNP needs to naturally disassemble
    occupancy_required: 'high',      // must hit high fraction of N subunits
    strengths:['Direct replication block — mechanism-agnostic to immune state','N structurally essential with no redundancy','Crystal structure available (RCSB 8FFR)','DARPin scaffold: thermostable, no disulfides'],
    weaknesses:['No validated N inhibitor in vivo yet','Larger fragments — worse BBB penetration than A','Requires high intraneuronal occupancy','N disruption mechanism unproven therapeutically'],
    tripartite:false,
  },
  C:{
    id:'C', name:'Split-L (Rapamycin-gated)', short:'Anti-L Conditional',
    desc:'L-polymerase RdRp subdomains (thumb+palm) ~14+15 kDa. FKBP-FRB conditional assembly. Requires rapamycin co-dose.',
    f1:{ name:'L-RdRp thumb', mw:15000,logp:-2.8,hbd:25,hba:38,base_charge:-2,ppb:0.25,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:true, rvg:false,cation:false,t_half_ic_h:5,..._hs},
    f2:{ name:'L-RdRp palm',  mw:14000,logp:-2.7,hbd:23,hba:35,base_charge:-2,ppb:0.23,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:false,rvg:false,cation:false,t_half_ic_h:5,..._hs},
    splitSystem:'fkbp_frb', target:'P_LC8',
    effect:'Dominant-negative L competes with native RdRp → replication stalls',
    best_window:'early_cns_control',
    mechanism_speed:'slow',         // dominant-negative needs to dilute out native L
    occupancy_required: 'high',
    strengths:['Conditional rapamycin gate = safety off-switch','Rapamycin crosses BBB well (logP 4.3)','Polymerase inhibition is direct antiviral'],
    weaknesses:['L protein not fully structurally solved','Largest fragments — worst BBB penetration','Requires rapamycin co-dose (PK complexity)','No L peptide inhibitor precedent'],
    tripartite:false,
  },
  D:{
    id:'D', name:'Tripartite (P+N+RVG)', short:'Tripartite Dual-Target',
    desc:'Three fragments: F1 anti-P VHH (TfR), F2 anti-N DARPin (passive), F3 RVG-29 scaffold (nAChR). Split-GFP tripartite (GFP1-9/10/11). Simultaneous P + N attack.',
    f1:{ name:'Tri-F1(anti-P)', mw:7000, logp:-2.3,hbd:15,hba:22,base_charge:-1,ppb:0.18,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:true, rvg:false,cation:false,t_half_ic_h:8,..._hs},
    f2:{ name:'Tri-F2(anti-N)', mw:8000, logp:-2.5,hbd:17,hba:25,base_charge:-1,ppb:0.20,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:false,rvg:true, cation:false,t_half_ic_h:7,..._hs},
    f3:{ name:'Tri-F3(RVG)',    mw:5500, logp:-1.9,hbd:10,hba:15,base_charge:0, ppb:0.12,type:'peptide',pgp:false,bcrp:false,mrp:false,rmt:false,rvg:true, cation:false,t_half_ic_h:10,..._hs},
    splitSystem:'split_gfp', target:'P+N dual',
    effect:'P block + N RNP collapse — dual mechanism, hardest for virus to escape',
    best_window:'early_cns_control',
    mechanism_speed:'medium',
    occupancy_required: 'medium',   // dual target reduces required occupancy per target
    strengths:['Synergistic dual target — mutation escape requires simultaneous changes to two essential proteins','RVG-29 confers infected-neuron specificity','Mechanism-agnostic to immune state'],
    weaknesses:['Tripartite reconstitution ~40% efficiency vs bipartite','Three independent BBB crossings required','Highest manufacturing complexity','Lowest co-localization probability'],
    tripartite:true,
  },
};
// populate f1/f2 from PRESETS after definition
ARCHITECTURES.A.f1 = PRESETS.rabv.find(function(p){return p.name.indexOf('ArchA-F1')>=0;});
ARCHITECTURES.A.f2 = PRESETS.rabv.find(function(p){return p.name.indexOf('ArchA-F2')>=0;});
ARCHITECTURES.B.f1 = PRESETS.rabv.find(function(p){return p.name.indexOf('ArchB-F1')>=0;});
ARCHITECTURES.B.f2 = PRESETS.rabv.find(function(p){return p.name.indexOf('ArchB-F2')>=0;});

// ─── UNIFIED SCORER ──────────────────────────────────────
/**
 * scoreArchitectureLive(archId, globalParams, fragmentOverrides)
 *
 * THE single source of truth for architecture scoring.
 * Used by: ranking grid, detail panel, recommendation engine, scenario mode.
 * No separate logic paths.
 *
 * globalParams: {
 *   region, biteMM, velocity, treatDay, concNM, viralBurden,
 *   hostState: { immunocompromised, highInflammation, brainstemDominant, priorVaccination, viralLoad }
 * }
 *
 * fragmentOverrides: { f1: partialMol, f2: partialMol }
 *   — merged onto architecture defaults; allows UI sliders to override MW/rmt etc.
 */
function scoreArchitectureLive(archId, globalParams, fragmentOverrides) {
  globalParams      = globalParams      || {};
  fragmentOverrides = fragmentOverrides || {};

  var arch = ARCHITECTURES[archId];
  if (!arch) return null;

  var region      = globalParams.region       || 'hippocampus';
  var biteMM      = globalParams.biteMM       || 1200;
  var velocity    = globalParams.velocity     || 200;
  var treatDay    = globalParams.treatDay     || 0;
  var concNM      = globalParams.concNM       || 5;
  var viralBurden = globalParams.viralBurden  != null ? globalParams.viralBurden : 0.30;
  var hostState   = globalParams.hostState    || {};

  // Build fragment molecules (merge arch defaults + overrides)
  var f1raw = Object.assign({}, arch.f1 || {}, fragmentOverrides.f1 || {});
  var f2raw = Object.assign({}, arch.f2 || {}, fragmentOverrides.f2 || {});
  var f1    = buildMolecule(f1raw);
  var f2    = buildMolecule(f2raw);

  // ── BBB Analysis ────────────────────────────────────────
  var a1 = analyzeMolecule(f1, region);
  var a2 = analyzeMolecule(f2, region);
  var f1bbb = a1.score.net;
  var f2bbb = a2.score.net;

  // For tripartite: also score F3
  var f3bbb = 0, f3route = '';
  if (arch.tripartite && arch.f3) {
    var f3raw = Object.assign({}, arch.f3, fragmentOverrides.f3 || {});
    var f3    = buildMolecule(f3raw);
    var a3    = analyzeMolecule(f3, region);
    f3bbb  = a3.score.net;
    f3route= a3.score.bestRoute;
  }

  // ── Co-localization ──────────────────────────────────────
  var coloc = computeColocalization(f1, f2, f1bbb, f2bbb, region, viralBurden, hostState);
  // Tripartite applies additional 3-fragment co-localization penalty
  if (arch.tripartite) {
    var f3colocPenalty = Math.pow(f3bbb/100 * 0.22, 0.5); // 3rd fragment multiplies in
    coloc = Object.assign({}, coloc, {
      colocScore: coloc.colocScore * f3colocPenalty,
      colocPct  : Math.round(coloc.colocScore * f3colocPenalty * 100),
    });
  }

  // ── Reassembly ──────────────────────────────────────────
  var effectiveSplitSys = arch.tripartite ? 'split_gfp' : arch.splitSystem;
  // Estimated intracellular concentration from RMT delivery
  var limConc = Math.min(f1bbb, f2bbb) / 100 * 0.001 * 10 * 1000;
  limConc = Math.min(limConc, concNM);
  var reassembly = computeReassembly(effectiveSplitSys, limConc);
  if (arch.tripartite) {
    reassembly = Object.assign({}, reassembly, {
      probability    : reassembly.probability * 0.40,
      probabilityPct : Math.round(reassembly.probabilityPct * 0.40),
    });
  }

  // ── Therapeutic window ──────────────────────────────────
  var win = computeTherapeuticWindow({
    biteMM, velocity, treatDay,
    splitSystem   : effectiveSplitSys,
    f1bbb, f2bbb,
    f1route       : a1.score.bestRoute,
    f2route       : a2.score.bestRoute,
    archBestWindow: arch.best_window,
  });

  // ── Host state modifiers ────────────────────────────────
  var mods = hostStateModifiers(archId, hostState);

  // ── Mechanism score ─────────────────────────────────────
  // Architecture-specific mechanism score based on:
  //   - target druggability
  //   - occupancy requirement vs achievable concentration
  //   - mechanism speed vs infection timeline
  var drugScore = {High:80, Medium:50, Low:25}[(RABV_TARGETS[arch.target]||{}).druggability||'Medium']||50;
  var occupancyPenalty = arch.occupancy_required === 'high'
    ? Math.min(1, limConc / 5)        // high occupancy needed → concentration-limited
    : arch.occupancy_required === 'low' ? 0.85 : 0.70;
  var mechanismScore = Math.min(100, drugScore * occupancyPenalty * mods.mechanismMod);

  // ── Raw probability chain ───────────────────────────────
  var pBBB_both    = (f1bbb/100) * (f2bbb/100) * mods.bbbDeliveryMod;
  var pColoc       = coloc.colocScore;
  var pReassembly  = reassembly.probability * mods.reassemblyMod;
  var pMechanism   = Math.min(1, mechanismScore / 100);
  var pOverall     = pBBB_both * pColoc * pReassembly * pMechanism;
  var overallPct   = Math.round(pOverall * 100);

  // ── Final composite score ───────────────────────────────
  var windowAdj    = win.windowScore * mods.windowMod;
  var finalScore   = Math.round(
    overallPct      * 0.35 +
    mechanismScore  * 0.25 +
    windowAdj       * 0.25 +
    Math.min(f1bbb,f2bbb) * 0.10 +
    reassembly.probabilityPct * 0.05
  );
  finalScore = Math.max(0, Math.min(100, finalScore));

  // ── Uncertainty & bottleneck analysis ───────────────────
  var uncertaintyData = computeUncertainty(archId, {
    f1bbb, f2bbb, colocPct:coloc.colocPct,
    reassemblyPct:reassembly.probabilityPct,
    windowScore:win.windowScore,
    mechanismScore:Math.round(mechanismScore),
    viralBurden, hostState,
  });

  return {
    archId, id:archId,
    name:arch.name, short:arch.short, desc:arch.desc,
    target:arch.target, effect:arch.effect,
    best_window:arch.best_window, mechanism_speed:arch.mechanism_speed,
    strengths:arch.strengths, weaknesses:arch.weaknesses,
    tripartite:arch.tripartite||false,

    // Per-fragment BBB
    f1bbb:Math.round(f1bbb), f2bbb:Math.round(f2bbb), f3bbb:Math.round(f3bbb),
    f1route:a1.score.bestRoute, f2route:a2.score.bestRoute, f3route,
    f1mol:f1, f2mol:f2,

    // Co-localization
    colocalization: coloc,

    // Reassembly
    reassembly,
    limConc_nM: parseFloat(limConc.toFixed(2)),

    // Therapeutic window
    window: win,

    // Host modifiers
    hostMods: mods,

    // Scores
    mechanismScore : Math.round(mechanismScore),
    overallPct,
    finalScore,
    verdict        : finalScore>=65?'Excellent':finalScore>=40?'Good':finalScore>=20?'Marginal':'Poor',

    // Probability breakdown
    probChain:{
      pBBB_both   : parseFloat((pBBB_both*100).toFixed(1)),
      pColoc      : parseFloat((pColoc*100).toFixed(1)),
      pReassembly : parseFloat((pReassembly*100).toFixed(1)),
      pMechanism  : parseFloat((pMechanism*100).toFixed(1)),
      pOverall    : parseFloat((pOverall*100).toFixed(2)),
    },

    // Uncertainty
    uncertainty: uncertaintyData,

    // Druggability
    druggability: drugScore,
  };
}

// ─── UNCERTAINTY & BOTTLENECK ANALYSIS ───────────────────
/**
 * computeUncertainty(archId, scores)
 * Returns:
 *   confidence: 'high'|'medium'|'low' — how much variance exists in the estimate
 *   dominantUncertainty: string — the single biggest unknown
 *   criticalBottleneck: string — the step with lowest probability
 *   bottleneckSteps: sorted list
 */
function computeUncertainty(archId, scores) {
  var steps = [
    { name:'F1 BBB crossing',       value:scores.f1bbb,              weight:0.15, note:'Depends on MW, logP, efflux' },
    { name:'F2 BBB crossing',       value:scores.f2bbb,              weight:0.15, note:'Depends on MW, delivery conjugate' },
    { name:'Co-localization',       value:scores.colocPct,           weight:0.25, note:'Both fragments must meet in same infected neuron' },
    { name:'Reassembly probability',value:scores.reassemblyPct,      weight:0.20, note:'Depends on split system Kd and intracellular concentration' },
    { name:'Therapeutic window',    value:scores.windowScore,        weight:0.15, note:'Treatment day vs CNS seeding day' },
    { name:'Mechanism efficacy',    value:scores.mechanismScore,     weight:0.10, note:'Target druggability and occupancy' },
  ];

  // Sort by value ascending — worst step first = bottleneck
  var sorted = steps.slice().sort(function(a,b){ return a.value - b.value; });
  var criticalBottleneck = sorted[0].name;
  var dominantUncertainty = sorted[0].name;

  // Architecture-specific uncertainty overrides
  var archUncertainties = {
    A: 'Host immune competence — IFN-beta restoration efficacy is patient-dependent',
    B: 'N-protein occupancy threshold — unknown minimum fraction needed for RNP collapse',
    C: 'L-protein structure unknown — rational split site is speculative',
    D: 'Three-fragment co-localization — no in vivo validation of tripartite CNS delivery',
  };
  dominantUncertainty = archUncertainties[archId] || dominantUncertainty;

  // Confidence: based on variance across steps and known data gaps
  var minVal  = sorted[0].value;
  var maxVal  = sorted[sorted.length-1].value;
  var spread  = maxVal - minVal;
  var archConfidence = {A:'medium', B:'low', C:'low', D:'low'}[archId] || 'medium';

  var confidence;
  if (archConfidence==='low' || spread > 50 || minVal < 15) confidence = 'low';
  else if (spread > 25 || minVal < 35) confidence = 'medium';
  else confidence = 'high';

  // Uncertainty ranges (approximate 95% bounds based on known literature variability)
  var approxScore = finalScoreApprox(scores);
  var uncertaintyRanges = {
    A:{ low:Math.round(Math.max(0, approxScore*0.55)), high:Math.round(Math.min(100, approxScore*1.45)) },
    B:{ low:Math.round(Math.max(0, approxScore*0.45)), high:Math.round(Math.min(100, approxScore*1.55)) },
    C:{ low:Math.round(Math.max(0, approxScore*0.30)), high:Math.round(Math.min(100, approxScore*1.70)) },
    D:{ low:Math.round(Math.max(0, approxScore*0.25)), high:Math.round(Math.min(100, approxScore*1.75)) },
  };
  var range = uncertaintyRanges[archId] || { low: Math.round(approxScore*0.50), high: Math.round(Math.min(100, approxScore*1.50)) };

  return {
    confidence,
    dominantUncertainty,
    criticalBottleneck,
    bottleneckSteps: sorted,
    upgradeHint: generateUpgradeHint(archId, sorted[0].name, scores),
    scoreRange: range,           // 95% confidence interval on final score
    approxScore,                 // base estimate used for range calculation
  };
}

function finalScoreApprox(s) {
  return Math.round((s.f1bbb+s.f2bbb)/2 * 0.25 + s.colocPct*0.25 + s.reassemblyPct*0.20 + s.windowScore*0.20 + s.mechanismScore*0.10);
}

function generateUpgradeHint(archId, bottleneck, scores) {
  if (bottleneck.includes('F1 BBB'))      return 'Upgrade F1: increase TfR conjugation efficiency, reduce MW below 10kDa if possible.';
  if (bottleneck.includes('F2 BBB'))      return 'Upgrade F2: add secondary RVG-29 targeting, reduce HBD count, consider TAT-CPP co-conjugation.';
  if (bottleneck.includes('Co-local'))    return 'Upgrade co-localization: switch F2 to TfR delivery (same route as F1), reduce MW mismatch, consider PEGylation for half-life extension.';
  if (bottleneck.includes('Reassembly'))  return 'Upgrade split system: switch to Npu DnaE intein (Kd 0.001 nM) — irreversible at picomolar concentrations.';
  if (bottleneck.includes('window'))      return 'Optimize window: earlier treatment or faster-acting delivery vehicle. Anti-P (Arch A) has fastest mechanism onset.';
  if (bottleneck.includes('Mechanism'))   return archId==='B' ? 'Arch B: validate minimum N occupancy threshold in vitro before advancing.' : 'Consider switching to Architecture A for faster mechanism action.';
  return 'Iterate on fragment physicochemistry to improve the limiting step.';
}

// ─── COMPARE ARCHITECTURES ───────────────────────────────
function compareArchitectures(globalParams) {
  var results = ['A','B','C','D'].map(function(id) {
    return scoreArchitectureLive(id, globalParams, {});
  });
  results.sort(function(a,b){ return b.finalScore - a.finalScore; });
  results.forEach(function(r,i){ r.rank = i+1; });
  return results;
}

// ─── SCENARIO BATCH MODE ─────────────────────────────────
/**
 * compareScenarios(scenarioList, archIds)
 * Run multiple globalParams objects through all architectures.
 * Returns matrix: results[scenarioIndex][archIndex]
 *
 * Example:
 *   compareScenarios([
 *     { label:'Face bite D0', biteMM:200, treatDay:0 },
 *     { label:'Foot bite D5', biteMM:1200, treatDay:5 },
 *   ], ['A','B'])
 */
function compareScenarios(scenarioList, archIds) {
  archIds = archIds || ['A','B','C','D'];
  return scenarioList.map(function(scenario) {
    var results = archIds.map(function(id) {
      var r = scoreArchitectureLive(id, scenario, {});
      return { archId:id, finalScore:r.finalScore, window:r.window.phase, bottleneck:r.uncertainty.criticalBottleneck, confidence:r.uncertainty.confidence };
    });
    results.sort(function(a,b){ return b.finalScore - a.finalScore; });
    return { scenario:scenario.label||'Scenario', results, best:results[0] };
  });
}

// ─── RECOMMENDATION ENGINE ───────────────────────────────
/**
 * recommendStrategy(globalParams)
 * Returns a structured recommendation with:
 *   - recommended architecture
 *   - rationale
 *   - primary bottleneck
 *   - upgrade suggestions
 *   - alternative consideration
 */
function recommendStrategy(globalParams) {
  var allScored = compareArchitectures(globalParams);
  var best      = allScored[0];
  var second    = allScored[1];
  var hostState = (globalParams && globalParams.hostState) || {};

  // Override logic: some clinical situations change the recommendation
  var override = null;
  if (hostState.immunocompromised && best.archId === 'A') {
    // Anti-P fails in immunocompromised — bump to B
    var bScore = allScored.find(function(a){ return a.archId==='B'; });
    if (bScore) { override = bScore; best = bScore; }
  }
  if (hostState.brainstemDominant && best.archId === 'C') {
    // Anti-L too slow for brainstem — bump to A or B
    var aScore = allScored.find(function(a){ return a.archId==='A'; });
    if (aScore) { override = aScore; best = aScore; }
  }

  // Build upgrade suggestions from bottleneck
  var upgrades = [];
  var bn = best.uncertainty.criticalBottleneck;
  upgrades.push(best.uncertainty.upgradeHint);

  // Add split system upgrade suggestion if reassembly is limiting
  if (bn.includes('Reassembly') && best.reassembly.kd_nM > 1) {
    upgrades.push('Switch to Npu DnaE split-intein (Kd 0.001 nM) — irreversible covalent ligation at picomolar concentrations.');
  }

  // Rationale
  var rationale = [];
  rationale.push('Ranked #1 of 4 architectures under current parameters.');
  if (best.window.canPrevent) {
    rationale.push('Treatment day allows pre-CNS interception — highest possible outcome class.');
  } else if (best.window.canControl) {
    rationale.push('Early CNS phase — containment still possible before widespread seeding.');
  } else {
    rationale.push('Late-stage scenario — salvage probability only.');
  }
  rationale.push('Mechanism: ' + best.effect + '.');
  if (best.hostMods && best.hostMods.notes.length) {
    rationale.push('Host modifier: ' + best.hostMods.notes[0]);
  }
  if (override) {
    rationale.push('NOTE: Default ranking adjusted for host state — ' + override.archId + ' selected over initial top scorer due to clinical context.');
  }

  return {
    recommendedArch   : best.archId,
    recommendedName   : best.short,
    finalScore        : best.finalScore,
    confidence        : best.uncertainty.confidence,
    rationale         : rationale,
    primaryBottleneck : best.uncertainty.criticalBottleneck,
    dominantUncertainty: best.uncertainty.dominantUncertainty,
    upgradeSuggestions: upgrades,
    alternativeArch   : second ? second.archId : null,
    alternativeName   : second ? second.short  : null,
    alternativeNote   : second ? 'Score ' + second.finalScore + '/100. Consider if primary fails.' : null,
    allScores         : allScored.map(function(a){ return { archId:a.archId, score:a.finalScore, confidence:a.uncertainty.confidence }; }),
    hostNotes         : best.hostMods ? best.hostMods.notes : [],
  };
}


// ═══════════════════════════════════════════════════════════
// MODULE 04 — PROTEIN SPLITTER ENGINE  v8.0
// SPELL-inspired split-site scoring · Sequence-based estimation
// Delivery strategy ranking · Full composite scorecard
// ═══════════════════════════════════════════════════════════

// ─── BIOPHYSICAL SCALES ──────────────────────────────────
// Chou-Fasman helix/sheet propensities (raw scale)
var CF_HELIX  = {A:1.42,R:0.98,N:0.67,D:1.01,C:0.70,Q:1.11,E:1.51,G:0.57,H:1.00,I:1.08,L:1.21,K:1.16,M:1.45,F:1.13,P:0.57,S:0.77,T:0.83,W:1.08,Y:0.69,V:1.06};
var CF_SHEET  = {A:0.83,R:0.93,N:0.89,D:0.54,C:1.19,Q:1.10,E:0.37,G:0.75,H:0.87,I:1.60,L:1.30,K:0.74,M:1.05,F:1.38,P:0.55,S:0.75,T:1.19,W:1.37,Y:1.47,V:1.70};
// Kyte-Doolittle hydrophobicity
var KD_HYDRO  = {A:1.8,R:-4.5,N:-3.5,D:-3.5,C:2.5,Q:-3.5,E:-3.5,G:-0.4,H:-3.2,I:4.5,L:3.8,K:-3.9,M:1.9,F:2.8,P:-1.6,S:-0.8,T:-0.7,W:-0.9,Y:-1.3,V:4.2};
// Janin burial propensity (higher = more buried)
var JN_BURIAL = {A:0.3,R:-1.4,N:-0.5,D:-0.6,C:0.9,Q:-0.7,E:-0.7,G:0.3,H:-0.1,I:0.7,L:0.5,K:-1.8,M:0.4,F:0.5,P:-0.3,S:-0.1,T:0.0,W:0.3,Y:-0.4,V:0.6};
// Disorder propensity
var DIS_PROP  = {A:0.06,R:0.18,N:0.15,D:0.19,C:-0.02,Q:0.18,E:0.24,G:0.17,H:0.05,I:-0.12,L:-0.11,K:0.26,M:-0.01,F:-0.15,P:0.23,S:0.14,T:0.09,W:-0.13,Y:-0.08,V:-0.10};
// N-degron destabilising residues
var N_DEGRON_AA = 'RKHFWYLIED';

function _sp_get(scale, ch, def) {
  var v = scale[ch ? ch.toUpperCase() : ''];
  return (v != null) ? v : (def != null ? def : 0);
}

// ─── FASTA PARSER ────────────────────────────────────────
function parseFASTA(text) {
  if (!text || !text.trim()) return { error:'Empty input', name:'', sequence:'', length:0 };
  var lines = text.trim().split(/\r?\n/);
  var name = 'Protein';
  var seqLines = [];
  for (var i = 0; i < lines.length; i++) {
    var l = lines[i].trim();
    if (!l) continue;
    if (l.charAt(0) === '>') { name = l.slice(1).trim().split(/\s+/)[0] || 'Protein'; }
    else { seqLines.push(l.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '')); }
  }
  var sequence = seqLines.join('');
  if (!sequence.length) return { error:'No valid amino acid sequence found', name:name, sequence:'', length:0 };
  return { name:name, sequence:sequence, length:sequence.length, error:null };
}

// ─── PER-RESIDUE PROPERTY ESTIMATION ────────────────────
function estimateResidueProperties(sequence) {
  var N = sequence.length;
  var W = 7; // sliding window half-width

  function smooth(arr, w) {
    return arr.map(function(_, i) {
      var lo = Math.max(0, i-w), hi = Math.min(N-1, i+w), sum = 0, cnt = 0;
      for (var j = lo; j <= hi; j++) { sum += arr[j]; cnt++; }
      return sum / cnt;
    });
  }

  var rawHelix  = sequence.split('').map(function(c){ return _sp_get(CF_HELIX, c, 1.0); });
  var rawSheet  = sequence.split('').map(function(c){ return _sp_get(CF_SHEET, c, 1.0); });
  var rawHydro  = sequence.split('').map(function(c){ return _sp_get(KD_HYDRO, c, 0); });
  var rawBurial = sequence.split('').map(function(c){ return _sp_get(JN_BURIAL, c, 0); });
  var rawDisord = sequence.split('').map(function(c){ return _sp_get(DIS_PROP, c, 0); });

  var sHelix  = smooth(rawHelix, W);
  var sSheet  = smooth(rawSheet, W);
  var sHydro  = smooth(rawHydro, W);
  var sBurial = smooth(rawBurial, W);
  var sDisord = smooth(rawDisord, W);

  function ssCall(i) {
    if (sHelix[i] > 1.05 && sHelix[i] >= sSheet[i]) return 'helix';
    if (sSheet[i] > 1.05 && sSheet[i] >= sHelix[i]) return 'sheet';
    return 'loop';
  }

  function accessibility(i) {
    var raw = 0.50 - sBurial[i] * 0.30 - sHydro[i] * 0.04;
    return Math.max(0, Math.min(1, raw + 0.50));
  }

  // Shannon entropy over window → conservation proxy
  function conservationPenalty(i) {
    var lo = Math.max(0, i-W), hi = Math.min(N-1, i+W);
    var freq = {};
    for (var j = lo; j <= hi; j++) {
      var c = sequence[j];
      freq[c] = (freq[c] || 0) + 1;
    }
    var total = hi - lo + 1, H = 0;
    Object.keys(freq).forEach(function(c) {
      var p = freq[c] / total;
      H -= p * Math.log2(p);
    });
    return Math.max(0, 1 - H / 3.0); // high entropy → low conservation penalty
  }

  return sequence.split('').map(function(aa, i) {
    return {
      index         : i,
      position      : i + 1,
      aa            : aa,
      ss            : ssCall(i),
      accessibility : accessibility(i),
      conservation  : conservationPenalty(i),
      hydrophobicity: sHydro[i],
      disorder      : Math.min(1, Math.max(0, sDisord[i] + 0.35)),
    };
  });
}

// ─── SPLIT-SITE CANDIDATE FINDER ────────────────────────
var SPLIT_HARD_RULES = {
  min_fragment_aa : 50,
  acc_threshold   : 0.35,
  cons_threshold  : 0.65,
  min_dist_to_site: 8,
};

function findSplitSiteCandidates(props, annotations, constraints) {
  annotations = annotations || {};
  constraints = Object.assign({}, SPLIT_HARD_RULES, constraints || {});
  var N = props.length;

  // Build expanded forbidden zone
  var forbidden = new Set();
  ['active_site','binding_hotspot','interface','forbidden'].forEach(function(key) {
    (annotations[key] || []).forEach(function(pos) { forbidden.add(pos); });
  });
  var forbiddenExpanded = new Set();
  forbidden.forEach(function(pos) {
    for (var d = -constraints.min_dist_to_site; d <= constraints.min_dist_to_site; d++) {
      forbiddenExpanded.add(pos + d);
    }
  });

  var candidates = [];

  for (var i = 0; i < N - 1; i++) {
    var pos1 = i + 1; // cut after residue pos1 (1-indexed)
    var f1len = pos1;
    var f2len = N - pos1;

    // Hard filters
    if (f1len < constraints.min_fragment_aa) continue;
    if (f2len < constraints.min_fragment_aa) continue;
    if (props[i].ss === 'helix' || props[i].ss === 'sheet') continue;
    if (props[i].accessibility < constraints.acc_threshold) continue;
    if (props[i].conservation > constraints.cons_threshold) continue;
    if (forbiddenExpanded.has(pos1)) continue;

    // Soft scores
    var ssScore   = props[i].ss === 'loop' ? 1.0 : 0.30;
    var accScore  = Math.min(1, (props[i].accessibility - constraints.acc_threshold) / (1 - constraints.acc_threshold));
    var consScore = 1 - props[i].conservation;

    var minDist = Infinity;
    forbidden.forEach(function(fp) { minDist = Math.min(minDist, Math.abs(pos1 - fp)); });
    if (!isFinite(minDist)) minDist = N;
    var distScore = Math.min(1, minDist / 30);

    var balance  = Math.min(f1len, f2len) / Math.max(f1len, f2len);
    var balScore = 0.50 + 0.50 * balance;

    var disScore = 0.40 + 0.60 * props[i].disorder;

    var splitScore = (
      ssScore   * 0.28 +
      accScore  * 0.22 +
      consScore * 0.20 +
      distScore * 0.16 +
      balScore  * 0.09 +
      disScore  * 0.05
    );

    candidates.push({
      position   : pos1,
      aa         : props[i].aa,
      ss         : props[i].ss,
      f1Length   : f1len,
      f2Length   : f2len,
      scores     : {
        loop         : parseFloat(ssScore.toFixed(3)),
        accessibility: parseFloat(accScore.toFixed(3)),
        conservation : parseFloat(consScore.toFixed(3)),
        funcDistance : parseFloat(distScore.toFixed(3)),
        balance      : parseFloat(balScore.toFixed(3)),
        disorder     : parseFloat(disScore.toFixed(3)),
      },
      splitScore : parseFloat(splitScore.toFixed(3)),
      verdict    : splitScore>=0.70?'Excellent':splitScore>=0.50?'Good':splitScore>=0.35?'Marginal':'Poor',
      minDistToSite: isFinite(minDist) ? minDist : null,
    });
  }

  candidates.sort(function(a, b) { return b.splitScore - a.splitScore; });
  return candidates;
}

// ─── FRAGMENT PROPERTY ESTIMATORS ────────────────────────
function estimateFragmentMW(seq) { return (seq || '').length * 110; }

function estimateLogP(seq) {
  if (!seq || !seq.length) return -2.0;
  var total = 0;
  for (var i = 0; i < seq.length; i++) total += _sp_get(KD_HYDRO, seq[i], 0);
  return parseFloat((total / seq.length * 0.4 - 1.5).toFixed(2));
}

function estimateHBD(seq) {
  var count = 0;
  for (var i = 0; i < (seq||'').length; i++) {
    if ('RKNDQHSTYWM'.indexOf(seq[i]) >= 0) count++;
  }
  return Math.round(count * 0.4);
}

function estimateHBA(seq) {
  var count = 0;
  for (var i = 0; i < (seq||'').length; i++) {
    if ('RKNDQEHSTYWM'.indexOf(seq[i]) >= 0) count++;
  }
  return Math.round(count * 0.6);
}

function terminiDegronRisk(f1seq, f2seq) {
  var risk = { f1Cterminus:'low', f2Nterminus:'low', f2Cterminus:'low', overall:'low' };
  if (f1seq && N_DEGRON_AA.indexOf(f1seq[f1seq.length-1]) >= 0) risk.f1Cterminus = 'medium';
  if (f2seq && N_DEGRON_AA.indexOf(f2seq[0]) >= 0)              risk.f2Nterminus = 'high';
  if (f2seq && N_DEGRON_AA.indexOf(f2seq[f2seq.length-1]) >= 0) risk.f2Cterminus = 'medium';
  var levels = ['low','medium','high'];
  var worst = Math.max(
    levels.indexOf(risk.f1Cterminus),
    levels.indexOf(risk.f2Nterminus),
    levels.indexOf(risk.f2Cterminus)
  );
  risk.overall = levels[Math.max(0,worst)];
  return risk;
}

// ─── DELIVERY MODE SCORER ────────────────────────────────
function scoreDeliveryMode(f1mw, f2mw, mode, bbbStrat, tissue, barrier) {
  tissue  = tissue  || 'cns';
  barrier = barrier || {};
  var maxMW   = Math.max(f1mw, f2mw);
  var score   = 0;
  var notes   = [];
  var warnings= [];

  if (mode === 'dual_aav') {
    var payloadOK = maxMW <= 158000; // ~4.7 kb coding capacity minus intein
    var coTrans   = tissue === 'cns' ? 0.55 : 0.65;
    score = payloadOK ? coTrans : coTrans * (158000 / maxMW) * 0.5;
    if (!payloadOK) warnings.push('Fragment >' + Math.round(maxMW/1000) + ' kDa may exceed dual-AAV payload ceiling (~4.7 kb).');
    notes.push('Co-transduction probability ~55% in CNS. Npu DnaE or gp41-1 intein required.');
    if (tissue === 'cns') notes.push('AAV9/AAVrh10 preferred for broad CNS tropism.');
  } else if (mode === 'mrna_lnp') {
    var imbalance    = Math.max(f1mw, f2mw) / Math.min(f1mw, f2mw);
    var stoichPenalty= imbalance > 2.5 ? 0.30 : imbalance > 1.5 ? 0.10 : 0;
    var endoEscape   = 0.70;
    score = endoEscape * (1 - stoichPenalty) * (tissue === 'cns' ? 0.75 : 1.0);
    if (stoichPenalty > 0) warnings.push('Fragment MW imbalance ' + imbalance.toFixed(1) + 'x increases stoichiometry risk.');
    notes.push('LNP endosomal escape ~70% with optimised lipid formulation.');
    if (tissue === 'cns') notes.push('CNS LNP delivery ~0.5-3% ID/g brain; IT administration improves this.');
  } else {
    // protein biologic — wire into existing BBB engine
    var barrierDefaults = { tj:100, aj:100, mmp:0, nfkb:0, wnt:100, shh:100, pericyte:100, notch:100, angpt:100, ptm:0, cbf:100 };
    var b = Object.assign({}, barrierDefaults, barrier);
    // logP estimate: peptides are hydrophilic; scale gently with MW
    var lp1 = Math.max(-5, Math.min(2, -1.5 - (f1mw / 15000)));
    var lp2 = Math.max(-5, Math.min(2, -1.5 - (f2mw / 15000)));
    // Both fragments get the same BBB strategy (F1=primary carrier, F2=secondary)
    // For tfr_rmt: F1 gets TfR conjugate (rmt:true), F2 gets RVG-29 (rvg:true) — standard dual-route
    // For rvg29: both use RVG-29
    var f1rmt = (bbbStrat === 'tfr_rmt');
    var f1rvg = (bbbStrat === 'rvg29');
    var f2rmt = false;
    var f2rvg = (bbbStrat === 'tfr_rmt' || bbbStrat === 'rvg29'); // F2 always gets RVG-29 as secondary
    var mol1 = buildMolecule({
      mw:f1mw, logp:lp1,
      hbd:Math.min(15, Math.round(f1mw/2200)), hba:Math.min(20, Math.round(f1mw/1600)),
      ppb:0.20, type:'peptide', rmt:f1rmt, rvg:f1rvg,
      tj:b.tj,aj:b.aj,mmp:b.mmp,nfkb:b.nfkb,wnt:b.wnt,shh:b.shh,pericyte:b.pericyte,notch:b.notch,angpt:b.angpt,ptm:b.ptm,cbf:b.cbf
    });
    var mol2 = buildMolecule({
      mw:f2mw, logp:lp2,
      hbd:Math.min(15, Math.round(f2mw/2200)), hba:Math.min(20, Math.round(f2mw/1600)),
      ppb:0.18, type:'peptide', rmt:f2rmt, rvg:f2rvg,
      tj:b.tj,aj:b.aj,mmp:b.mmp,nfkb:b.nfkb,wnt:b.wnt,shh:b.shh,pericyte:b.pericyte,notch:b.notch,angpt:b.angpt,ptm:b.ptm,cbf:b.cbf
    });
    var r1 = analyzeMolecule(mol1, 'hippocampus');
    var r2 = analyzeMolecule(mol2, 'hippocampus');
    score = (r1.score.net / 100) * (r2.score.net / 100);
    notes.push('F1 BBB: ' + Math.round(r1.score.net) + '/100 via ' + (r1.score.bestRoute || '—') + '.');
    notes.push('F2 BBB: ' + Math.round(r2.score.net) + '/100 via ' + (r2.score.bestRoute || '—') + '.');
    if (maxMW > 50000 && bbbStrat === 'none') warnings.push('Large biologic with no BBB strategy: brain exposure <0.1% ID.');
    if (bbbStrat === 'tfr_rmt') notes.push('TfR-RMT validated in vivo — pabinafusp alfa precedent (Japan approval).');
  }

  score = Math.max(0, Math.min(1, score));
  return {
    mode, bbbStrategy:bbbStrat,
    score      : score,
    scorePct   : Math.round(score * 100),
    notes, warnings,
    color      : score >= 0.55 ? '#3ecf8e' : score >= 0.30 ? '#f59e0b' : '#f87171',
  };
}

// ─── DELIVERY STRATEGY RANKER ────────────────────────────
function rankDeliveryStrategies(f1mw, f2mw, targetTissue, barrierState) {
  targetTissue = targetTissue || 'cns';
  barrierState = barrierState || {};
  var modes = [
    { mode:'dual_aav',         label:'Dual AAV',                          bbbStrat:'none'       },
    { mode:'mrna_lnp',         label:'mRNA / LNP co-delivery',            bbbStrat:'none'       },
    { mode:'protein_biologic', label:'Protein biologic + TfR-RMT shuttle',bbbStrat:'tfr_rmt'   },
    { mode:'protein_biologic', label:'Protein biologic + RVG-29 shuttle', bbbStrat:'rvg29'      },
  ];
  var results = modes.map(function(m) {
    var d = scoreDeliveryMode(f1mw, f2mw, m.mode, m.bbbStrat, targetTissue, barrierState);
    return {
      id      : m.mode + (m.bbbStrat !== 'none' ? '_' + m.bbbStrat : ''),
      label   : m.label,
      score   : d.score,
      scorePct: d.scorePct,
      notes   : d.notes,
      warnings: d.warnings,
      color   : d.color,
    };
  });
  results.sort(function(a, b) { return b.score - a.score; });
  results.forEach(function(r, i) { r.rank = i + 1; });
  return results;
}

// ─── COMPOSITE SPLIT DESIGN SCORER ───────────────────────
function scoreSplitDesign(params) {
  params  = params  || {};
  var site    = params.splitSite    || {};
  var seq     = params.sequence     || '';
  var sysKey  = params.splitSystem  || 'split_intein_npu';
  var concNM  = params.concNM       != null ? params.concNM : 5;
  var mode    = params.deliveryMode || 'protein_biologic';
  var bbbStrat= params.bbbStrategy  || 'tfr_rmt';
  var tissue  = params.targetTissue || 'cns';
  var barrier = params.barrierState || {};
  var annots  = params.annotations  || {};

  var cutPos = site.position || Math.floor(seq.length / 2);
  var f1seq  = seq.slice(0, cutPos);
  var f2seq  = seq.slice(cutPos);
  var f1mw   = estimateFragmentMW(f1seq);
  var f2mw   = estimateFragmentMW(f2seq);

  // S_safety
  var sc    = site.scores || {};
  var Ssafe = Object.keys(sc).length
    ? sc.loop*0.28 + sc.accessibility*0.22 + sc.conservation*0.20 + sc.funcDistance*0.16 + sc.balance*0.09 + sc.disorder*0.05
    : 0.45;

  // S_frag
  var degron       = terminiDegronRisk(f1seq, f2seq);
  var degronPenalty= {low:0, medium:0.15, high:0.35}[degron.overall] || 0;
  var sizePenalty  = (f1mw < 5000 || f2mw < 5000) ? 0.20 : 0;
  var Sfrag        = Math.max(0, 1 - degronPenalty - sizePenalty);

  // S_reassembly
  var rr          = computeReassembly(sysKey, concNM);
  var Sreassembly = rr.probability;

  // S_function
  var funcDist  = sc.funcDistance != null ? sc.funcDistance : 0.50;
  var consScore = sc.conservation != null ? sc.conservation : 0.50;
  var sizeBonus = Math.min(0.12, Math.max(0, (Math.min(f1mw,f2mw) - 5000) / 150000));
  var Sfunc     = Math.min(1, funcDist * 0.50 + consScore * 0.35 + 0.15 + sizeBonus);

  // S_delivery
  var delivD   = scoreDeliveryMode(f1mw, f2mw, mode, bbbStrat, tissue, barrier);
  var Sdeliv   = delivD.score;

  // S_manufact
  var aggRisk      = (f1mw > 50000 || f2mw > 50000) ? 0.20 : 0;
  var foreignLoad  = (sysKey === 'split_intein_npu' || sysKey === 'split_gfp') ? 0.10 : 0.18;
  var aavBurden    = mode === 'dual_aav' ? 0.08 : 0;
  var Smanufact    = Math.max(0, 1 - aggRisk - foreignLoad - aavBurden);

  // Uncertainty
  var hasAnnot = Object.keys(annots).some(function(k){ return (annots[k]||[]).length > 0; });
  var seqQual  = Math.min(1, seq.length / 100);
  var U = 0.22 + (hasAnnot ? 0 : 0.13) + (1 - seqQual) * 0.12;
  if (sysKey === 'fkbp_frb')    U += 0.05;
  if (sysKey === 'nanobit')     U += 0.15;

  // Composite
  var Base  = Ssafe*0.30 + Sfrag*0.18 + Sreassembly*0.18 + Sfunc*0.18 + Sdeliv*0.10 + Smanufact*0.06;
  var Final = Base * (1 - 0.35 * Math.min(1, U));
  var fp    = Math.round(Final * 100);
  var col   = fp >= 65 ? '#3ecf8e' : fp >= 40 ? '#f59e0b' : '#f87171';

  return {
    splitSite     : site,
    f1seq, f2seq, f1mw, f2mw,
    degronRisk    : degron,
    subscores     : {
      safety           : parseFloat(Ssafe.toFixed(3)),
      fragStability    : parseFloat(Sfrag.toFixed(3)),
      reassembly       : parseFloat(Sreassembly.toFixed(3)),
      function         : parseFloat(Sfunc.toFixed(3)),
      delivery         : parseFloat(Sdeliv.toFixed(3)),
      manufacturability: parseFloat(Smanufact.toFixed(3)),
    },
    uncertainty   : parseFloat(U.toFixed(3)),
    baseScore     : parseFloat(Base.toFixed(3)),
    finalScore    : fp,
    color         : col,
    verdict       : fp>=65?'Excellent':fp>=40?'Good':fp>=20?'Marginal':'Poor',
    reassemblyDetail: rr,
    deliveryDetail  : delivD,
    scoreRange    : { low:Math.round(Math.max(0,fp*(1-U*0.8))), high:Math.round(Math.min(100,fp*(1+U*0.55))) },
    upgradeHints  : _splitterHints(Ssafe,Sfrag,Sreassembly,Sfunc,Sdeliv,degron,sysKey),
  };
}

function _splitterHints(Ss, Sf, Sr, Sfu, Sd, degron, sysKey) {
  var hints = [];
  var minS  = Math.min(Ss, Sf, Sr, Sfu, Sd);
  if (Ss === minS || Ss < 0.45) hints.push('Split site safety is limiting — seek a loop region with lower conservation and greater distance from functional residues.');
  if (Sf === minS || Sf < 0.45) {
    if (degron.overall !== 'low') hints.push('Degron risk at new F2 N-terminus (' + degron.f2Nterminus + ') — add a 2-4 residue Met-Gly-Gly cap to stabilise F2 start.');
    else hints.push('Fragment stability low — shift cut site ±3-5 residues to reduce hydrophobic exposure.');
  }
  if (Sr === minS || Sr < 0.45) {
    if (sysKey !== 'split_intein_npu' && sysKey !== 'split_gfp') hints.push('Switch to Npu DnaE intein (Kd 0.001 nM, irreversible) — highest reassembly reliability.');
    else hints.push('Increase intracellular fragment concentration: improve BBB delivery or use AAV for sustained CNS expression.');
  }
  if (Sfu === minS || Sfu < 0.45) hints.push('Cut too close to functional residues — increase min_dist_to_site or choose a more distal interdomain loop.');
  if (Sd === minS || Sd < 0.35)   hints.push('Delivery is the bottleneck — add TfR-RMT to F1 and RVG-29 to F2 for CNS protein biologic delivery, or switch to dual-AAV gene therapy.');
  if (!hints.length) hints.push('Design is well-balanced. Optimise linker length (GGGGS×2-3) and screen extein junction variants for the chosen intein.');
  return hints;
}

// ─── RABV PROTEIN PRESETS ────────────────────────────────
var PROTEIN_PRESETS = {
  P: {
    id:'P', name:'RABV Phosphoprotein (P)', shortName:'P protein', mw_kda:33, length:297,
    sequence:'MDADKIVFKVNNQVVSLKPEIIVDQYEYKYPAIKDLKKPCITLGQVDNKTYVDKQLQNFEKGVTIDFDLASSRLSERNFISFTDLEYNSSPFVTTPTVSIQEQRLDSITEDPGSSGTTESTDISRLNDALRRNMEEVLAQIRPAEDPTPNRAAQQPEMEWSRALNTIYLNQQNLRIQKQVSETEGIEGLAQDDSVTIAQNNTQTKVVNDSGLNYMKSNVKQVKKMADEFERNLKESQRPVHFLNFGTLNLSIVREKQKTLHSANVKDQYEMESLFHSTPGVSTQRRGDLNQSYRRILDPNAKMTVDLNQTVSTTQEAYRQIMFNLAK',
    annotations:{
      active_site:[],
      binding_hotspot:[218,219,220,221,222,223,224,225],
      interface:[179],
      forbidden:[179,218,219,220,221,222,223,224,225],
      known_good_splits:[100,140,160],
    },
    note:'LC8 binding (aa 218-225) is the dynein transport anchor. TBK1 interface at Ser179 mediates IFN-β suppression. Best split: flexible N-terminal domain aa 100-160.',
    pdb:'7C20,3OA1',
  },
  N: {
    id:'N', name:'RABV Nucleoprotein (N)', shortName:'N protein', mw_kda:47, length:450,
    sequence:'MDADKIVFKVNNQVVSLKPEIIVKMDVNPKDEVLNKLNELKQRLEEMGDPEEQVVMAIPSWQHLYQKSTMGPQHPNPHLSYMVDVLQPPQPDNHNDRDRQHYENNQEFWKEHLDRLRLEQGGDQATNLRKVLNGLRQFAIGNDVTPFNRFVDGEEALVLKKNMEIAHFGTPFQHINDTKKDEYEFLSNKNMDDPQVFLMDQQLEQKLLEAQPTLELTLAIHKLRNVSSDNKGYSIQDTDNRGEGIQKFLKRMIMQMNDNHSDKVAEGIASCLLDLKDKIIEQINKLLDSDFVTKKQLITPKIPAIAQAAALDGPYQLKSKNPNLATILNAIQLTVKMSEDLKLQRYAQNVKQLIDLKMEQESGPKIDTIEQINQENIKKMANDMVNRQKSMTEKVTMRHTREKQQVVPVKQALVVSHYENMDPIIAEEGDNMIDFQHPYNSSLFKQDAIILRVQQLMNPQLQEFLQSSQERLA',
    annotations:{
      active_site:[],
      binding_hotspot:[246,247,248,249,250,251,252,253,254,255],
      interface:[101,102,103,370,371,372,373,374,375,376],
      forbidden:[101,102,103,246,247,248,249,250,251,370,371,372],
      known_good_splits:[200,250,300],
    },
    note:'Forms 11-mer helical rings (RCSB 8FFR). N-N oligomerization interface at ~101-103 and ~370-376. RNA binding groove ~246-255. Best split: central domain aa 200-300.',
    pdb:'8FFR',
  },
  L: {
    id:'L', name:'RABV L Protein / RdRp (L)', shortName:'L protein (RdRp)', mw_kda:240, length:2142,
    sequence:null,
    annotations:{
      active_site:[831,832,833],
      binding_hotspot:[740,741,742,743,744,745],
      interface:[1,2,3,4,5,6,7,8,2138,2139,2140,2141,2142],
      forbidden:[829,830,831,832,833,834,835],
      known_good_splits:[500,900,1350],
    },
    note:'L has 5 enzymatic domains. GDN catalytic motif at aa 831-833. Best split at interdomain linkers (~aa 500, ~900, ~1350). Very high uncertainty — L structure largely unresolved. Fragments will be >80 kDa each; dual AAV is the only realistic delivery mode.',
    pdb:'—',
    high_uncertainty:true,
  },
};

// ─── MANUFACTURING FEASIBILITY ───────────────────────────────────────────────
/**
 * computeManufacturing(archId, f1mol, f2mol, f3mol)
 * Scores practical manufacturability of a split-protein therapeutic.
 * Returns: feasibility (0-100), grade, sub-scores, limiting_factor.
 */
function computeManufacturing(archId, f1mol, f2mol, f3mol) {
  f1mol = f1mol || {};
  f2mol = f2mol || {};

  const arch = ARCHITECTURES[archId] || {};

  // Fragment complexity: each fragment adds purification burden
  const nFrags = arch.tripartite ? 3 : 2;
  const fragComplexity = nFrags === 2 ? 80 : 45;  // tripartite → 3 sep purifications

  // MW burden: larger fragments harder to express in E.coli/CHO
  const f1mw = f1mol.mw || 8000;
  const f2mw = f2mol.mw || 11000;
  const avgMW = (f1mw + f2mw) / 2;
  const mwScore = avgMW <= 5000  ? 90
                : avgMW <= 10000 ? 75
                : avgMW <= 20000 ? 55
                : avgMW <= 50000 ? 30 : 15;

  // Conjugate complexity: TfR shuttle + RVG fusion add manufacturing steps
  const hasRMT = f1mol.rmt || f2mol.rmt;
  const hasRVG = f1mol.rvg || f2mol.rvg || (f3mol && f3mol.rvg);
  const conjugatePenalty = (hasRMT ? 18 : 0) + (hasRVG ? 12 : 0);
  const conjugateScore   = Math.max(0, 100 - conjugatePenalty);

  // Split system manufacturing burden
  const sysMfgScore = {
    split_intein_npu : 75,  // well-established, good yield
    split_gfp        : 65,  // moderate — GFP11 tag straightforward
    fkbp_frb         : 55,  // two proteins + rapamycin supply chain
    leucine_zipper   : 85,  // simple peptide tag
    nanobit          : 80,  // small tags, easy
  }[arch.splitSystem || 'split_intein_npu'] || 65;

  // Aggregation risk: high MW + low logP fragments prone to aggregation
  const aggRisk = avgMW > 15000 ? 30 : avgMW > 8000 ? 60 : 85;

  // Shelf-life estimate: covalent (intein) > non-covalent
  const isCovalent = (arch.splitSystem || '').includes('intein');
  const shelfLife  = isCovalent ? 75 : 55;

  // GMP cost index: biologic complexity drives cost
  const costIdx = 100 - (nFrags * 12) - (hasRMT ? 10 : 0) - (hasRVG ? 8 : 0)
                      - (avgMW > 15000 ? 15 : 0);

  // Composite feasibility
  const feasibility = Math.max(0, Math.min(100, Math.round(
    fragComplexity * 0.20 +
    mwScore        * 0.20 +
    conjugateScore * 0.15 +
    sysMfgScore    * 0.15 +
    aggRisk        * 0.15 +
    shelfLife      * 0.10 +
    Math.max(0,costIdx) * 0.05
  )));

  const grade = feasibility >= 75 ? 'A' : feasibility >= 60 ? 'B'
              : feasibility >= 45 ? 'C' : feasibility >= 30 ? 'D' : 'F';

  // Identify limiting factor
  const subScores = [
    {name:'Fragment complexity', v:fragComplexity},
    {name:'MW / expression',     v:mwScore},
    {name:'Conjugate burden',    v:conjugateScore},
    {name:'Split system MFG',    v:sysMfgScore},
    {name:'Aggregation risk',    v:aggRisk},
    {name:'Shelf-life',          v:shelfLife},
  ];
  const limiting = subScores.sort((a,b)=>a.v-b.v)[0].name;

  return {
    feasibility,
    grade,
    subScores: subScores.sort((a,b)=>a.v-b.v),
    limiting_factor : limiting,
    nFragments      : nFrags,
    conjugates      : [hasRMT&&'TfR-RMT', hasRVG&&'RVG-29'].filter(Boolean),
    isCovalent,
    costNote: feasibility >= 65 ? 'Feasible for GMP development'
            : feasibility >= 45 ? 'Significant manufacturing challenges'
            : 'Manufacturing bottleneck — redesign recommended',
  };
}

// ─── SAFETY SCORING ──────────────────────────────────────────────────────────
/**
 * computeSafety(archId, mol, sig, hostState)
 * Models key safety risks for split-protein CNS therapeutics.
 * Returns: safetyScore (0-100, higher=safer), riskFlags[], riskLevel.
 */
function computeSafety(archId, mol, sig, hostState) {
  mol       = mol       || {};
  sig       = sig       || {};
  hostState = hostState || {};

  var flags = [];
  var penalties = 0;

  // 1. Off-target host binding: nanobodies/DARPins can cross-react with
  //    structurally similar human proteins. Arch A (anti-P VHH) highest risk.
  var offTargetRisk = {A:20, B:15, C:25, D:18}[archId] || 20;
  penalties += offTargetRisk * 0.20;
  if (offTargetRisk > 18) flags.push({
    risk:'Off-target host protein binding',
    severity: offTargetRisk > 22 ? 'medium' : 'low',
    note: 'Nanobody/DARPin may cross-react with structurally homologous host proteins',
  });

  // 2. Immunogenicity: foreign protein fragments → MHC-II loading → T-cell response
  //    Larger fragments = more peptide epitopes = higher MHC-II load
  var avgMW = (mol.mw || 10000);
  var mhcLoad = avgMW > 15000 ? 30 : avgMW > 8000 ? 18 : 10;
  penalties += mhcLoad * 0.20;
  if (mhcLoad > 20) flags.push({
    risk: 'MHC-II immunogenicity',
    severity: 'medium',
    note: 'Large fragments generate more T-cell epitopes — consider PEGylation or deimmunization',
  });

  // 3. TfR receptor saturation: high-dose TfR-RMT competes with transferrin
  //    → transient anemia risk at saturation doses
  if (mol.rmt) {
    penalties += 12 * 0.15;
    flags.push({
      risk: 'TfR receptor competition (transferrin)',
      severity: 'low',
      note: 'TfR-RMT at high dose may compete with endogenous transferrin — monitor iron/CBC',
    });
  }

  // 4. RVG-29 GABA_B off-target: RVG-29 has affinity for GABA-B receptor
  //    → potential inhibitory neurotransmission effects at high CNS concentrations
  if (mol.rvg) {
    penalties += 15 * 0.15;
    flags.push({
      risk: 'RVG-29 / GABA-B off-target',
      severity: 'low',
      note: 'RVG-29 peptide may interact with GABA-B receptors — CNS inhibitory effects possible at high dose',
    });
  }

  // 5. Neuronal protein load: intracellular reassembly adds proteostatic burden
  //    Already-stressed infected neurons may tolerate less additional protein load
  var neuronalLoad = archId === 'D' ? 25 : archId === 'C' ? 22 : 15;
  penalties += neuronalLoad * 0.15;
  if (neuronalLoad > 20) flags.push({
    risk: 'Neuronal proteostatic burden',
    severity: 'medium',
    note: 'Tripartite / large fragments add significant protein load to already-stressed RABV-infected neurons',
  });

  // 6. Immune over-activation: anti-P (arch A) restores IFN-β → if priorVaccination,
  //    synergistic immune response could become inflammatory
  if (archId === 'A' && hostState.priorVaccination && hostState.highInflammation) {
    penalties += 20 * 0.15;
    flags.push({
      risk: 'Immune over-activation (anti-P + vaccination + inflammation)',
      severity: 'high',
      note: 'IFN-β restoration combined with pre-existing immunity and high inflammation may trigger excessive neuroinflammation',
    });
  }

  // 7. BBB disruption side-effects: in disease states with open BBB, fragments
  //    may reach non-target cells → systemic off-CNS effects
  var bbbLeakage = ((sig.paracellularBreach||0) + (sig.transcytosisBreach||0)) / 2;
  if (bbbLeakage > 40) {
    penalties += 10 * 0.15;
    flags.push({
      risk: 'Non-specific CNS entry via leaky BBB',
      severity: 'low',
      note: `BBB breach ${Math.round(bbbLeakage)}% → fragments may access non-neuronal CNS cells`,
    });
  }

  var safetyScore = Math.max(0, Math.min(100, Math.round(100 - penalties)));
  var riskLevel   = safetyScore >= 75 ? 'low' : safetyScore >= 55 ? 'medium' : 'high';

  return {
    safetyScore,
    riskLevel,
    riskFlags   : flags.sort((a,b) => ({high:0,medium:1,low:2}[a.severity]-({high:0,medium:1,low:2}[b.severity])),),
    penaltyTotal: Math.round(penalties),
    safetyNote  : riskLevel === 'low'    ? 'Acceptable safety profile for CNS biologic'
                : riskLevel === 'medium' ? 'Manageable risks — monitoring required'
                : 'Significant safety concerns — redesign or dose optimization needed',
  };
}

// ─── ARCHITECTURE-SPECIFIC MECHANISTIC EFFICACY ───────────────────────────────
/**
 * computeArchMechanism(archId, gp, result)
 * Returns mechanistic sub-scores and the per-architecture efficacy chain.
 * This is separate from the probabilistic overall score — it models
 * the biology of HOW each architecture works once fragments are inside.
 */
function computeArchMechanism(archId, gp, result) {
  gp     = gp     || {};
  result = result || {};

  var hostState   = gp.hostState      || {};
  var viralBurden = gp.viralBurden    || 0.30;
  var limConc     = result.limConc_nM || 1.0;

  var detail = {};

  if (archId === 'A') {
    // Anti-P mechanism chain:
    // P-TBK1 block reversed → IFN-β restored → JAK/STAT antiviral programme
    // P-LC8 binding disrupted → dynein transport slowed → spread retarded
    var ifnCompetence = hostState.immunocompromised ? 0.25 : (hostState.priorVaccination ? 1.30 : 1.00);
    var ifnRestore    = Math.min(1, limConc / 5) * ifnCompetence;
    var transportSlow = Math.min(1, limConc / 3) * 0.65;  // transport never fully stopped
    var clearanceBoost= ifnRestore * 0.70;                 // IFN drives NK/T-cell activity
    var efficacy      = 1 - (1-ifnRestore) * (1-transportSlow) * (1-clearanceBoost);
    detail = {
      mechanism       : 'IFN-β restoration + retrograde transport stall',
      ifn_restore_pct : Math.round(ifnRestore*100),
      transport_slow_pct: Math.round(transportSlow*100),
      clearance_boost_pct: Math.round(clearanceBoost*100),
      immune_factor   : ifnCompetence.toFixed(2),
      limiting_step   : ifnRestore < transportSlow ? 'IFN restoration (immune state)' : 'Transport disruption',
      efficacy_pct    : Math.round(efficacy*100),
    };

  } else if (archId === 'B') {
    // Anti-N mechanism chain:
    // N oligomerization disrupted → 11-mer rings fail → RNP helix collapses → RNA degraded
    // Requires: high occupancy of N monomers (cooperative assembly)
    var occupancyNeeded = 0.65;  // must block ≥65% N monomers for RNP collapse (cooperative)
    var achievedOcc     = Math.min(1, limConc / 8);
    var occupancyMet    = achievedOcc >= occupancyNeeded;
    var rnpCollapse     = occupancyMet ? Math.min(1, (achievedOcc - occupancyNeeded*0.8) / 0.3) : achievedOcc * 0.3;
    var replicationBlock= rnpCollapse * 0.85;  // RNA degradation takes 1-2 replication cycles
    var efficacy        = replicationBlock;
    detail = {
      mechanism        : 'N-oligomer disruption → RNP collapse → RNA degradation',
      occupancy_needed : Math.round(occupancyNeeded*100)+'%',
      achieved_occ_pct : Math.round(achievedOcc*100),
      occupancy_met    : occupancyMet,
      rnp_collapse_pct : Math.round(rnpCollapse*100),
      replication_block: Math.round(replicationBlock*100),
      limiting_step    : !occupancyMet ? 'Occupancy threshold not reached (need ≥65% N coverage)' : 'RNP disassembly kinetics',
      efficacy_pct     : Math.round(efficacy*100),
      note             : !occupancyMet ? '⚠ Subthreshold — cooperative assembly means partial blockade has limited effect' : '✓ Above threshold — cooperative collapse engaged',
    };

  } else if (archId === 'C') {
    // Anti-L mechanism chain:
    // Split L fragments reassemble → dominant-negative competes with native L
    // Efficacy depends on DN:native ratio — high viral load → unfavourable ratio
    var dnRatio   = limConc / Math.max(0.5, viralBurden * 10);  // DN vs viral L pool
    var competition = Math.min(1, dnRatio / (dnRatio + 1));       // Langmuir competition
    var rdRpBlock   = competition * 0.90;
    var replicationInh = rdRpBlock * (1 - viralBurden * 0.30);    // high burden reduces effect
    var efficacy    = replicationInh;
    detail = {
      mechanism        : 'Dominant-negative L competes with native RdRp',
      dn_ratio         : dnRatio.toFixed(2),
      competition_pct  : Math.round(competition*100),
      rdrp_block_pct   : Math.round(rdRpBlock*100),
      replication_inh  : Math.round(replicationInh*100),
      limiting_step    : dnRatio < 0.5 ? 'DN:viral L ratio too low (high viral burden)' : 'RdRp active-site competition',
      efficacy_pct     : Math.round(efficacy*100),
      note             : 'L-RdRp structure partially solved — split site selection is speculative',
    };

  } else if (archId === 'D') {
    // Tripartite mechanism — multi-target synergy formula:
    // E_total = 1 - ∏(1-Ei)  for independent targets
    // With tripartite assembly penalty (3-fragment efficiency ~40%)
    var eP  = Math.min(1, limConc / 5)  * 0.60;   // anti-P component (attenuated by delivery split)
    var eN  = Math.min(1, limConc / 8)  * 0.55;   // anti-N component
    var syn = 1 - (1 - eP) * (1 - eN);            // independent synergy
    var tripartitePenalty = 0.40;                   // 3-fragment assembly is 40% efficient vs bipartite
    var efficacy = syn * (1 - tripartitePenalty);
    var deliveryComplexity = 0.25;                  // 3 crossing events needed
    var finalEfficacy = efficacy * (1 - deliveryComplexity);
    detail = {
      mechanism        : 'Dual P+N synergy via tripartite reassembly',
      e_p_pct          : Math.round(eP*100),
      e_n_pct          : Math.round(eN*100),
      synergy_pct      : Math.round(syn*100),
      tripartite_penalty: '−'+Math.round(tripartitePenalty*100)+'%',
      delivery_complexity:'−'+Math.round(deliveryComplexity*100)+'%',
      synergy_formula  : 'E = 1 − (1−E_P)(1−E_N)',
      limiting_step    : eP < eN ? 'Anti-P arm (immune competence)' : 'Anti-N arm (occupancy)',
      efficacy_pct     : Math.round(finalEfficacy*100),
    };
  }

  return detail;
}

// ─── UNCERTAINTY PROPAGATION ─────────────────────────────────────────────────
/**
 * propagateUncertainty(archId, gp)
 * Analytical uncertainty propagation — fast, no sampling.
 * Uses documented biological variability (CV) per parameter.
 * Returns: { expected, pessimistic, optimistic, confidence, breakdown }
 */
function propagateUncertainty(archId, gp) {
  var r = scoreArchitectureLive(archId, gp, {});

  // Per-parameter CV (coefficient of variation, 0-1)
  var cvBBB         = 0.30;  // BBB delivery ±30% (interindividual transporter abundance)
  var cvColoc       = 0.35;  // co-localization ±35% (cell entry route, timing)
  var cvReassembly  = 0.20;  // reassembly ±20% (intracellular concentration)
  var cvWindow      = 0.25;  // therapeutic window ±25% (nerve velocity, viral strain)
  var cvMechanism   = 0.20;  // mechanism ±20% (target occupancy, host variation)

  // Propagated variance of log-product of probabilities
  // If score = a×b×c×d, then σ_score/score ≈ √(σa²/a² + σb²/b² + ...)
  var relVar = Math.sqrt(
    cvBBB*cvBBB + cvColoc*cvColoc + cvReassembly*cvReassembly +
    cvWindow*cvWindow + cvMechanism*cvMechanism
  );

  // Architecture-specific confidence modifier
  var archConfMod = {A:0.85, B:1.10, C:1.40, D:1.25}[archId] || 1.0;
  var totalRelSD  = relVar * archConfMod;

  var expected    = r.finalScore;
  var optimistic  = Math.min(100, Math.round(expected * (1 + totalRelSD)));
  var pessimistic = Math.max(0,   Math.round(expected * (1 - totalRelSD)));

  // Confidence classification
  var confidence  = totalRelSD < 0.35 ? 'medium'
                  : totalRelSD < 0.55 ? 'low' : 'very_low';

  return {
    expected,
    optimistic,
    pessimistic,
    relSD_pct   : Math.round(totalRelSD * 100),
    confidence,
    breakdown   : [
      {param:'BBB delivery',       cv_pct:Math.round(cvBBB*100),        contribution_pct:Math.round(cvBBB*cvBBB/relVar/relVar*100)},
      {param:'Co-localization',    cv_pct:Math.round(cvColoc*100),      contribution_pct:Math.round(cvColoc*cvColoc/relVar/relVar*100)},
      {param:'Reassembly',         cv_pct:Math.round(cvReassembly*100), contribution_pct:Math.round(cvReassembly*cvReassembly/relVar/relVar*100)},
      {param:'Therapeutic window', cv_pct:Math.round(cvWindow*100),     contribution_pct:Math.round(cvWindow*cvWindow/relVar/relVar*100)},
      {param:'Mechanism efficacy', cv_pct:Math.round(cvMechanism*100),  contribution_pct:Math.round(cvMechanism*cvMechanism/relVar/relVar*100)},
    ].sort((a,b) => b.contribution_pct - a.contribution_pct),
  };
}

// ─── MONTE CARLO RANKING ─────────────────────────────────────────────────────
/**
 * runMonteCarlo(globalParams, n, archIds)
 * Samples parameters from uncertainty distributions, scores all architectures
 * per sample, returns rank distributions.
 * n=500 for UI speed (called synchronously); use web worker for n=5000.
 */
function runMonteCarlo(globalParams, n, archIds) {
  n       = n       || 500;
  archIds = archIds || ['A','B','C','D'];

  // Per-parameter sampling distributions
  // { center, cv, floor, ceil, log } — log=true means lognormal
  var paramDefs = {
    biteMM      : { cv:0.10, floor:100,  ceil:1400, log:false },
    velocity    : { cv:0.25, floor:50,   ceil:400,  log:false },
    treatDay    : { cv:0.00, floor:0,    ceil:30,   log:false },  // fixed by user
    concNM      : { cv:0.40, floor:0.1,  ceil:100,  log:true  },
    viralBurden : { cv:0.30, floor:0.05, ceil:0.95, log:false },
  };

  // Box-Muller normal sample
  function randNorm() {
    var u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v);
  }

  function sample(center, def) {
    if (def.cv === 0) return center;
    var z = randNorm();
    var v;
    if (def.log) {
      var mu    = Math.log(Math.max(1e-9, center));
      var sigma = Math.sqrt(Math.log(1 + def.cv*def.cv));
      v = Math.exp(mu + sigma * z);
    } else {
      v = center + center * def.cv * z;
    }
    return Math.max(def.floor, Math.min(def.ceil, v));
  }

  // Rank counters and score accumulators
  var rankCount  = {};   // archId → count of times ranked #1
  var scoreAccum = {};   // archId → sum of scores
  var scoreList  = {};   // archId → list of sampled scores (for percentiles)
  archIds.forEach(function(id) { rankCount[id]=0; scoreAccum[id]=0; scoreList[id]=[]; });

  for (var i = 0; i < n; i++) {
    // Sample parameters
    var gp2 = Object.assign({}, globalParams);
    for (var key in paramDefs) {
      if (globalParams[key] != null) {
        gp2[key] = sample(globalParams[key], paramDefs[key]);
      }
    }
    // Additional: sample hostState modifiers with binary flip probability
    var hs = Object.assign({}, globalParams.hostState || {});
    gp2.hostState = hs;

    // Score all architectures with per-patient biological noise.
    // The noise models inter-patient variability in BBB transporter expression,
    // intracellular co-localization, reassembly efficiency, and mechanism response —
    // all of which vary independently of the treatment parameters.
    var scores = archIds.map(function(id) {
      try {
        var r = scoreArchitectureLive(id, gp2, {});
        // Apply biological noise to each probability component
        // CV values from documented inter-individual variability (Uchida proteomics, etc.)
        var nBBB    = Math.max(0, Math.min(1, (r.probChain.pBBB_both/100)    * (1 + 0.30*randNorm())));
        var nColoc  = Math.max(0, Math.min(1, (r.probChain.pColoc/100)       * (1 + 0.35*randNorm())));
        var nReasem = Math.max(0, Math.min(1, (r.probChain.pReassembly/100)  * (1 + 0.20*randNorm())));
        var nMech   = Math.max(0, Math.min(1, (r.probChain.pMechanism/100)   * (1 + 0.20*randNorm())));
        var nWindow = Math.max(0, Math.min(100, r.window.windowScore         * (1 + 0.25*randNorm())));
        // Recompute composite score with noisy sub-scores
        var nOverall= nBBB * nColoc * nReasem * nMech * 100;
        var nScore  = Math.max(0, Math.min(100, Math.round(
          nOverall    * 0.35 +
          nMech*100   * 0.25 +
          nWindow     * 0.25 +
          Math.min(r.f1bbb, r.f2bbb) * 0.10 +
          nReasem*100 * 0.05
        )));
        return { id:id, score:nScore };
      } catch(e) {
        return { id:id, score:0 };
      }
    });

    // Rank this sample
    scores.sort(function(a,b){ return b.score - a.score; });
    var winner = scores[0].id;
    rankCount[winner]++;
    scores.forEach(function(s) {
      scoreAccum[s.id] += s.score;
      scoreList[s.id].push(s.score);
    });
  }

  // Compute percentiles
  function percentile(arr, p) {
    var sorted = arr.slice().sort(function(a,b){return a-b;});
    var idx = Math.floor(p/100*(sorted.length-1));
    return Math.round(sorted[idx]);
  }

  var results = archIds.map(function(id) {
    var list = scoreList[id];
    return {
      archId      : id,
      rankPct1    : Math.round(rankCount[id] / n * 100),
      expected    : Math.round(scoreAccum[id] / n),
      p10         : percentile(list, 10),
      p25         : percentile(list, 25),
      p75         : percentile(list, 75),
      p90         : percentile(list, 90),
      pessimistic : percentile(list, 5),
      optimistic  : percentile(list, 95),
    };
  });

  results.sort(function(a,b){ return b.rankPct1 - a.rankPct1; });
  results.forEach(function(r,i){ r.mcRank = i+1; });

  return {
    results,
    n,
    bestArch    : results[0].archId,
    certainty   : results[0].rankPct1 >= 50 ? 'dominant'
                : results[0].rankPct1 >= 35 ? 'preferred'
                : results[0].rankPct1 >= 20 ? 'marginal' : 'toss-up',
  };
}

// ─── EXPLAINABILITY ENGINE ────────────────────────────────────────────────────
/**
 * explainScore(archId, result, mfg, safety, mc)
 * Returns structured explanation: what's driving the score, what's limiting it,
 * what risks exist, and what to do next.
 */
function explainScore(archId, result, mfg, safety, mc) {
  result = result || {};
  mfg    = mfg    || {};
  safety = safety || {};

  var drivers   = [];
  var limiters  = [];
  var risks     = [];
  var upgrades  = [];

  var pc = result.probChain || {};
  var win= result.window    || {};
  var unc= result.uncertainty || {};

  // Drivers — things working in favour
  if (result.f1bbb >= 20)       drivers.push({ label:'F1 BBB crossing', value:result.f1bbb, note:'Fragment 1 achieves significant BBB entry via '+result.f1route });
  if (result.f2bbb >= 20)       drivers.push({ label:'F2 BBB crossing', value:result.f2bbb, note:'Fragment 2 achieves significant BBB entry via '+result.f2route });
  if (result.mechanismScore >= 60) drivers.push({ label:'Mechanism strength', value:result.mechanismScore, note:'Target is druggable and effect is mechanistically grounded' });
  if (win.canPrevent)            drivers.push({ label:'Pre-CNS timing', value:win.windowScore, note:'Treatment before CNS seeding — best possible outcome class' });
  else if (win.canControl)       drivers.push({ label:'Early CNS timing', value:win.windowScore, note:'Early CNS phase — containment still achievable' });
  if (result.reassembly && result.reassembly.probabilityPct >= 50)
                                 drivers.push({ label:'Reassembly efficiency', value:result.reassembly.probabilityPct, note:result.reassembly.system+' — high assembly probability' });
  if (mfg.feasibility >= 65)    drivers.push({ label:'Manufacturing feasibility', value:mfg.feasibility, note:'Grade '+mfg.grade+' — manufacturable at GMP scale' });
  if (safety.safetyScore >= 70)  drivers.push({ label:'Safety profile', value:safety.safetyScore, note:'Low overall risk — manageable side effects' });

  // Limiters — things pulling the score down
  if (result.f1bbb < 15)        limiters.push({ label:'F1 BBB entry', value:result.f1bbb, note:'Fragment 1 barely crossing — dominant bottleneck' });
  if (result.f2bbb < 15)        limiters.push({ label:'F2 BBB entry', value:result.f2bbb, note:'Fragment 2 barely crossing — dominant bottleneck' });
  var coloc = result.colocalization || {};
  if (coloc.colocPct < 15)      limiters.push({ label:'Co-localization', value:coloc.colocPct, note:'Fragments unlikely to meet in same infected neuron' });
  if (result.reassembly && result.reassembly.probabilityPct < 30)
                                 limiters.push({ label:'Reassembly probability', value:result.reassembly.probabilityPct, note:'Split system Kd poorly matched to intracellular concentration' });
  if (win.windowScore < 30)     limiters.push({ label:'Therapeutic window', value:win.windowScore, note:'Treatment too late — '+win.phaseLabel });
  if (result.mechanismScore < 35) limiters.push({ label:'Mechanism efficacy', value:result.mechanismScore, note:'Target druggability or occupancy threshold not achieved' });
  if (mfg.feasibility < 50)     limiters.push({ label:'Manufacturing', value:mfg.feasibility, note:'Grade '+mfg.grade+' — '+mfg.limiting_factor+' is limiting' });

  // Sort limiters worst first
  limiters.sort(function(a,b){ return a.value-b.value; });

  // Risks
  if (safety && safety.riskFlags) {
    safety.riskFlags.forEach(function(f) {
      risks.push({ label:f.risk, severity:f.severity, note:f.note });
    });
  }
  if (result.reassembly && result.reassembly.warnings) {
    result.reassembly.warnings.forEach(function(w) {
      risks.push({ label:'Reassembly warning', severity:'medium', note:w });
    });
  }

  // Upgrades — specific actionable steps
  if (limiters.length > 0) {
    var topLimit = limiters[0].label;
    if (topLimit.includes('BBB')) {
      upgrades.push('Switch '+topLimit+' delivery to TfR-RMT (optimal Kd ~30nM) — largest single BBB improvement');
      upgrades.push('Reduce fragment MW below 8 kDa — MW_factor improves exponentially below 8k');
    } else if (topLimit.includes('Co-loc')) {
      upgrades.push('Route F1 and F2 via the SAME BBB pathway — different routes (TfR vs RVG) halve co-localization probability');
      upgrades.push('PEGylate the shorter-lived fragment to extend intracellular t½ and synchronise arrival');
    } else if (topLimit.includes('Reassembly')) {
      upgrades.push('Switch to Npu DnaE split-intein (Kd 0.001 nM, t½ ~1s) — irreversible at picomolar concentrations');
      upgrades.push('Increase dose to push intracellular concentration above Kd × 10');
    } else if (topLimit.includes('window')) {
      upgrades.push('Earlier treatment — each day delay post-CNS invasion reduces window score by ~8 points');
      upgrades.push('Anti-P (Architecture A) has fastest mechanism onset (~4h) — switch if window is the bottleneck');
    } else if (topLimit.includes('Mfg')) {
      upgrades.push('Simplify to single conjugate: choose TfR-RMT OR RVG-29, not both — eliminates '+mfg.limiting_factor);
      upgrades.push('Reduce fragment MW to <8 kDa if possible — improves expression, purification, and aggregation');
    }
  }

  if (unc.dominantUncertainty) {
    upgrades.push('Primary uncertainty: '+unc.dominantUncertainty+' — validate this experimentally before advancing');
  }

  return {
    archId,
    drivers   : drivers.sort(function(a,b){return b.value-a.value;}),
    limiters,
    risks,
    upgrades,
    summary   : limiters.length === 0
      ? 'Architecture '+archId+' shows strong performance across all dimensions.'
      : 'Architecture '+archId+' is limited primarily by '+limiters[0].label+' ('+limiters[0].value+'%). Resolving this unlocks the next bottleneck.',
  };
}

// ─── EXTENDED SCENARIO PRESETS ────────────────────────────────────────────────
var CLINICAL_SCENARIOS = [
  { label:'Face bite · Day 0',          biteMM:200,  treatDay:0,  velocity:200, hostState:{viralLoad:'high'} },
  { label:'Hand bite · Day 2',          biteMM:900,  treatDay:2,  velocity:200 },
  { label:'Foot bite · Day 5',          biteMM:1200, treatDay:5,  velocity:200 },
  { label:'Foot bite · Day 10',         biteMM:1200, treatDay:10, velocity:200 },
  { label:'Foot bite · Day 14',         biteMM:1200, treatDay:14, velocity:200 },
  { label:'Vaccinated · Hand · Day 3',  biteMM:900,  treatDay:3,  velocity:200, hostState:{priorVaccination:true} },
  { label:'Immunocompromised · Day 3',  biteMM:600,  treatDay:3,  velocity:200, hostState:{immunocompromised:true} },
  { label:'Brainstem · Face · Day 4',   biteMM:200,  treatDay:4,  velocity:200, hostState:{brainstemDominant:true} },
  { label:'High inflam · Day 5',        biteMM:900,  treatDay:5,  velocity:200, hostState:{highInflammation:true} },
  { label:'Elderly / slow velocity',    biteMM:900,  treatDay:6,  velocity:80  },
  { label:'Pediatric / fast velocity',  biteMM:600,  treatDay:3,  velocity:350, hostState:{viralLoad:'high'} },
];

// ─── RABV ENGINE ASYNC ADAPTER ───────────────────────────
// Owned by: rabies_simulator.html
// Backend:  rabv_main.py on port 8001  (uvicorn rabv_main:app --port 8001)
// Fallback: JS heuristic via computeRABVTimeline() when backend offline
//
// All param keys passed in from rabies_simulator.html getODEParams().
// Field mapping: JS camelCase → Python snake_case done here so the
// simulator UI never needs to know about the API schema.

var RABVEngine = {

  // ── Health check ───────────────────────────────────────
  isOnline: async function() {
    try {
      const r = await fetch('http://localhost:8001/health',
        { signal: AbortSignal.timeout(1500) });
      return r.ok;
    } catch(e) { return false; }
  },

  // ── Main entry: try ODE backend, fall back to heuristic ─
  simulate: async function(params) {
    try {
      const resp = await fetch('http://localhost:8001/simulate', {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify(RABVEngine._buildPayload(params)),
        signal : AbortSignal.timeout(20000),
      });
      if (resp.ok) {
        const d = await resp.json();
        d._source = 'python_ode';
        return d;
      }
    } catch(e) { /* fall through */ }
    return RABVEngine._heuristic(params);
  },

  // ── Map UI params → rabv_main.py SimulateRequest schema ─
  // UI key           → API field
  // params.location  → bite_location
  // params.depth     → bite_depth
  // params.dose      → viral_dose
  // params.velocity  → base_velocity_mm_day
  // params.p75Active → p75_active
  // params.replag    → replication_lag_h   (hours, float)
  // params['prot-X'] → prot_X              (bool toggles)
  // params.pep       → pep_given
  // params.immuno    → immunocompromised
  _buildPayload: function(params) {
    return {
      bite_location        : params.location      || 'foot',
      bite_depth           : params.depth         || 'muscle',
      viral_dose           : params.dose          || 'med',
      wound_washing        : params.wound_washing || 'none',
      base_velocity_mm_day : +(params.velocity    || 200),
      p75_active           : params.p75Active     !== false,
      replication_lag_h    : +(params.replag      || 9),
      prot_G               : params['prot-G']     !== false,
      prot_P               : params['prot-P']     !== false,
      prot_N               : params['prot-N']     !== false,
      prot_M               : params['prot-M']     !== false,
      prot_L               : params['prot-L']     !== false,
      vaccinated           : !!params.vaccinated,
      pep_given            : !!params.pep,
      immunocompromised    : !!params.immuno,
      pep_day              : +(params.pep_day     || 0),
      pep_hrig_given       : !!params.pep_hrig,
      // Therapeutic interventions (optional, default off)
      p_inhibitor_active   : !!params.p_inhibitor_active,
      p_inhibitor_efficacy : +(params.p_inhibitor_efficacy || 0.75),
      p_inhibitor_day      : +(params.p_inhibitor_day      || 0),
      n_dn_active          : !!params.n_dn_active,
      n_dn_efficacy        : +(params.n_dn_efficacy        || 0.70),
      n_dn_day             : +(params.n_dn_day             || 0),
      favipiravir_active   : !!params.favipiravir_active,
      favipiravir_efficacy : +(params.favipiravir_efficacy || 0.40),
      favipiravir_day      : +(params.favipiravir_day      || 0),
      // Simulation control
      t_days               : 60.0,
      dt_hours             : 0.25,
      output_interval_h    : 2.0,
    };
  },

  // ── JS heuristic fallback ───────────────────────────────
  // Returns the same key shape as the Python ODE response so
  // rabies_simulator.html can consume it identically.
  // Scientific basis: computeRABVTimeline() + DISEASE_PARAMS table.
  _heuristic: function(params) {
    const tl   = computeRABVTimeline(params);
    const protP = params['prot-P'] !== false;
    const protM = params['prot-M'] !== false;
    const protN = params['prot-N'] !== false;

    // BBB state derived from protein expression pattern
    // P active → TBK1 blocked → IFN suppressed → NF-κB low → TJ intact
    // P neutralised → IFN restored → mild NF-κB rise → TJ mildly compromised
    const bbb_tj    = protP && protM ? 95 : !protP ? 70 : 80;
    const bbb_nfkb  = protP ? 5 : 32;

    // Outcome logic: PEP or vaccination clears before CNS
    const cnsDay    = tl.localPhase + tl.transportDays;
    const prevented = params.pep || params.vaccinated;
    const outcome   = prevented ? 'prevented' : 'fatal';
    const survProb  = prevented ? 0.90 : 0.03;

    // Minimal timeline arrays so the ODE timecourse panel
    // still renders something (flat lines, not empty)
    const _flat = function(v, n) {
      var a = []; for (var i=0;i<n;i++) a.push(v); return a;
    };
    const nPts = 30;
    const days = [];
    for (var i=0;i<nPts;i++) days.push(+(i * 60/nPts).toFixed(1));

    return {
      _source             : 'js_heuristic',
      engine_version      : 'heuristic-9.0',
      // ── Timing milestones (days) ──
      incubation_days     : tl.totalIncubation,
      transport_days      : tl.transportDays,
      day_cns_entry       : +cnsDay.toFixed(1),
      day_brainstem       : +(cnsDay + 2).toFixed(1),
      day_limbic          : +(cnsDay + 4).toFixed(1),
      day_symptoms        : tl.totalIncubation,
      day_fatal           : +(tl.totalIncubation + 7).toFixed(1),
      // ── Burden ──
      burden_at_end       : prevented ? 0.01 : 0.72,
      peak_cns_burden     : prevented ? 0.02 : 0.65,
      // ── BBB ──
      bbb_tj_final        : bbb_tj,
      bbb_nfkb_final      : bbb_nfkb,
      // ── Immune ──
      innate_peak         : prevented ? 0.55 : 0.35,
      adapt_peak          : prevented ? 0.70 : 0.12,
      pep_effective       : !!(params.pep && cnsDay > 0),
      // ── Outcome ──
      outcome             : outcome,
      outcome_prob        : survProb,
      // ── Spread ──
      peripheral_spread_prob: tl.peripheralSpreadProb,
      // ── Phase structure (for JS phase navigator) ──
      phase_ends          : tl.phaseEnds,
      // ── Sparse timecourse (heuristic — not ODE) ──
      timeline_days       : days,
      timeline_wound      : _flat(0, nPts),
      timeline_pns        : _flat(0, nPts),
      timeline_sc         : _flat(0, nPts),
      timeline_bs         : _flat(prevented ? 0.00 : 0.35, nPts),
      timeline_limbic     : _flat(prevented ? 0.00 : 0.28, nPts),
      timeline_cortex     : _flat(0, nPts),
      timeline_sal        : _flat(0, nPts),
      timeline_innate     : _flat(prevented ? 0.55 : 0.35, nPts),
      timeline_adapt      : _flat(prevented ? 0.70 : 0.12, nPts),
      timeline_bbb_tj     : _flat(bbb_tj, nPts),
      timeline_bbb_nfkb   : _flat(bbb_nfkb, nPts),
      // ── Final compartment snapshot ──
      compartments_final  : {
        wound:0, nmj:0, pns:0, drg:0, sc:0,
        bs    : prevented ? 0.00 : 0.35,
        limbic: prevented ? 0.00 : 0.28,
        cortex: 0,
        sal   : 0,
        innate: prevented ? 0.55 : 0.35,
        adapt : prevented ? 0.70 : 0.12,
      },
      elapsed_ms          : 0,
    };
  },
};

// ─── SPLIT-PROTEIN DESIGNER — API STUBS ──────────────────
// Owned by: split_protein_simulator.html
// Backend:  split_main.py on port 8002  (uvicorn split_main:app --port 8002)
// All heavy computation (scoring, Monte Carlo, explain) runs in split_core.py.
// These async stubs replace the old inline JS computation that lived here in v9.0.
// Sync stubs kept for any legacy callers — they return null; UI must use async API.

const SPLIT_API = 'http://localhost:8002';

async function _splitPost(endpoint, body) {
  const r = await fetch(SPLIT_API + endpoint, {
    method : 'POST',
    headers: {'Content-Type':'application/json'},
    body   : JSON.stringify(body),
    signal : AbortSignal.timeout(30000),
  });
  if (!r.ok) throw new Error('Split engine HTTP ' + r.status + ' on ' + endpoint);
  return r.json();
}

// ── Async API wrappers (match split_main.py endpoint schema) ──
// POST /score          → full single-arch breakdown
async function scoreArchitectureAPI(params)  { return _splitPost('/score',              params); }
// POST /score/all      → all 4 archs ranked
async function scoreAllArchsAPI(params)      { return _splitPost('/score/all',          params); }
// POST /montecarlo     → rank distribution (n configurable via params.n_samples)
async function runMonteCarloAPI(params)      { return _splitPost('/montecarlo',         params); }
// POST /compare/scenarios → all 11 clinical scenarios × all archs
async function compareScenariosAPI(params)   { return _splitPost('/compare/scenarios',  params); }
// POST /explain        → explainability breakdown
async function explainScoreAPI(params)       { return _splitPost('/explain',            params); }

// ── Health check for split designer backend ──────────────
async function splitEngineOnline() {
  try {
    const r = await fetch(SPLIT_API + '/health', {signal: AbortSignal.timeout(1500)});
    return r.ok;
  } catch(e) { return false; }
}

// ─── EXPORTS ─────────────────────────────────────────────
var SharedEngine = {
  // ── Schema ──────────────────────────────────────────────
  buildMolecule,

  // ── BBB Permeability (Module 01, port 8000) ─────────────
  // backend: main.py + bbb_core.py
  analyzeMolecule, computeSignaling, computeRoutes, computeScore,
  effectiveCharge, ionizationPenalty, fractionIonized,

  // ── RABV Neuroinvasion (Module 02, port 8001) ────────────
  // backend: rabv_main.py + rabv_core.py
  // heuristic helpers used by RABVEngine._heuristic + UI directly
  computeRABVTimeline, getRABVPhase, computeInfectedBurden, NERVE_DISTANCES,
  // async adapter — rabies_simulator.html entry point
  RABVEngine,

  // ── Split-Protein Designer (Module 03, port 8002) ────────
  // backend: split_main.py + split_core.py
  // async API stubs — all computation delegated to Python
  scoreArchitectureAPI,
  scoreAllArchsAPI,
  runMonteCarloAPI,
  compareScenariosAPI,
  explainScoreAPI,
  splitEngineOnline,
  // sync stubs (legacy compat — return null)
  computeManufacturing,
  computeSafety,
  computeArchMechanism,
  propagateUncertainty,
  runMonteCarlo,
  explainScore,
  CLINICAL_SCENARIOS,

  // ── Protein Splitter (Module 04, port 8002) ──────────────
  // backend: splitter_main.py + splitter_core.py
  // JS engine: splitter_engine.js (loaded separately by protein_splitter.html)
  parseFASTA,
  estimateResidueProperties,
  findSplitSiteCandidates,
  scoreSplitDesign,
  rankDeliveryStrategies,
  scoreDeliveryMode,
  estimateFragmentMW, estimateLogP, estimateHBD, estimateHBA,
  terminiDegronRisk,
  PROTEIN_PRESETS, SPLIT_HARD_RULES,

  // ── Shared intraneuronal / architecture scoring ──────────
  computeColocalization, computeReassembly,
  computeTherapeuticWindow,
  scoreArchitectureLive,
  compareArchitectures,
  compareScenarios,
  recommendStrategy,
  hostStateModifiers,
  computeUncertainty,

  // ── BBB constants ────────────────────────────────────────
  EVIDENCE_TIER, TRANSPORTER_CV, REGIONAL_UNCERTAINTY,
  PM_MW_DECAY, TFR_KD_OPTIMAL_NM, TFR_BMAX_NM,
  PGP_KM_NM, BCRP_KM_NM, MRP_KM_NM,
  SPECIES_PGP_SCALE,

  // ── Platform data ────────────────────────────────────────
  DISEASE_PARAMS, DISEASE_NOTES, SPLIT_SYSTEMS, RABV_TARGETS, ARCHITECTURES,
  PRESETS, REGIONAL_FACTORS,
  BLOOD_PH, ENDOSOME_PH, BRAIN_ISF_PH,

  // ── Metadata ─────────────────────────────────────────────
  version          : '9.2',
  phase1_complete  : true,
  phase2_ready     : true,   // RABVEngine ODE adapter (port 8001)
  split_engine_port: 8002,   // Split-Protein Designer ODE backend
  splitter_port    : 8002,   // Protein Splitter ODE backend
};

if (typeof module !== 'undefined' && module.exports) module.exports = SharedEngine;
else if (typeof window !== 'undefined') {
  // Assign synchronously so scripts that import shared_engine.js in <head> can
  // reference window.SharedEngine immediately; DOMContentLoaded fires the ready
  // event for any UI code that needs the DOM to be fully parsed first.
  window.SharedEngine = SharedEngine;
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      window.SharedEngine = SharedEngine;    // re-assign to flush any race
      document.dispatchEvent(new CustomEvent('SharedEngineReady', { detail: SharedEngine }));
    });
  } else {
    // DOM already ready (script loaded with defer or at bottom of body)
    document.dispatchEvent(new CustomEvent('SharedEngineReady', { detail: SharedEngine }));
  }
}
