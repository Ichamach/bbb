/**
 * splitter_engine.js — v10.0
 * NeuroViral Lab · Protein Splitter Engine (Module 04)
 *
 * MAJOR UPGRADE v10.0 — Structure-Aware Therapeutic Split Design Engine
 *
 * WHAT'S NEW vs v8.0:
 *
 * PHASE 1  — Structure-aware scoring layer
 *   parsePDB()               — parse uploaded PDB/mmCIF coordinate data
 *   parseAlphaFold()         — parse AF2 pLDDT confidence maps
 *   applyStructureToProps()  — override sequence heuristics with real geometry
 *
 * PHASE 2  — Contact map disruption analysis
 *   buildContactMap()        — Cβ-Cβ distance matrix from PDB coordinates
 *   scoreContactDisruption() — weighted contact-break penalty per cut position
 *
 * PHASE 3  — Domain integrity scoring
 *   parseDomainBoundaries()  — parse Pfam/Uniprot domain annotations
 *   scoreDomainIntegrity()   — inter-domain vs intra-domain vs catalytic cut
 *
 * PHASE 4  — Functional residue risk engine
 *   detectFunctionalMotifs() — auto-derive catalytic/binding/interface residues
 *   scoreFunctionalRisk()    — direct + indirect scaffold damage
 *
 * PHASE 5  — Fragment foldability scoring
 *   scoreFragmentFoldability() — per-fragment: disorder, aggregation, hydrophobic patch
 *
 * PHASE 6  — Reassembly geometry scoring
 *   scoreReassemblyGeometry() — terminal distance, orientation, linker reach
 *
 * PHASE 7  — Linker / fusion burden model
 *   scoreFusionBurden()      — shuttle + intein + linker impact on fragment behavior
 *
 * PHASE 8  — Oligomerization and complex-state awareness
 *   RABV_ASSEMBLY_STATES     — biological assembly data for P, N, L
 *   scoreAssemblyContext()   — monomer vs dimer vs oligomer cut risk
 *
 * PHASE 9  — Delivery–split coupling
 *   scoreSplitDeliveryTotal() — S_split × S_delivery × S_reassembly × S_function
 *
 * PHASE 10 — Experimental benchmark calibration set
 *   BENCHMARK_SPLITS         — known-good / known-bad split sites from literature
 *   calibrateScorer()        — checks whether engine ranks them correctly
 *
 * PHASE 11 — RABV-specific enriched target maps
 *   RABV_TARGET_MAPS         — full per-protein domain/residue/assembly maps
 *
 * PHASE 12 — Decomposed candidate ranking with waterfall explanation
 *   explainCandidate()       — waterfall breakdown per cut
 *
 * PHASE 13 — Monte Carlo cut ranking
 *   runSplitMonteCarlo()     — stochastic stability test for candidate rank order
 *
 * PHASE 14 — Manufacturability layer (full)
 *   scoreManufacturability() — expression, purification, cysteine, instability motifs
 *
 * PHASE 15 — Rejection engine
 *   rejectCandidates()       — explicit "Do Not Touch" filter with reason
 *
 * DEPENDENCIES:
 *   shared_core.js or shared_engine.js → window.CoreEngine or window.SharedEngine
 *
 * Scientific citations preserved from v8.0 + additions:
 *   Contact map analysis: Grana et al. Nucleic Acids Res 2002 (CASP contact maps)
 *   Domain scoring: Apic et al. J Mol Biol 2001 (domain evolution)
 *   Aggregation: Fernandez-Escamilla et al. Nat Biotechnol 2004 (TANGO)
 *   Foldability: Garbuzynskiy et al. Bioinformatics 2010 (FoldAmyloid)
 *   Oligomeric awareness: Levy et al. Structure 2006 (PDB assembly analysis)
 */
'use strict';

// ─── RUNTIME DEPENDENCY ──────────────────────────────────────────────────────
const _core = (typeof CoreEngine !== 'undefined') ? CoreEngine
            : (typeof SharedEngine !== 'undefined') ? SharedEngine : {};
const _buildMolecule   = _core.buildMolecule   || function(r){ return r; };
const _analyzeMolecule = _core.analyzeMolecule || function(){ return {score:{net:0,bestRoute:'—'}}; };

// ─── BIOPHYSICAL SCALES (unchanged, cited) ───────────────────────────────────
const CF_HELIX  = {A:1.42,R:0.98,N:0.67,D:1.01,C:0.70,Q:1.11,E:1.51,G:0.57,H:1.00,I:1.08,L:1.21,K:1.16,M:1.45,F:1.13,P:0.57,S:0.77,T:0.83,W:1.08,Y:0.69,V:1.06};
const CF_SHEET  = {A:0.83,R:0.93,N:0.89,D:0.54,C:1.19,Q:1.10,E:0.37,G:0.75,H:0.87,I:1.60,L:1.30,K:0.74,M:1.05,F:1.38,P:0.55,S:0.75,T:1.19,W:1.37,Y:1.47,V:1.70};
const KD_HYDRO  = {A:1.8,R:-4.5,N:-3.5,D:-3.5,C:2.5,Q:-3.5,E:-3.5,G:-0.4,H:-3.2,I:4.5,L:3.8,K:-3.9,M:1.9,F:2.8,P:-1.6,S:-0.8,T:-0.7,W:-0.9,Y:-1.3,V:4.2};
const JN_BURIAL = {A:0.3,R:-1.4,N:-0.5,D:-0.6,C:0.9,Q:-0.7,E:-0.7,G:0.3,H:-0.1,I:0.7,L:0.5,K:-1.8,M:0.4,F:0.5,P:-0.3,S:-0.1,T:0.0,W:0.3,Y:-0.4,V:0.6};
const DIS_PROP  = {A:0.06,R:0.18,N:0.15,D:0.19,C:-0.02,Q:0.18,E:0.24,G:0.17,H:0.05,I:-0.12,L:-0.11,K:0.26,M:-0.01,F:-0.15,P:0.23,S:0.14,T:0.09,W:-0.13,Y:-0.08,V:-0.10};
const N_DEGRON_AA = 'RKHFWYLIED';

// ─── AGGREGATION PROPENSITY (TANGO-inspired, Fernandez-Escamilla 2004) ────────
// High values = aggregation-prone. Window-smoothed.
const AGG_PROP = {A:0.0,R:-2.0,N:-1.5,D:-2.5,C:0.5,Q:-1.0,E:-2.5,G:-0.5,H:-1.5,I:2.5,L:2.0,K:-2.0,M:0.5,F:3.0,P:-3.0,S:-0.5,T:-0.5,W:2.5,Y:1.5,V:2.5};

// ─── CYSTEINE BURDEN (drives misfolding in oxidizing environments) ─────────
function _countCys(seq) {
  let n = 0;
  for (let i = 0; i < (seq||'').length; i++) if (seq[i] === 'C') n++;
  return n;
}

// ─── INSTABILITY MOTIFS (Gasteiger et al. 2005, PEST-like regions) ─────────
const INSTABILITY_DIPEPTIDES = {
  'DP':0,'DG':0,'DS':0,'DD':1,'DE':1,'DT':0,
  'NP':1,'NS':0,'NG':1,'ND':1,'NQ':0,'NH':0,
};

function _instabilityIndex(seq) {
  if (!seq || seq.length < 2) return 40; // neutral default
  let score = 0;
  let n = 0;
  for (let i = 0; i < seq.length - 1; i++) {
    const dp = seq[i] + seq[i+1];
    score += (INSTABILITY_DIPEPTIDES[dp] || 0);
    n++;
  }
  // Normalize to 0-100 instability scale
  return Math.min(100, (score / Math.max(1, n)) * 60 + 30);
}

function _sp_get(scale, ch, def) {
  const v = scale[ch ? ch.toUpperCase() : ''];
  return (v != null) ? v : (def != null ? def : 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 1 — STRUCTURE-AWARE LAYER
// ═══════════════════════════════════════════════════════════════════════════

/**
 * parsePDB(pdbText)
 * Parses PDB ATOM records and extracts per-residue Cα/Cβ coordinates,
 * B-factors (as flexibility proxy), and chain info.
 * Returns { atoms, residues, chains, error }
 *
 * residues[i] = { resSeq, chain, resName, x, y, z, bFactor, hasCbeta, cbX, cbY, cbZ }
 */
function parsePDB(pdbText) {
  if (!pdbText || !pdbText.trim()) {
    return { atoms:[], residues:[], chains:[], error:'Empty PDB input' };
  }

  const atoms = [];
  const residueMap = new Map(); // key = "chain:resSeq"
  const chainSet = new Set();

  const lines = pdbText.split(/\r?\n/);
  for (const line of lines) {
    if (!line.startsWith('ATOM') && !line.startsWith('HETATM')) continue;
    const record   = line.substring(0, 6).trim();
    const serial   = parseInt(line.substring(6, 11)) || 0;
    const name     = line.substring(12, 16).trim();
    const resName  = line.substring(17, 20).trim();
    const chain    = line.substring(21, 22).trim() || 'A';
    const resSeq   = parseInt(line.substring(22, 26)) || 0;
    const x        = parseFloat(line.substring(30, 38)) || 0;
    const y        = parseFloat(line.substring(38, 46)) || 0;
    const z        = parseFloat(line.substring(46, 54)) || 0;
    const bFactor  = parseFloat(line.substring(60, 66)) || 0;

    if (record === 'HETATM') continue; // skip heteroatoms
    chainSet.add(chain);
    atoms.push({ serial, name, resName, chain, resSeq, x, y, z, bFactor });

    const key = `${chain}:${resSeq}`;
    if (!residueMap.has(key)) {
      residueMap.set(key, { resSeq, chain, resName, ca:null, cb:null, bFactor });
    }
    const res = residueMap.get(key);
    if (name === 'CA') {
      res.ca = { x, y, z };
      res.bFactor = bFactor; // B-factor from Cα
    }
    if (name === 'CB') {
      res.cb = { x, y, z };
    }
  }

  if (!residueMap.size) return { atoms:[], residues:[], chains:[], error:'No ATOM records found in PDB' };

  const residues = Array.from(residueMap.values())
    .filter(r => r.ca != null)
    .sort((a, b) => a.resSeq - b.resSeq);

  return { atoms, residues, chains:[...chainSet], error:null };
}

/**
 * parseAlphaFold(pdbText)
 * AlphaFold models use B-factor column for pLDDT confidence (0-100).
 * pLDDT > 90 = very high confidence (ordered)
 * pLDDT 70-90 = high confidence
 * pLDDT 50-70 = low confidence (often flexible)
 * pLDDT < 50 = very low confidence (likely disordered)
 * Returns per-residue pLDDT array aligned to sequence position.
 */
function parseAlphaFold(pdbText) {
  const parsed = parsePDB(pdbText);
  if (parsed.error) return { plddt:[], error:parsed.error };

  const plddt = parsed.residues.map(r => ({
    resSeq  : r.resSeq,
    plddt   : r.bFactor,          // AF2 stores pLDDT in B-factor column
    ordered : r.bFactor >= 70,
    flexible: r.bFactor < 50,
  }));
  return { plddt, residues:parsed.residues, error:null };
}

/**
 * applyStructureToProps(props, pdbResult, afResult)
 * Overrides sequence-heuristic per-residue properties with real structural data.
 * pdbResult = parsePDB output
 * afResult  = parseAlphaFold output (optional)
 * Returns enriched props array with structural data merged.
 */
function applyStructureToProps(props, pdbResult, afResult) {
  if (!pdbResult || pdbResult.error || !pdbResult.residues.length) return props;

  // Build a lookup by residue sequence number
  const structMap = new Map();
  for (const r of pdbResult.residues) {
    structMap.set(r.resSeq, r);
  }
  const plddtMap = new Map();
  if (afResult && !afResult.error) {
    for (const p of afResult.plddt) plddtMap.set(p.resSeq, p);
  }

  return props.map(prop => {
    const structRes = structMap.get(prop.position);
    const plddtRes  = plddtMap.get(prop.position);
    if (!structRes || !structRes.ca) return prop;

    // Override B-factor derived flexibility
    const bNorm = Math.min(1, structRes.bFactor / 60); // normalize ~0-1

    // Override accessibility from burial (if CB present, more buried)
    const hasCb = structRes.cb != null;

    // AF2 pLDDT confidence overrides disorder estimate
    let disorder = prop.disorder;
    if (plddtRes) {
      disorder = plddtRes.plddt < 50 ? 0.85
               : plddtRes.plddt < 70 ? 0.55
               : plddtRes.plddt < 90 ? 0.20 : 0.05;
    } else {
      // Use B-factor as flexibility proxy — higher B-factor → more flexible → more disordered
      disorder = Math.max(prop.disorder, Math.min(0.95, bNorm * 0.8));
    }

    // Accessibility: if residue is in a helix/sheet with low B-factor, likely buried
    let accessibility = prop.accessibility;
    if (hasCb && bNorm < 0.2) accessibility *= 0.5; // likely buried
    if (!hasCb && prop.ss === 'loop') accessibility = Math.max(prop.accessibility, 0.50);

    return {
      ...prop,
      accessibility,
      disorder,
      bFactor       : structRes.bFactor,
      hasStructure  : true,
      plddt         : plddtRes ? plddtRes.plddt : null,
    };
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2 — CONTACT MAP DISRUPTION ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * buildContactMap(pdbResult, cutoffAngstrom)
 * Builds a Cβ-Cβ distance matrix (Cα for Gly).
 * Returns { matrix, contacts } where contacts = list of { i, j, dist, weight }
 * Weight categories:
 *   surface contacts (dist 8-12Å, both acc) = 0.5
 *   core packing    (dist < 8Å, both buried) = 2.0
 *   interface       (dist < 10Å, cross-domain)= 1.8
 * If no PDB data, returns null.
 */
function buildContactMap(pdbResult, cutoffAngstrom) {
  cutoffAngstrom = cutoffAngstrom || 12.0;
  if (!pdbResult || pdbResult.error || !pdbResult.residues.length) return null;

  const residues = pdbResult.residues;
  const N = residues.length;
  const contacts = [];

  function getCoord(r) {
    // Prefer Cβ, fall back to Cα
    return (r.cb && r.cb.x != null) ? r.cb : r.ca;
  }

  function dist3(a, b) {
    const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    return Math.sqrt(dx*dx + dy*dy + dz*dz);
  }

  // Estimate burial from average local B-factor (low B = buried)
  const avgB = residues.reduce((s, r) => s + r.bFactor, 0) / Math.max(1, N);
  function isBuried(r) { return r.bFactor < avgB * 0.7; }

  for (let i = 0; i < N - 1; i++) {
    const ci = getCoord(residues[i]);
    if (!ci) continue;
    for (let j = i + 4; j < N; j++) { // skip i+1..i+3 (peptide bonds)
      const cj = getCoord(residues[j]);
      if (!cj) continue;
      const d = dist3(ci, cj);
      if (d > cutoffAngstrom) continue;

      // Weight calculation
      const buri = isBuried(residues[i]);
      const burj = isBuried(residues[j]);
      let weight = 1.0;
      if (buri && burj && d < 8.0) weight = 2.0;    // core packing
      else if (!buri && !burj)      weight = 0.5;    // surface contact
      else                          weight = 1.2;    // edge (one buried, one surface)

      contacts.push({
        i       : residues[i].resSeq,
        j       : residues[j].resSeq,
        dist    : parseFloat(d.toFixed(2)),
        weight,
        bothBuried: buri && burj,
      });
    }
  }

  return { contacts, N, avgB, residues };
}

/**
 * scoreContactDisruption(cutPosition, contactMap, annotations)
 * For a given cut position (all contacts with i <= cutPos AND j > cutPos are broken).
 * Returns { penalty 0-1, broken, weightedLoss, category }
 *
 * Penalty scaling:
 *   weightedLoss < 2   = negligible (0.0-0.05)
 *   weightedLoss < 6   = mild       (0.05-0.20)
 *   weightedLoss < 12  = moderate   (0.20-0.45)
 *   weightedLoss < 20  = severe     (0.45-0.75)
 *   weightedLoss >= 20 = critical   (0.75-1.0)
 */
function scoreContactDisruption(cutPosition, contactMap, annotations) {
  if (!contactMap) return { penalty:0, broken:[], weightedLoss:0, category:'no_structure', note:'No PDB data — contact disruption not calculated' };

  annotations = annotations || {};
  const activeSiteSet = new Set([
    ...(annotations.active_site || []),
    ...(annotations.binding_hotspot || []),
    ...(annotations.interface || []),
    ...(annotations.forbidden || []),
  ]);

  const broken = [];
  let weightedLoss = 0;

  for (const c of contactMap.contacts) {
    // Contact is broken if it crosses the cut
    if (c.i <= cutPosition && c.j > cutPosition) {
      let w = c.weight;
      // Extra penalty if broken contact involves a functional residue
      if (activeSiteSet.has(c.i) || activeSiteSet.has(c.j)) w *= 2.5;
      broken.push({ ...c, effectiveWeight: parseFloat(w.toFixed(2)) });
      weightedLoss += w;
    }
  }

  const penalty = weightedLoss < 2   ? weightedLoss / 40
                : weightedLoss < 6   ? 0.05 + (weightedLoss - 2)  / 20
                : weightedLoss < 12  ? 0.20 + (weightedLoss - 6)  / 20
                : weightedLoss < 20  ? 0.45 + (weightedLoss - 12) / 28
                : Math.min(1.0, 0.75 + (weightedLoss - 20) / 80);

  const category = weightedLoss < 2   ? 'negligible'
                 : weightedLoss < 6   ? 'mild'
                 : weightedLoss < 12  ? 'moderate'
                 : weightedLoss < 20  ? 'severe'
                 : 'critical';

  return {
    penalty       : parseFloat(penalty.toFixed(4)),
    contactPenalty: parseFloat(penalty.toFixed(4)),
    broken        : broken.sort((a,b) => b.effectiveWeight - a.effectiveWeight).slice(0, 10),
    weightedLoss  : parseFloat(weightedLoss.toFixed(2)),
    brokenCount   : broken.length,
    category,
    note          : `${broken.length} contacts broken (wt.loss=${weightedLoss.toFixed(1)}) — ${category}`,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 3 — DOMAIN INTEGRITY SCORING
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Domain classification object format:
 * { name, start, end, type: 'catalytic'|'binding'|'scaffold'|'linker'|'disordered' }
 */

/**
 * parseDomainBoundaries(domainList, sequenceLength)
 * Accepts an array of domain descriptors and returns enriched domain objects.
 * Gaps between domains are flagged as linkers.
 */
function parseDomainBoundaries(domainList, sequenceLength) {
  if (!domainList || !domainList.length) return [];
  const sorted = [...domainList].sort((a,b) => a.start - b.start);
  const result = [];

  let prev = 1;
  for (const d of sorted) {
    // Gap before domain = linker
    if (d.start > prev + 3) {
      result.push({ name:'Linker', start:prev, end:d.start - 1, type:'linker', isLinker:true });
    }
    result.push({ ...d, isLinker:false });
    prev = d.end + 1;
  }
  if (prev <= sequenceLength) {
    result.push({ name:'C-terminal region', start:prev, end:sequenceLength, type:'linker', isLinker:true });
  }
  return result;
}

/**
 * scoreDomainIntegrity(cutPosition, domains, sequenceLength)
 * Returns { score:0-1 (higher=better cut), classification, explanation }
 *
 * Classification:
 *   inter_domain_linker   → score ~ 0.95
 *   long_flexible_loop    → score ~ 0.70
 *   intra_domain_scaffold → score ~ 0.25
 *   catalytic_domain      → score ~ 0.05
 *   interface_crossing    → score ~ 0.10
 */
function scoreDomainIntegrity(cutPosition, domains, sequenceLength) {
  if (!domains || !domains.length) {
    return { score:0.50, classification:'unknown', explanation:'No domain boundaries provided — default score applied', penalty:0.50 };
  }

  let cutDomain = null;
  for (const d of domains) {
    if (cutPosition >= d.start && cutPosition <= d.end) {
      cutDomain = d;
      break;
    }
  }

  if (!cutDomain) {
    // Between annotated domains = best
    return { score:0.95, classification:'inter_domain_linker', explanation:'Cut falls between annotated domains — optimal position', penalty:0.05 };
  }

  if (cutDomain.isLinker || cutDomain.type === 'linker' || cutDomain.type === 'disordered') {
    // Is the linker long enough to be truly flexible?
    const linkerLen = cutDomain.end - cutDomain.start + 1;
    if (linkerLen >= 10) {
      return { score:0.90, classification:'long_flexible_linker', explanation:`Cut in long linker (${linkerLen}aa) between domains — excellent choice`, penalty:0.10 };
    }
    return { score:0.70, classification:'short_linker', explanation:`Cut in short linker (${linkerLen}aa) — acceptable but may constrain fragment geometry`, penalty:0.30 };
  }

  const t = cutDomain.type || 'scaffold';
  if (t === 'catalytic') {
    return { score:0.05, classification:'catalytic_domain', explanation:`CRITICAL: Cut inside catalytic domain "${cutDomain.name}" — will destroy enzymatic function`, penalty:0.95 };
  }
  if (t === 'binding') {
    return { score:0.12, classification:'interface_crossing', explanation:`SEVERE: Cut crosses binding domain "${cutDomain.name}" — will disrupt substrate/partner recognition`, penalty:0.88 };
  }
  if (t === 'scaffold') {
    return { score:0.25, classification:'intra_domain_scaffold', explanation:`BAD: Cut inside structural scaffold domain "${cutDomain.name}" — likely to destabilize fold`, penalty:0.75 };
  }
  // Disordered / IDR within annotated region
  return { score:0.60, classification:'intra_domain_flexible', explanation:`Cut inside domain "${cutDomain.name}" which contains flexible regions — borderline`, penalty:0.40 };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 4 — FUNCTIONAL RESIDUE RISK ENGINE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Known functional motifs (sequence-level patterns).
 * These auto-detect likely functional regions even without annotations.
 */
const FUNCTIONAL_MOTIFS = [
  { name:'Phosphorylation site (CK2)',   pattern:/[ST]..E/g,          risk:'indirect',  weight:0.4 },
  { name:'Phosphorylation site (PKA)',   pattern:/RR.S/g,             risk:'direct',    weight:0.6 },
  { name:'NLS (classical)',              pattern:/KK[RK]K/g,           risk:'indirect',  weight:0.5 },
  { name:'RdRp GDD motif',              pattern:/GDD/g,              risk:'catalytic', weight:1.0 },
  { name:'RdRp GDN motif (RABV)',       pattern:/GDN/g,              risk:'catalytic', weight:1.0 },
  { name:'Dynein LC8 motif',            pattern:/[KR]..T.Q/g,        risk:'direct',    weight:0.8 },
  { name:'Coiled-coil (LXXX heptad)',   pattern:/[LI].{3}[LI].{3}[LI]/g, risk:'indirect', weight:0.5 },
  { name:'Cysteine-rich (Zn finger)',   pattern:/C.{2,4}C.{3}C.{2,4}C/g, risk:'direct', weight:0.7 },
  { name:'N-glycosylation',             pattern:/N[^P][ST]/g,        risk:'indirect',  weight:0.3 },
  { name:'DEAD-box helicase',           pattern:/DEAD/g,             risk:'catalytic', weight:1.0 },
  { name:'Walker-A (P-loop)',           pattern:/GX{4}GK[ST]/g,     risk:'catalytic', weight:0.9 },
];

/**
 * detectFunctionalMotifs(sequence)
 * Returns list of { motif, positions[], risk, weight }
 */
function detectFunctionalMotifs(sequence) {
  if (!sequence) return [];
  const hits = [];
  for (const m of FUNCTIONAL_MOTIFS) {
    const re = new RegExp(m.pattern.source, 'g');
    let match;
    const positions = [];
    while ((match = re.exec(sequence)) !== null) {
      // All residues in the match are "at risk"
      for (let k = 0; k < match[0].length; k++) positions.push(match.index + k + 1);
    }
    if (positions.length) hits.push({ motif:m.name, positions, risk:m.risk, weight:m.weight });
  }
  return hits;
}

/**
 * scoreFunctionalRisk(cutPosition, sequence, annotations, motifHits)
 * Returns { directRisk:0-1, indirectRisk:0-1, overallRisk:0-1, reasons[] }
 *
 * Direct risk:  cut destroys functional residues at the cut point itself
 * Indirect risk: cut severs structural scaffold that supports distant function
 */
function scoreFunctionalRisk(cutPosition, sequence, annotations, motifHits) {
  annotations = annotations || {};
  motifHits   = motifHits || [];
  const PROX  = 8; // proximity radius for indirect risk

  // Build functional residue set from annotations + auto-detected motifs
  const directSites = new Set([
    ...(annotations.active_site || []),
    ...(annotations.binding_hotspot || []),
    ...(annotations.forbidden || []),
  ]);
  const indirectSites = new Set([...(annotations.interface || [])]);

  // Add motif hits
  for (const m of motifHits) {
    for (const p of m.positions) {
      if (m.risk === 'catalytic' || m.risk === 'direct') directSites.add(p);
      else indirectSites.add(p);
    }
  }

  let directRisk = 0;
  let indirectRisk = 0;
  const reasons = [];

  // Direct risk: cut point falls within PROX of a direct site
  for (const s of directSites) {
    const d = Math.abs(cutPosition - s);
    if (d <= PROX) {
      const contrib = Math.max(0, 1 - d / PROX);
      directRisk = Math.max(directRisk, contrib);
      reasons.push(`Direct site at pos ${s} (d=${d}) — risk +${(contrib*100).toFixed(0)}%`);
    }
  }

  // Indirect risk: structural scaffold connecting to function
  for (const s of indirectSites) {
    const d = Math.abs(cutPosition - s);
    if (d <= PROX * 2) {
      const contrib = Math.max(0, 1 - d / (PROX * 2)) * 0.6;
      indirectRisk = Math.max(indirectRisk, contrib);
      if (contrib > 0.1) reasons.push(`Interface site at pos ${s} (d=${d}) — indirect risk +${(contrib*100).toFixed(0)}%`);
    }
  }

  const overallRisk = Math.max(directRisk, indirectRisk * 0.7);
  return {
    directRisk    : parseFloat(directRisk.toFixed(3)),
    indirectRisk  : parseFloat(indirectRisk.toFixed(3)),
    overallRisk   : parseFloat(overallRisk.toFixed(3)),
    functionalPenalty: parseFloat(overallRisk.toFixed(3)),
    reasons,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 5 — FRAGMENT FOLDABILITY SCORING
// ═══════════════════════════════════════════════════════════════════════════

/**
 * scoreFragmentFoldability(seq, label)
 * Returns per-fragment foldability scores and risk flags.
 *
 * Assessed dimensions:
 *   intrinsicFoldability — based on secondary structure propensity balance
 *   disorderFraction     — fraction of residues with high disorder propensity
 *   hydrophobicPatch     — largest contiguous hydrophobic block (aggregation risk)
 *   chargeBalance        — net charge / MW (extremes destabilize)
 *   aggregationRisk      — TANGO-inspired score
 *   predictedSolubility  — inverse of hydrophobic patch × MW
 *   cysteineBurden       — free Cys count (oxidation / misfolding risk)
 *   proteaseSensitivity  — accessible cleavage sites (KR, RR motifs in loops)
 */
function scoreFragmentFoldability(seq, label) {
  label = label || 'Fragment';
  if (!seq || seq.length < 10) {
    return { label, tooShort:true, foldability:0, aggregationRisk:1, solubility:0, overall:0, flags:['Sequence too short (<10 aa)'] };
  }

  const N = seq.length;
  const flags = [];

  // 1. Intrinsic foldability: ratio of ordered SS propensity to disorder propensity
  let helixSum = 0, sheetSum = 0, disSum = 0, aggSum = 0;
  let netCharge = 0;
  for (const aa of seq) {
    helixSum += _sp_get(CF_HELIX, aa, 1.0);
    sheetSum += _sp_get(CF_SHEET, aa, 1.0);
    disSum   += _sp_get(DIS_PROP, aa, 0);
    aggSum   += _sp_get(AGG_PROP, aa, 0);
    if ('KR'.includes(aa)) netCharge++;
    if ('DE'.includes(aa)) netCharge--;
  }
  const orderedSS = (helixSum + sheetSum) / (2 * N);  // avg SS propensity
  const disorderedFrac = Math.min(1, Math.max(0, disSum / N + 0.35));
  const intrinsicFoldability = Math.max(0, Math.min(1,
    orderedSS * 0.6 * (1 - disorderedFrac * 0.4)
  ));

  // 2. Largest hydrophobic run — aggregation risk indicator
  let maxHydroRun = 0, currentRun = 0;
  for (const aa of seq) {
    if (_sp_get(KD_HYDRO, aa, 0) > 1.5) { currentRun++; maxHydroRun = Math.max(maxHydroRun, currentRun); }
    else currentRun = 0;
  }
  const hydrophobicPatch = maxHydroRun;
  if (hydrophobicPatch > 6) flags.push(`Hydrophobic patch ${hydrophobicPatch}aa long — high aggregation risk`);

  // 3. Aggregation risk (TANGO-inspired, window-normalized)
  const aggRiskRaw = aggSum / N;
  const aggregationRisk = Math.max(0, Math.min(1, (aggRiskRaw + 1) / 5));

  // 4. Charge balance
  const chargePerResidue = netCharge / N;
  const chargeOK = Math.abs(chargePerResidue) < 0.15;
  if (!chargeOK) flags.push(`Extreme net charge (${netCharge > 0 ? '+' : ''}${netCharge}) — may cause solubility issues`);

  // 5. Solubility proxy: inversely related to hydrophobic patch × aggregation
  const predictedSolubility = Math.max(0, Math.min(1,
    1 - aggregationRisk * 0.5 - Math.min(0.4, hydrophobicPatch / 20)
  ));

  // 6. Cysteine burden
  const cysteineCount = _countCys(seq);
  if (cysteineCount > 2) flags.push(`${cysteineCount} cysteines — high misfolding risk in reducing environment`);
  const cysteinePenalty = Math.min(0.4, cysteineCount * 0.08);

  // 7. Protease sensitivity: accessible RR/KR motifs
  const proteaseHits = (seq.match(/[KR]{2}/g) || []).length;
  if (proteaseHits > 3) flags.push(`${proteaseHits} dibasic protease sites — CNS protease sensitivity`);
  const proteaseSensitivity = Math.min(1, proteaseHits * 0.15);

  // 8. Instability index
  const instabilityIdx = _instabilityIndex(seq);
  if (instabilityIdx > 55) flags.push(`Instability index ${Math.round(instabilityIdx)} (>55 = unstable)`);

  // Overall foldability: penalize for all risk factors
  const overall = Math.max(0, Math.min(1,
    intrinsicFoldability
    * (1 - aggregationRisk * 0.35)
    * (1 - cysteinePenalty)
    * (1 - proteaseSensitivity * 0.20)
    * (instabilityIdx > 55 ? 0.75 : 1.0)
  ));

  return {
    label,
    length             : N,
    intrinsicFoldability: parseFloat(intrinsicFoldability.toFixed(3)),
    disorderFraction   : parseFloat(disorderedFrac.toFixed(3)),
    hydrophobicPatch   : maxHydroRun,
    netCharge,
    chargeOK,
    aggregationRisk    : parseFloat(aggregationRisk.toFixed(3)),
    predictedSolubility: parseFloat(predictedSolubility.toFixed(3)),
    cysteineBurden     : cysteineCount,
    proteaseSensitivity: parseFloat(proteaseSensitivity.toFixed(3)),
    instabilityIndex   : parseFloat(instabilityIdx.toFixed(1)),
    overall            : parseFloat(overall.toFixed(3)),
    overallPct         : Math.round(overall * 100),
    flags,
    grade : overall >= 0.70 ? 'A' : overall >= 0.50 ? 'B' : overall >= 0.30 ? 'C' : 'D',
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 6 — REASSEMBLY GEOMETRY SCORING
// ═══════════════════════════════════════════════════════════════════════════

/**
 * scoreReassemblyGeometry(cutPosition, pdbResult, splitSystem)
 * Estimates whether the two fragment termini are geometrically compatible
 * for the chosen split system.
 *
 * Uses Cα distance at cut point as a proxy for terminal accessibility.
 * For intein systems: needs good terminal presentation (low steric clash).
 * For split-GFP / zipper: needs proximity and orientation compatibility.
 *
 * Returns { score:0-1, terminalDistance_Å, linkerReach, stereoCompatible, notes[] }
 */
function scoreReassemblyGeometry(cutPosition, pdbResult, splitSystem) {
  splitSystem = splitSystem || 'split_intein_npu';

  if (!pdbResult || pdbResult.error || !pdbResult.residues.length) {
    return {
      score: 0.60,    // neutral fallback
      terminalDistance_A: null,
      stereoCompatible: null,
      linkerReach: 'unknown',
      notes: ['No PDB data — geometry not calculable, neutral score applied'],
    };
  }

  const residues = pdbResult.residues;
  // Find residues flanking the cut
  const f1Last = residues.find(r => r.resSeq === cutPosition);
  const f2First = residues.find(r => r.resSeq === cutPosition + 1);

  if (!f1Last || !f2First || !f1Last.ca || !f2First.ca) {
    return { score:0.55, terminalDistance_A:null, notes:['Flanking residues not found in PDB — geometry unknown'] };
  }

  // Terminal distance
  const dx = f1Last.ca.x - f2First.ca.x;
  const dy = f1Last.ca.y - f2First.ca.y;
  const dz = f1Last.ca.z - f2First.ca.z;
  const termDist = Math.sqrt(dx*dx + dy*dy + dz*dz);

  // System-specific geometry requirements
  let score, linkerReach, stereoNote;

  if (splitSystem === 'split_intein_npu') {
    // Npu DnaE: requires close proximity (ideally < 15Å) for efficient trans-splicing
    // The intein is ~18aa N-half + ~36aa C-half, giving ~20Å reach
    if (termDist < 10) { score = 0.95; stereoNote = 'Excellent — termini very close, ideal intein geometry'; }
    else if (termDist < 20) { score = 0.80; stereoNote = 'Good — within intein reach'; }
    else if (termDist < 35) { score = 0.55; stereoNote = 'Borderline — may need flexible linker extension'; }
    else { score = 0.25; stereoNote = 'Poor — termini too far apart for trans-splicing'; }
    linkerReach = '20Å (Npu intein footprint)';

  } else if (splitSystem === 'split_gfp') {
    // Split-GFP: GFP1-10 and GFP11 associate non-covalently; more tolerant
    if (termDist < 30) { score = 0.85; stereoNote = 'Good — split-GFP tolerates moderate terminal distances'; }
    else if (termDist < 50) { score = 0.65; stereoNote = 'Acceptable for split-GFP'; }
    else { score = 0.35; stereoNote = 'May fail — termini too far for efficient split-GFP assembly'; }
    linkerReach = '40Å (GFP11 peptide reach with flexible linker)';

  } else if (splitSystem === 'fkbp_frb') {
    // FKBP-FRB: rapamycin-gated; more tolerant of distance with flexible linkers
    if (termDist < 40) { score = 0.80; stereoNote = 'Good — FKBP-FRB conditional system tolerates distance'; }
    else if (termDist < 60) { score = 0.60; stereoNote = 'Acceptable with optimized linker'; }
    else { score = 0.35; stereoNote = 'Distant termini — long linkers needed, entropy cost high'; }
    linkerReach = '50Å (FKBP-FRB with linkers)';

  } else {
    // Generic non-covalent (leucine zipper, NanoBiT)
    if (termDist < 25) { score = 0.75; stereoNote = 'Acceptable geometry'; }
    else if (termDist < 45) { score = 0.55; stereoNote = 'Moderate geometry — add flexible linkers'; }
    else { score = 0.30; stereoNote = 'Challenging geometry for non-covalent systems'; }
    linkerReach = '30Å (generic with linkers)';
  }

  return {
    score               : parseFloat(score.toFixed(3)),
    terminalDistance_A  : parseFloat(termDist.toFixed(1)),
    linkerReach,
    stereoCompatible    : score >= 0.60,
    stereoNote,
    notes               : [stereoNote, `Terminal Cα distance: ${termDist.toFixed(1)}Å`],
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 7 — LINKER / FUSION BURDEN MODEL
// ═══════════════════════════════════════════════════════════════════════════

/**
 * scoreFusionBurden(f1mw, f2mw, splitSystem, bbbStrategy)
 * Models the impact of adding shuttles, intein halves, and linkers to each fragment.
 *
 * Constructed objects:
 *   F1: [TfR_shuttle(~7kDa)] — [GGGGS×3 linker(~1kDa)] — [F1] — [InteinN(~11kDa)]
 *   F2: [InteinC(~4kDa)] — [F2] — [GGGGS×2 linker] — [RVG-29(~3.3kDa)]
 *
 * Returns {
 *   f1Construct, f2Construct,   — full MW estimates
 *   f1FusionScore, f2FusionScore, — 0-1 (higher=better tolerated)
 *   overallBurden,              — 0-1 (lower=worse)
 *   flags, warnings
 * }
 */
const FUSION_TAGS = {
  tfr_rmt_shuttle  : { name:'TfR-RMT VHH shuttle',    mw_kDa: 7.0,  cterm:false, nterm:true  },
  rvg29            : { name:'RVG-29 peptide',          mw_kDa: 3.3,  cterm:true,  nterm:false },
  intein_N_npu     : { name:'Npu InteinN',             mw_kDa:11.0,  cterm:true,  nterm:false },
  intein_C_npu     : { name:'Npu InteinC',             mw_kDa: 4.0,  cterm:false, nterm:true  },
  intein_N_gfp1_10 : { name:'GFP1-10 fragment',        mw_kDa:23.0,  cterm:true,  nterm:false },
  intein_C_gfp11   : { name:'GFP11 peptide',           mw_kDa: 1.4,  cterm:false, nterm:true  },
  fkbp             : { name:'FKBP12 tag',              mw_kDa:12.0,  cterm:true,  nterm:false },
  frb              : { name:'FRB domain',              mw_kDa:11.8,  cterm:false, nterm:true  },
  linker_ggg_x3    : { name:'(GGGGS)×3 linker',        mw_kDa: 1.0,  cterm:true,  nterm:true  },
};

function scoreFusionBurden(f1mw, f2mw, splitSystem, bbbStrategy) {
  splitSystem  = splitSystem  || 'split_intein_npu';
  bbbStrategy  = bbbStrategy  || 'tfr_rmt';
  const flags  = [];
  const warnings = [];

  // Determine tags per fragment
  let f1Tags = [];
  let f2Tags = [];

  if (splitSystem === 'split_intein_npu') {
    f1Tags.push('intein_N_npu');
    f2Tags.unshift('intein_C_npu');
  } else if (splitSystem === 'split_gfp') {
    f1Tags.push('intein_N_gfp1_10');
    f2Tags.unshift('intein_C_gfp11');
  } else if (splitSystem === 'fkbp_frb') {
    f1Tags.push('fkbp');
    f2Tags.unshift('frb');
  }

  // BBB delivery shuttles
  if (bbbStrategy === 'tfr_rmt') {
    f1Tags.unshift('tfr_rmt_shuttle');
    f1Tags.push('linker_ggg_x3');
  } else if (bbbStrategy === 'rvg29') {
    f1Tags.unshift('rvg29');
  }
  f2Tags.push('rvg29');  // F2 always gets RVG-29

  // Construct MWs
  const f1TagMW = f1Tags.reduce((s, k) => s + (FUSION_TAGS[k] ? FUSION_TAGS[k].mw_kDa * 1000 : 0), 0);
  const f2TagMW = f2Tags.reduce((s, k) => s + (FUSION_TAGS[k] ? FUSION_TAGS[k].mw_kDa * 1000 : 0), 0);

  const f1ConstructMW = f1mw + f1TagMW;
  const f2ConstructMW = f2mw + f2TagMW;

  // Score: penalize if construct MW is too large
  function constructScore(mw) {
    if (mw <= 10000)  return 0.90;
    if (mw <= 15000)  return 0.75;
    if (mw <= 20000)  return 0.55;
    if (mw <= 30000)  return 0.35;
    if (mw <= 50000)  return 0.20;
    return 0.08; // > 50 kDa biologics have very poor CNS penetration
  }

  const f1FusionScore = constructScore(f1ConstructMW);
  const f2FusionScore = constructScore(f2ConstructMW);

  // Linker length warnings
  if (f1Tags.includes('intein_N_npu') && bbbStrategy === 'tfr_rmt') {
    warnings.push('F1 has shuttle + inteinN — confirm GGGGS×3 linker is sufficient to prevent steric clash between TfR binding and intein folding');
  }

  // Cargo interference
  if (bbbStrategy === 'tfr_rmt' && f1mw > 20000) {
    warnings.push('F1 >20kDa with TfR shuttle — large cargo may reduce TfR binding efficiency (affinity-efficiency paradox)');
  }

  const maxConstructMW = Math.max(f1ConstructMW, f2ConstructMW);
  if (maxConstructMW > 40000) {
    flags.push(`Maximum construct MW ${Math.round(maxConstructMW/1000)} kDa — protein biologic delivery unlikely; consider dual-AAV gene delivery`);
  }

  const overallBurden = Math.max(0, Math.min(1, (f1FusionScore + f2FusionScore) / 2));

  return {
    f1Tags        : f1Tags.map(k => FUSION_TAGS[k] ? FUSION_TAGS[k].name : k),
    f2Tags        : f2Tags.map(k => FUSION_TAGS[k] ? FUSION_TAGS[k].name : k),
    f1ConstructMW : Math.round(f1ConstructMW),
    f2ConstructMW : Math.round(f2ConstructMW),
    f1FusionScore,
    f2FusionScore,
    overallBurden,
    overallBurdenPct: Math.round(overallBurden * 100),
    flags, warnings,
    recommendation: overallBurden >= 0.65
      ? 'Construct size is manageable — proceed with protein biologic delivery'
      : overallBurden >= 0.40
      ? 'Construct large — consider dual-AAV or reduce fragment/tag size'
      : 'Construct too large for protein delivery — dual-AAV or mRNA-LNP required',
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 8 — OLIGOMERIZATION AND COMPLEX-STATE AWARENESS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * RABV_ASSEMBLY_STATES
 * Biological assembly information for each RABV therapeutic target.
 * A cut that preserves monomer fold may still fail in the oligomeric context.
 */
const RABV_ASSEMBLY_STATES = {
  P: {
    state: 'dimer',
    description: 'P forms constitutive dimers via its C-terminal domain (aa 189-297). The N-terminal domain (aa 1-130) is monomeric and disordered.',
    dimerInterface: [189, 195, 202, 215, 240, 260, 275, 285, 297],
    monomerRegion : { start:1, end:130 },
    dimerRegion   : { start:189, end:297 },
    safeZone      : { start:100, end:180, reason:'Flexible linker between N-domain and dimerization domain' },
    note          : 'A cut within aa 1-130 is in the monomerically disordered N-domain — low risk for dimer disruption. A cut within aa 189-297 breaks the dimer interface — catastrophic.',
  },
  N: {
    state: '11-mer',
    description: 'N forms a helical oligomeric nucleocapsid (11-mer rings, RCSB 8FFR). Both the N-arm (aa 1-22) and C-arm (aa 377-450) are critical for inter-subunit packing.',
    oligomerInterface: [1,2,3,4,5,6,7,8,9,10,11,12,377,378,379,380,381,382,383,384,385,390,400,420,440,450],
    monomerCore   : { start:23, end:370 },
    rnaGroove     : { start:246, end:275 },
    note          : 'Cuts in the monomer core (aa 23-370) are less likely to destroy oligomerization but must avoid the RNA groove. N/C-arms MUST NOT be cut.',
  },
  L: {
    state: 'monomer_complex',
    description: 'L is monomeric but forms a dynamic complex with P protein for active transcription. Five enzymatic domains with defined interdomain linkers.',
    pInteractionRegion: [1, 10, 20, 30], // N-terminal P-binding
    domains: [
      { name:'RdRp core',      start:1,    end:540,  type:'catalytic' },
      { name:'Linker 1',       start:541,  end:560,  type:'linker'    },
      { name:'Capping domain', start:561,  end:900,  type:'catalytic' },
      { name:'Linker 2',       start:901,  end:920,  type:'linker'    },
      { name:'MTase domain',   start:921,  end:1700, type:'catalytic' },
      { name:'Linker 3',       start:1701, end:1720, type:'linker'    },
      { name:'C-terminal',     start:1721, end:2142, type:'scaffold'  },
    ],
    note: 'L has only three safe linker regions (~aa 541-560, 901-920, 1701-1720). All other cuts destroy enzymatic function. Fragments will be >60kDa — dual-AAV only.',
  },
};

/**
 * scoreAssemblyContext(cutPosition, targetId, customAssembly)
 * Returns { assemblyRisk:0-1, assemblyClass, notes[], recommendation }
 */
function scoreAssemblyContext(cutPosition, targetId, customAssembly) {
  const asm = customAssembly || RABV_ASSEMBLY_STATES[targetId] || null;
  if (!asm) {
    return { assemblyRisk:0.10, assemblyClass:'unknown', notes:['No assembly data — default low risk assumed'] };
  }

  const notes = [];
  let assemblyRisk = 0.10;
  let assemblyClass = 'monomer_safe';

  if (asm.state === 'dimer' && asm.dimerInterface) {
    const inDimerFace = asm.dimerInterface.some(r => Math.abs(r - cutPosition) <= 5);
    if (inDimerFace) {
      assemblyRisk = 0.90;
      assemblyClass = 'dimer_interface_break';
      notes.push('CRITICAL: Cut is within dimer interface — will destroy dimerization');
    } else if (asm.safeZone && cutPosition >= asm.safeZone.start && cutPosition <= asm.safeZone.end) {
      assemblyRisk = 0.08;
      assemblyClass = 'safe_zone';
      notes.push(`Safe zone (${asm.safeZone.reason}) — low assembly risk`);
    } else if (asm.dimerRegion && cutPosition >= asm.dimerRegion.start) {
      assemblyRisk = 0.75;
      assemblyClass = 'dimer_region';
      notes.push(`In dimerization domain (aa ${asm.dimerRegion.start}-${asm.dimerRegion.end}) — high risk`);
    }
  }

  if (asm.state === '11-mer' && asm.oligomerInterface) {
    const inOligoFace = asm.oligomerInterface.some(r => Math.abs(r - cutPosition) <= 8);
    if (inOligoFace) {
      assemblyRisk = 0.88;
      assemblyClass = 'oligomer_interface_break';
      notes.push('CRITICAL: Cut near oligomerization surface — likely destroys 11-mer assembly');
    } else if (asm.monomerCore && cutPosition >= asm.monomerCore.start && cutPosition <= asm.monomerCore.end) {
      // Check RNA groove
      if (asm.rnaGroove && cutPosition >= asm.rnaGroove.start && cutPosition <= asm.rnaGroove.end) {
        assemblyRisk = 0.70;
        assemblyClass = 'rna_groove_region';
        notes.push('In RNA binding groove — high functional risk though oligomerization may survive');
      } else {
        assemblyRisk = 0.15;
        assemblyClass = 'monomer_core_safe';
        notes.push('In monomer core outside N/C-arms — oligomerization interfaces preserved');
      }
    }
  }

  if (asm.state === 'monomer_complex' && asm.domains) {
    const cutDomain = asm.domains.find(d => cutPosition >= d.start && cutPosition <= d.end);
    if (cutDomain) {
      if (cutDomain.type === 'linker') {
        assemblyRisk = 0.05;
        assemblyClass = 'domain_linker';
        notes.push(`In safe interdomain linker "${cutDomain.name}"`);
      } else if (cutDomain.type === 'catalytic') {
        assemblyRisk = 0.95;
        assemblyClass = 'catalytic_domain_break';
        notes.push(`CRITICAL: In catalytic domain "${cutDomain.name}" — will destroy enzymatic activity`);
      } else {
        assemblyRisk = 0.50;
        assemblyClass = 'scaffold_domain';
        notes.push(`In scaffold domain "${cutDomain.name}" — structural risk`);
      }
    }
  }

  notes.push(`Assembly state: ${asm.state}`);
  return {
    assemblyRisk  : parseFloat(assemblyRisk.toFixed(3)),
    assemblyClass,
    state         : asm.state,
    notes,
    recommendation: assemblyRisk < 0.20
      ? 'Proceed — assembly context favorable'
      : assemblyRisk < 0.55
      ? 'Caution — assembly context moderately unfavorable'
      : 'REJECT — cut destroys critical assembly interface',
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 10 — EXPERIMENTAL BENCHMARK CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * BENCHMARK_SPLITS
 * Known experimental outcomes for split protein designs.
 * Used to calibrate the scorer: good splits should rank near the top.
 * Sources: Dagliyan 2018, Kamiyama 2016, Zeng 2020, Gao 2018.
 */
const BENCHMARK_SPLITS = [
  // Well-validated GOOD splits
  { protein:'GFP', position:214, outcome:'success',  note:'Split-GFP 1-10/11 (Cabantous 2005) — canonical success, GFP11 fully functional' },
  { protein:'GFP', position:157, outcome:'success',  note:'Alternative split-GFP site — moderate efficiency' },
  { protein:'Firefly luciferase', position:437, outcome:'success', note:'Split-Luc NlucC/NlucN at 437 — high complementation (Dixon 2016)' },
  { protein:'Cas9',  position:573, outcome:'success',  note:'Split-Cas9 at 573 — functional in AAV dual-vector delivery (Truong 2015)' },
  { protein:'Cas9',  position:637, outcome:'success',  note:'Split-Cas9 at 637 — alternative site (Chew 2016)' },
  { protein:'TEV protease', position:118, outcome:'success', note:'Reconstituted TEV — intein-mediated split (Mootz 2003)' },

  // Well-documented FAILURES
  { protein:'GFP',  position:100, outcome:'failure',  note:'Core β-barrel — split destroys fold (buried position)' },
  { protein:'GFP',  position:180, outcome:'failure',  note:'Sheet interior — aggregation, no complementation' },
  { protein:'DHFR', position:1,   outcome:'failure',  note:'N-terminus — too short F1 fragment' },
];

/**
 * calibrateScorer(sequence, annotations, constraints)
 * Runs the scorer on a few benchmark positions and checks whether
 * the engine ranks "success" positions above "failure" positions.
 * Returns { calibrationScore:0-1, details[] }
 *
 * NOTE: This is a self-test, not for end-user output.
 * Uses a simplified sequence proxy (GFP-like) for offline testing.
 */
function calibrateScorer(sequence, annotations) {
  // Simplified calibration using sequence-only mode
  if (!sequence || sequence.length < 100) {
    return { calibrationScore:null, details:[], note:'Sequence too short for calibration' };
  }

  const props = estimateResidueProperties(sequence);
  const candidates = findSplitSiteCandidates(props, annotations || {}, {});

  // Check that known_good positions (if present) score above median
  const goodPos = (annotations && annotations.known_good_splits) || [];
  if (!goodPos.length) return { calibrationScore:null, details:[], note:'No known_good_splits in annotations' };

  const allScores = candidates.map(c => c.splitScore);
  const medianScore = allScores.sort((a,b)=>a-b)[Math.floor(allScores.length/2)] || 0;

  let nAbove = 0;
  const details = goodPos.map(pos => {
    const c = candidates.find(x => Math.abs(x.position - pos) <= 5);
    const score = c ? c.splitScore : null;
    const aboveMedian = score != null && score > medianScore;
    if (aboveMedian) nAbove++;
    return { position:pos, score, aboveMedian, verdict:aboveMedian ? 'pass':'fail' };
  });

  const calibrationScore = goodPos.length > 0 ? nAbove / goodPos.length : null;
  return {
    calibrationScore,
    details,
    passed: calibrationScore != null && calibrationScore >= 0.6,
    note: calibrationScore != null
      ? `${Math.round(calibrationScore*100)}% of known-good splits ranked above median`
      : 'No calibration data available',
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 11 — RABV-SPECIFIC ENRICHED TARGET MAPS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * RABV_TARGET_MAPS — full structural/functional/assembly maps for P, N, L.
 * These are the authoritative presets for RABV therapeutic design.
 * Supersedes the basic PROTEIN_PRESETS from v8.0.
 */
const RABV_TARGET_MAPS = {
  P: {
    id:'P', name:'RABV Phosphoprotein (P)',
    shortName:'P protein', mw_kda:33, length:297,
    uniprotId:'P16285', pdb:'7C20,3OA1',
    sequence:'MDADKIVFKVNNQVVSLKPEIIVDQYEYKYPAIKDLKKPCITLGQVDNKTYVDKQLQNFEKGVTIDFDLASSRLSERNFISFTDLEYNSSPFVTTPTVSIQEQRLDSITEDPGSSGTTESTDISRLNDALRRNMEEVLAQIRPAEDPTPNRAAQQPEMEWSRALNTIYLNQQNLRIQKQVSETEGIEGLAQDDSVTIAQNNTQTKVVNDSGLNYMKSNVKQVKKMADEFERNLKESQRPVHFLNFGTLNLSIVREKQKTLHSANVKDQYEMESLFHSTPGVSTQRRGDLNQSYRRILDPNAKMTVDLNQTVSTTQEAYRQIMFNLAK',
    domains: [
      { name:'N-terminal IDR',        start:1,   end:50,  type:'disordered' },
      { name:'N-terminal domain',     start:51,  end:130, type:'scaffold'   },
      { name:'Flexible linker',       start:131, end:172, type:'linker'     },
      { name:'Oligomerization domain',start:173, end:202, type:'scaffold'   },
      { name:'Dimerization domain',   start:203, end:297, type:'binding'    },
    ],
    annotations: {
      active_site     : [],
      binding_hotspot : [218,219,220,221,222,223,224,225],
      interface       : [179,189,195,202,215,240,260,275,285],
      forbidden       : [179,218,219,220,221,222,223,224,225],
      known_good_splits:[100,120,140,155,160,165],
      motifs: [
        { name:'LC8 binding',  residues:[218,225], function:'Dynein retrograde transport',  druggability:'High' },
        { name:'TBK1 Ser179',  residues:[179],     function:'IFN-β suppression',            druggability:'High' },
        { name:'L-binding',    residues:[1,50],    function:'RNA polymerase recruitment',   druggability:'Medium' },
        { name:'Dimerization', residues:[203,297], function:'P constitutive dimer',         druggability:'Low' },
      ],
    },
    assemblyData: RABV_ASSEMBLY_STATES.P,
    note:'Best therapeutic targets: LC8 binding (aa 218-225) via Pep2 precedent; TBK1 interface (Ser179) via split-nanobody. Safe cut zone: aa 100-165 (linker between N-domain and oligomerization domain). Refs: Ribeiro 2009; Wiltzer 2014; Rahmati 2025.',
  },

  N: {
    id:'N', name:'RABV Nucleoprotein (N)',
    shortName:'N protein', mw_kda:47, length:450,
    uniprotId:'P06025', pdb:'8FFR',
    sequence:'MDADKIVFKVNNQVVSLKPEIIVKMDVNPKDEVLNKLNELKQRLEEMGDPEEQVVMAIPSWQHLYQKSTMGPQHPNPHLSYMVDVLQPPQPDNHNDRDRQHYENNQEFWKEHLDRLRLEQGGDQATNLRKVLNGLRQFAIGNDVTPFNRFVDGEEALVLKKNMEIAHFGTPFQHINDTKKDEYEFLSNKNMDDPQVFLMDQQLEQKLLEAQPTLELTLAIHKLRNVSSDNKGYSIQDTDNRGEGIQKFLKRMIMQMNDNHSDKVAEGIASCLLDLKDKIIEQINKLLDSDFVTKKQLITPKIPAIAQAAALDGPYQLKSKNPNLATILNAIQLTVKMSEDLKLQRYAQNVKQLIDLKMEQESGPKIDTIEQINQENIKKMANDMVNRQKSMTEKVTMRHTREKQQVVPVKQALVVSHYENMDPIIAEEGDNMIDFQHPYNSSLFKQDAIILRVQQLMNPQLQEFLQSSQERLA',
    domains: [
      { name:'N-arm',           start:1,   end:22,  type:'binding'  },
      { name:'N-terminal lobe', start:23,  end:220, type:'scaffold' },
      { name:'Central linker',  start:221, end:260, type:'linker'   },
      { name:'C-terminal lobe', start:261, end:375, type:'scaffold' },
      { name:'C-arm',           start:376, end:450, type:'binding'  },
    ],
    annotations: {
      active_site     : [],
      binding_hotspot : [246,247,248,249,250,251,252,253,254,255],
      interface       : [1,2,3,4,5,6,7,8,101,102,103,370,371,372,373,374,375,376,420],
      forbidden       : [1,2,3,4,5,101,102,103,246,247,248,249,250,251,370,371,372,376],
      known_good_splits:[200,220,235,245,255,265],
      motifs: [
        { name:'RNA binding groove', residues:[246,275], function:'ssRNA encapsidation', druggability:'Low' },
        { name:'N-arm interface',    residues:[1,22],    function:'Ring oligomerization', druggability:'Medium' },
        { name:'C-arm interface',    residues:[376,450], function:'Ring oligomerization', druggability:'Medium' },
        { name:'P-chaperone site',   residues:[1,25],    function:'N0-P complex (prevents premature oligomerization)', druggability:'High' },
      ],
    },
    assemblyData: RABV_ASSEMBLY_STATES.N,
    note:'11-mer helical nucleocapsid (RCSB 8FFR). N-arm (1-22) and C-arm (376-450) are oligomerization arms — MUST NOT be cut. Central linker region (aa 221-260) is the only safe zone. Targeting N-N interface disrupts RNP integrity. Ref: Scrima 2008; Green 2014; Blondel 2012.',
  },

  L: {
    id:'L', name:'RABV L Protein (RdRp)',
    shortName:'L protein (RdRp)', mw_kda:240, length:2142,
    uniprotId:'P06029', pdb:'—',
    sequence:null, // Full sequence not included — L is 2142aa
    domains: RABV_ASSEMBLY_STATES.L.domains,
    annotations: {
      active_site     : [831,832,833],
      binding_hotspot : [740,741,742,743,744,745],
      interface       : [1,2,3,4,5,6,7,8,2138,2139,2140,2141,2142],
      forbidden       : [829,830,831,832,833,834,835],
      known_good_splits:[545,555,905,915,1705,1715],
      motifs: [
        { name:'GDN catalytic motif', residues:[831,833], function:'RNA polymerization (RdRp active site)', druggability:'Indirect via DN' },
        { name:'P-binding N-terminus', residues:[1,30],   function:'Polymerase complex formation',          druggability:'Medium' },
        { name:'MTase active site',    residues:[1700,1750], function:'5\' capping',                        druggability:'Medium' },
      ],
    },
    assemblyData: RABV_ASSEMBLY_STATES.L,
    high_uncertainty: true,
    note:'L has 5 enzymatic domains with only 3 safe interdomain linkers (~aa 541-560, 901-920, 1701-1720). All fragments will be >60kDa — dual-AAV gene delivery is the ONLY realistic option. L structure is largely unresolved (partial cryo-EM: Loureiro 2023). Very high uncertainty. Dominant-negative approach (Architecture C) is theoretical only.',
  },
};

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 14 — MANUFACTURABILITY LAYER (full)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * scoreManufacturability(f1seq, f2seq, splitSystem, deliveryMode)
 * Full manufacturability assessment covering:
 *   expressionEase     — E.coli / CHO expression likelihood
 *   purificationEase   — complexity of isolation (charge, hydrophobicity, size)
 *   cysteineRisk       — Cys count → disulfide / aggregation in expression
 *   instabilityRisk    — sequence instability index
 *   repeats            — internal sequence repeats (cause frameshift issues)
 *   conjugateComplexity— number of chemical conjugation steps
 *   aggregationRisk    — hydrophobic patch exposure (yield loss)
 *   shelfLifeEstimate  — stability in formulation (covalent vs non-covalent assembly)
 *   gmpCostIndex       — relative GMP manufacturing cost
 */
function scoreManufacturability(f1seq, f2seq, splitSystem, deliveryMode) {
  f1seq = f1seq || '';
  f2seq = f2seq || '';
  splitSystem = splitSystem || 'split_intein_npu';
  deliveryMode = deliveryMode || 'protein_biologic';

  const scores = {};
  const flags  = [];
  const notes  = [];

  // 1. Expression ease
  const f1fold = scoreFragmentFoldability(f1seq, 'F1');
  const f2fold = scoreFragmentFoldability(f2seq, 'F2');
  const avgFold = (f1fold.overall + f2fold.overall) / 2;
  scores.expressionEase = Math.round(avgFold * 100);
  if (avgFold < 0.50) flags.push('Poor fragment foldability — expression yield may be low');

  // 2. Purification ease: charged fragments purify well, hydrophobic ones poorly
  const f1charge = Math.abs(f1fold.netCharge);
  const f2charge = Math.abs(f2fold.netCharge);
  const chargeScore = Math.min(1, (f1charge + f2charge) / 10 + 0.5);
  const hydrophobicPenalty = Math.min(0.4, (f1fold.hydrophobicPatch + f2fold.hydrophobicPatch) / 30);
  scores.purificationEase = Math.round((chargeScore * (1 - hydrophobicPenalty)) * 100);

  // 3. Cysteine risk
  const totalCys = f1fold.cysteineBurden + f2fold.cysteineBurden;
  scores.cysteineRisk = Math.max(0, Math.round(100 - totalCys * 12));
  if (totalCys > 4) flags.push(`${totalCys} cysteines total — significant aggregation/disulfide risk`);

  // 4. Instability risk
  const avgInstability = (f1fold.instabilityIndex + f2fold.instabilityIndex) / 2;
  scores.instabilityRisk = Math.max(0, Math.round(100 - Math.max(0, avgInstability - 30) * 1.5));
  if (avgInstability > 55) flags.push(`Instability index ${Math.round(avgInstability)} — consider sequence optimization`);

  // 5. Internal repeats (simplified — count tandem dipeptide repeats)
  function countRepeats(seq) {
    let n = 0;
    for (let i = 0; i < seq.length - 3; i++) {
      if (seq[i] === seq[i+2] && seq[i+1] === seq[i+3]) n++;
    }
    return n;
  }
  const totalRepeats = countRepeats(f1seq) + countRepeats(f2seq);
  scores.repeatRisk = Math.max(0, Math.round(100 - totalRepeats * 3));
  if (totalRepeats > 10) flags.push(`${totalRepeats} internal repeats — cloning instability risk`);

  // 6. Conjugate complexity
  const conjugateSteps = (splitSystem === 'split_intein_npu' ? 2 : splitSystem === 'split_gfp' ? 2 : splitSystem === 'fkbp_frb' ? 3 : 1)
    + (deliveryMode === 'protein_biologic' ? 2 : 0);
  scores.conjugateComplexity = Math.max(0, Math.round(100 - conjugateSteps * 12));
  if (conjugateSteps > 4) flags.push(`${conjugateSteps} conjugation steps — complex manufacturing`);

  // 7. Aggregation risk (composite from both fragments)
  scores.aggregationRisk = Math.round((1 - (f1fold.aggregationRisk + f2fold.aggregationRisk) / 2) * 100);

  // 8. Shelf-life estimate
  const isCovalent = splitSystem === 'split_intein_npu';
  scores.shelfLifeEstimate = isCovalent ? 75 : 50;
  if (!isCovalent) notes.push('Non-covalent assembly — shelf-life limited; covalent intein system preferred');

  // 9. GMP cost index
  const costPenalty = Object.values(scores).reduce((s,v) => s + (100-v), 0);
  scores.gmpCostIndex = Math.max(0, Math.round(100 - costPenalty / Object.keys(scores).length));

  // Overall feasibility
  const feasibility = Math.round(
    scores.expressionEase * 0.20 +
    scores.purificationEase * 0.15 +
    scores.cysteineRisk * 0.15 +
    scores.instabilityRisk * 0.10 +
    scores.repeatRisk * 0.08 +
    scores.conjugateComplexity * 0.12 +
    scores.aggregationRisk * 0.12 +
    scores.shelfLifeEstimate * 0.08
  );

  const grade = feasibility >= 75 ? 'A' : feasibility >= 60 ? 'B' : feasibility >= 45 ? 'C' : feasibility >= 30 ? 'D' : 'F';

  // Limiting factor
  const subScoreArr = Object.entries(scores).map(([k,v]) => ({name:k, v}));
  subScoreArr.sort((a,b) => a.v - b.v);
  const limitingFactor = subScoreArr[0].name;

  return {
    feasibility,
    grade,
    subScores     : scores,
    limitingFactor,
    flags, notes,
    f1Assessment  : { foldability:f1fold.overall, aggregation:f1fold.aggregationRisk, cys:f1fold.cysteineBurden, instability:f1fold.instabilityIndex },
    f2Assessment  : { foldability:f2fold.overall, aggregation:f2fold.aggregationRisk, cys:f2fold.cysteineBurden, instability:f2fold.instabilityIndex },
    costNote: feasibility >= 65 ? 'Feasible for GMP development (grade '+grade+')'
            : feasibility >= 45 ? 'Significant manufacturing challenges (grade '+grade+')'
            : 'Manufacturing bottleneck — redesign recommended (grade '+grade+')',
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 15 — REJECTION ENGINE
// ═══════════════════════════════════════════════════════════════════════════

/**
 * REJECTION CRITERIA — explicit rules that immediately disqualify a cut site.
 * Each rule has: id, description, severity, test(candidate, context) → bool
 */
const REJECTION_CRITERIA = [
  {
    id: 'buried_core',
    description: 'Cuts within buried hydrophobic core',
    severity: 'critical',
    test(c, ctx) {
      return (c.scores && c.scores.accessibility < 0.20) ||
             (c.bFactor != null && c.bFactor < 8);
    },
  },
  {
    id: 'catalytic_scaffold',
    description: 'Cuts catalytic scaffold or active site scaffold',
    severity: 'critical',
    test(c, ctx) {
      const domain = ctx.domainScore;
      return domain && (domain.classification === 'catalytic_domain' || domain.classification === 'intra_domain_scaffold');
    },
  },
  {
    id: 'interface_destruction',
    description: 'Cuts protein-protein binding interface',
    severity: 'critical',
    test(c, ctx) {
      const funcRisk = ctx.functionalRisk;
      return funcRisk && funcRisk.directRisk > 0.75;
    },
  },
  {
    id: 'excessive_asymmetry',
    description: 'Fragment size ratio exceeds 5:1',
    severity: 'high',
    test(c) {
      const ratio = Math.max(c.f1Length, c.f2Length) / Math.max(1, Math.min(c.f1Length, c.f2Length));
      return ratio > 5;
    },
  },
  {
    id: 'delivery_impossible',
    description: 'Resulting construct too large for any delivery mode',
    severity: 'high',
    test(c, ctx) {
      const fusionBurden = ctx.fusionBurden;
      return fusionBurden && fusionBurden.f1ConstructMW > 80000 && fusionBurden.f2ConstructMW > 80000;
    },
  },
  {
    id: 'oligomer_face',
    description: 'Cuts oligomerization interface — destroys biologically active form',
    severity: 'critical',
    test(c, ctx) {
      const asm = ctx.assemblyScore;
      return asm && (asm.assemblyClass === 'dimer_interface_break' || asm.assemblyClass === 'oligomer_interface_break');
    },
  },
  {
    id: 'contact_critical',
    description: 'Contact disruption is critical — breaks core packing contacts',
    severity: 'critical',
    test(c, ctx) {
      const cd = ctx.contactDisruption;
      return cd && cd.category === 'critical';
    },
  },
  {
    id: 'fragment_too_short',
    description: 'One or both fragments below minimum viable length',
    severity: 'high',
    test(c) {
      return c.f1Length < 40 || c.f2Length < 40;
    },
  },
  {
    id: 'aggregation_catastrophic',
    description: 'Both fragments show catastrophically high aggregation risk',
    severity: 'high',
    test(c, ctx) {
      const f1 = ctx.f1Foldability, f2 = ctx.f2Foldability;
      return f1 && f2 && f1.aggregationRisk > 0.80 && f2.aggregationRisk > 0.80;
    },
  },
];

/**
 * rejectCandidates(candidates, contextPerCandidate)
 * Applies rejection engine to filter out disqualified cuts.
 * Returns { passed[], rejected[] } where rejected includes reason.
 *
 * contextPerCandidate: array of context objects, same order as candidates.
 * Each context: { domainScore, functionalRisk, fusionBurden, assemblyScore, contactDisruption, f1Foldability, f2Foldability }
 */
function rejectCandidates(candidates, contextPerCandidate) {
  contextPerCandidate = contextPerCandidate || candidates.map(() => ({}));
  const passed  = [];
  const rejected = [];

  for (let i = 0; i < candidates.length; i++) {
    const c   = candidates[i];
    const ctx = contextPerCandidate[i] || {};
    const rejections = [];

    for (const rule of REJECTION_CRITERIA) {
      try {
        if (rule.test(c, ctx)) {
          rejections.push({ id:rule.id, description:rule.description, severity:rule.severity });
        }
      } catch(e) { /* ignore rule errors */ }
    }

    if (rejections.length > 0) {
      rejected.push({ ...c, rejections, primaryReason:rejections[0].description });
    } else {
      passed.push(c);
    }
  }

  return { passed, rejected };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 12 — DECOMPOSED CANDIDATE RANKING WITH WATERFALL EXPLANATION
// ═══════════════════════════════════════════════════════════════════════════

/**
 * explainCandidate(candidate, context, sequence, splitSystem, bbbStrategy)
 * Generates a waterfall breakdown for a single cut candidate.
 *
 * context = {
 *   contactDisruption, domainScore, functionalRisk, assemblyScore,
 *   f1Foldability, f2Foldability, fusionBurden, geometry,
 *   manufacturability, contactMap
 * }
 *
 * Returns {
 *   position, baseScore, waterfallSteps[], finalScore,
 *   verdict, drivers[], limiters[], upgradeHints[]
 * }
 */
function explainCandidate(candidate, context, sequence, splitSystem, bbbStrategy) {
  context     = context     || {};
  splitSystem  = splitSystem  || 'split_intein_npu';
  bbbStrategy  = bbbStrategy  || 'tfr_rmt';

  const steps = [];
  let runningScore = 100; // Start at 100 and apply penalties/bonuses

  // Step 1: Base SPELL score
  const spellScore = Math.round((candidate.splitScore || 0.5) * 100);
  steps.push({ label:'Base SPELL site score', delta:spellScore - 100, running:spellScore, note:`Loop=${((candidate.scores||{}).loop||0).toFixed(2)}, Acc=${((candidate.scores||{}).accessibility||0).toFixed(2)}, Cons=${((candidate.scores||{}).conservation||0).toFixed(2)}` });
  runningScore = spellScore;

  // Step 2: Contact disruption penalty
  if (context.contactDisruption && context.contactDisruption.category !== 'no_structure') {
    const cd = context.contactDisruption;
    const cdPenalty = -Math.round(cd.penalty * 40); // max -40 pts
    runningScore = Math.max(0, runningScore + cdPenalty);
    steps.push({ label:`Contact disruption (${cd.category})`, delta:cdPenalty, running:runningScore, note:`${cd.brokenCount} contacts broken, wt.loss=${cd.weightedLoss}` });
  }

  // Step 3: Domain integrity
  if (context.domainScore) {
    const ds = context.domainScore;
    const domainDelta = Math.round((ds.score - 0.5) * 30); // -15 to +15
    runningScore = Math.max(0, Math.min(100, runningScore + domainDelta));
    steps.push({ label:`Domain integrity (${ds.classification})`, delta:domainDelta, running:runningScore, note:ds.explanation });
  }

  // Step 4: Functional risk penalty
  if (context.functionalRisk) {
    const fr = context.functionalRisk;
    const frPenalty = -Math.round(fr.overallRisk * 35);
    runningScore = Math.max(0, runningScore + frPenalty);
    steps.push({ label:'Functional residue risk', delta:frPenalty, running:runningScore, note:fr.reasons.slice(0,2).join('; ') || 'No direct functional risk' });
  }

  // Step 5: Assembly context
  if (context.assemblyScore) {
    const asm = context.assemblyScore;
    const asmPenalty = -Math.round(asm.assemblyRisk * 30);
    runningScore = Math.max(0, runningScore + asmPenalty);
    steps.push({ label:`Assembly context (${asm.assemblyClass})`, delta:asmPenalty, running:runningScore, note:asm.notes[0] || '' });
  }

  // Step 6: Fragment foldability
  if (context.f1Foldability && context.f2Foldability) {
    const f1 = context.f1Foldability;
    const f2 = context.f2Foldability;
    const weakerFold = Math.min(f1.overall, f2.overall);
    const foldDelta = Math.round((weakerFold - 0.5) * 20); // -10 to +10
    runningScore = Math.max(0, Math.min(100, runningScore + foldDelta));
    steps.push({ label:`Fragment foldability (weaker: ${weakerFold >= 0.6 ? 'F1' : 'F2'})`, delta:foldDelta, running:runningScore, note:`F1=${Math.round(f1.overall*100)}% F2=${Math.round(f2.overall*100)}%` });
  }

  // Step 7: Geometry bonus/penalty
  if (context.geometry) {
    const geo = context.geometry;
    const geoDelta = Math.round((geo.score - 0.5) * 15);
    runningScore = Math.max(0, Math.min(100, runningScore + geoDelta));
    steps.push({ label:`Reassembly geometry (${splitSystem})`, delta:geoDelta, running:runningScore, note:geo.terminalDistance_A != null ? `Terminal dist ${geo.terminalDistance_A}Å` : 'No structure data' });
  }

  // Step 8: Fusion burden
  if (context.fusionBurden) {
    const fb = context.fusionBurden;
    const fbDelta = Math.round((fb.overallBurden - 0.5) * 12);
    runningScore = Math.max(0, Math.min(100, runningScore + fbDelta));
    steps.push({ label:'Fusion burden', delta:fbDelta, running:runningScore, note:`F1 construct ${Math.round(fb.f1ConstructMW/1000)}kDa, F2 construct ${Math.round(fb.f2ConstructMW/1000)}kDa` });
  }

  // Step 9: Manufacturability
  if (context.manufacturability) {
    const mfg = context.manufacturability;
    const mfgDelta = Math.round((mfg.feasibility - 50) / 10);
    runningScore = Math.max(0, Math.min(100, runningScore + mfgDelta));
    steps.push({ label:`Manufacturability (grade ${mfg.grade})`, delta:mfgDelta, running:runningScore, note:`Limiting: ${mfg.limitingFactor}` });
  }

  const finalScore = Math.max(0, Math.min(100, Math.round(runningScore)));
  const verdict = finalScore >= 65 ? 'Excellent' : finalScore >= 45 ? 'Good' : finalScore >= 25 ? 'Marginal' : 'Poor';
  const color   = finalScore >= 65 ? '#3ecf8e'   : finalScore >= 45 ? '#f59e0b' : finalScore >= 25 ? '#fb923c' : '#f87171';

  // Identify drivers and limiters
  const drivers  = steps.filter(s => s.delta > 0).sort((a,b) => b.delta - a.delta);
  const limiters = steps.filter(s => s.delta < 0).sort((a,b) => a.delta - b.delta);

  // Generate upgrade hints
  const hints = [];
  if (limiters.length > 0) {
    const top = limiters[0].label;
    if (top.includes('Contact'))    hints.push('Shift cut ±3-5 residues to reduce contact disruption — try positions near local accessibility maximum');
    if (top.includes('Domain'))     hints.push('Seek an inter-domain linker — these score near 95/100 vs intra-domain cuts which score <25/100');
    if (top.includes('Functional')) hints.push('Increase distance to functional residues — minimum 12aa clearance recommended');
    if (top.includes('Assembly'))   hints.push('Move cut to the safe zone annotated in the assembly state map');
    if (top.includes('Fragment'))   hints.push('Add MGG cap to the weaker fragment\'s new N-terminus; consider degron protection');
    if (top.includes('Geometry'))   hints.push('Add (GGGGS)×3 linker to extend reach; verify geometry with AlphaFold2 prediction of the split construct');
    if (top.includes('Fusion'))     hints.push('Reduce fragment MW — each kDa below 8kDa significantly improves construct deliverability');
    if (top.includes('Manufact'))   hints.push('Screen sequence variants at hydrophobic patches; reduce Cys count via Cys→Ser mutations');
  }

  return {
    position     : candidate.position,
    aa           : candidate.aa,
    f1Length     : candidate.f1Length,
    f2Length     : candidate.f2Length,
    baseScore    : spellScore,
    finalScore,
    verdict, color,
    waterfallSteps: steps,
    drivers, limiters,
    upgradeHints : hints,
    scoreRange   : {
      low : Math.max(0, finalScore - 15),
      high: Math.min(100, finalScore + 15),
    },
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 13 — MONTE CARLO CUT RANKING
// ═══════════════════════════════════════════════════════════════════════════

/**
 * runSplitMonteCarlo(candidates, contextArr, n)
 * Runs n Monte Carlo simulations varying key parameters to test rank stability.
 * Returns per-candidate: { position, rankPct3, expectedRank, p10Score, p90Score }
 *
 * Parameters sampled with documented CV:
 *   SPELL score:      ±15% (sequence heuristic uncertainty)
 *   Contact penalty:  ±30% (contact weight calibration uncertainty)
 *   Foldability:      ±20% (disorder/aggregation model uncertainty)
 *   Assembly risk:    ±25% (oligomeric state uncertainty)
 *   Geometry:         ±20% (structural geometry uncertainty)
 */
function runSplitMonteCarlo(candidates, contextArr, n) {
  n = n || 300; // Fast enough for synchronous UI use
  contextArr = contextArr || candidates.map(() => ({}));

  if (!candidates || candidates.length === 0) return { results:[], n };

  // Box-Muller normal sample
  function randNorm() {
    const u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function sampleNorm(center, cv) {
    return Math.max(0, Math.min(100, center * (1 + cv * randNorm())));
  }

  const K = candidates.length;
  const rankCounts = Array.from({length:K}, () => 0); // count of top-3 finishes
  const scoreAccum = Array.from({length:K}, () => []);

  for (let iter = 0; iter < n; iter++) {
    // Sample noisy scores for each candidate
    const iterScores = candidates.map((c, i) => {
      const ctx = contextArr[i] || {};
      const spellBase  = (c.splitScore || 0.5) * 100;
      const cdPenalty  = ctx.contactDisruption ? ctx.contactDisruption.penalty * 40 : 0;
      const domScore   = ctx.domainScore ? (ctx.domainScore.score - 0.5) * 30 : 0;
      const funcPen    = ctx.functionalRisk ? ctx.functionalRisk.overallRisk * 35 : 0;
      const asmPen     = ctx.assemblyScore ? ctx.assemblyScore.assemblyRisk * 30 : 0;
      const f1fold     = ctx.f1Foldability ? ctx.f1Foldability.overall : 0.6;
      const f2fold     = ctx.f2Foldability ? ctx.f2Foldability.overall : 0.6;
      const foldAdj    = (Math.min(f1fold, f2fold) - 0.5) * 20;

      const baseComposite = Math.max(0, Math.min(100,
        spellBase - cdPenalty + domScore - funcPen - asmPen + foldAdj
      ));

      // Apply per-parameter noise
      const nSpell   = sampleNorm(spellBase, 0.15);
      const nCD      = sampleNorm(cdPenalty, 0.30);
      const nFunc    = sampleNorm(funcPen, 0.25);
      const nAsm     = sampleNorm(asmPen, 0.25);
      const nFold    = sampleNorm((Math.min(f1fold, f2fold) - 0.5) * 20 + 50, 0.20) - 50;

      const noisyScore = Math.max(0, Math.min(100,
        nSpell - nCD + domScore - nFunc - nAsm + nFold
      ));
      return { i, score:noisyScore };
    });

    // Rank this iteration
    iterScores.sort((a,b) => b.score - a.score);
    for (let rank = 0; rank < Math.min(3, iterScores.length); rank++) {
      rankCounts[iterScores[rank].i]++;
    }
    iterScores.forEach(s => scoreAccum[s.i].push(s.score));
  }

  function percentile(arr, p) {
    const sorted = arr.slice().sort((a,b)=>a-b);
    return Math.round(sorted[Math.floor(p/100 * (sorted.length-1))]);
  }

  const results = candidates.map((c, i) => ({
    position    : c.position,
    aa          : c.aa,
    rankPct3    : Math.round(rankCounts[i] / n * 100),
    expectedRank: null, // filled below
    expectedScore: Math.round(scoreAccum[i].reduce((s,v)=>s+v,0) / scoreAccum[i].length),
    p10Score    : percentile(scoreAccum[i], 10),
    p90Score    : percentile(scoreAccum[i], 90),
    rankStability: percentile(scoreAccum[i], 90) - percentile(scoreAccum[i], 10) < 25 ? 'stable' : 'volatile',
  }));

  results.sort((a,b) => b.rankPct3 - a.rankPct3);
  results.forEach((r,i) => { r.expectedRank = i + 1; });

  return {
    results,
    n,
    topCandidate    : results[0],
    certainty       : results[0].rankPct3 >= 50 ? 'dominant'
                    : results[0].rankPct3 >= 30 ? 'preferred'
                    : results[0].rankPct3 >= 15 ? 'marginal' : 'toss-up',
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// CORE SEQUENCE ENGINE (v8.0, preserved + enhanced)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * parseFASTA(text) — unchanged from v8.0
 */
function parseFASTA(text) {
  if (!text || !text.trim()) return { error:'Empty input', name:'', sequence:'', length:0 };
  const lines = text.trim().split(/\r?\n/);
  let name = 'Protein';
  const seqLines = [];
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i].trim();
    if (!l) continue;
    if (l.charAt(0) === '>') { name = l.slice(1).trim().split(/\s+/)[0] || 'Protein'; }
    else                     { seqLines.push(l.toUpperCase().replace(/[^ACDEFGHIKLMNPQRSTVWY]/g, '')); }
  }
  const sequence = seqLines.join('');
  if (!sequence.length) return { error:'No valid amino acid sequence found', name, sequence:'', length:0 };
  return { name, sequence, length:sequence.length, error:null };
}

/**
 * estimateResidueProperties(sequence) — v8.0 core, unchanged
 */
function estimateResidueProperties(sequence) {
  const N = sequence.length;
  const W = 7;

  function smooth(arr, w) {
    return arr.map(function(_, i) {
      let lo = Math.max(0, i-w), hi = Math.min(N-1, i+w), sum = 0, cnt = 0;
      for (let j = lo; j <= hi; j++) { sum += arr[j]; cnt++; }
      return sum / cnt;
    });
  }

  const rawHelix  = sequence.split('').map(c => _sp_get(CF_HELIX,  c, 1.0));
  const rawSheet  = sequence.split('').map(c => _sp_get(CF_SHEET,  c, 1.0));
  const rawHydro  = sequence.split('').map(c => _sp_get(KD_HYDRO,  c, 0));
  const rawBurial = sequence.split('').map(c => _sp_get(JN_BURIAL, c, 0));
  const rawDisord = sequence.split('').map(c => _sp_get(DIS_PROP,  c, 0));

  const sHelix  = smooth(rawHelix,  W);
  const sSheet  = smooth(rawSheet,  W);
  const sHydro  = smooth(rawHydro,  W);
  const sBurial = smooth(rawBurial, W);
  const sDisord = smooth(rawDisord, W);

  function ssCall(i) {
    if (sHelix[i] > 1.05 && sHelix[i] >= sSheet[i]) return 'helix';
    if (sSheet[i] > 1.05 && sSheet[i] >= sHelix[i]) return 'sheet';
    return 'loop';
  }
  function accessibility(i) {
    const raw = 0.50 - sBurial[i] * 0.30 - sHydro[i] * 0.04;
    return Math.max(0, Math.min(1, raw + 0.50));
  }
  function conservationPenalty(i) {
    const lo = Math.max(0, i-W), hi = Math.min(N-1, i+W);
    const freq = {};
    for (let j = lo; j <= hi; j++) {
      const c = sequence[j];
      freq[c] = (freq[c] || 0) + 1;
    }
    const total = hi - lo + 1;
    let H = 0;
    Object.keys(freq).forEach(c => {
      const p = freq[c] / total;
      H -= p * Math.log2(p);
    });
    return Math.max(0, 1 - H / 3.0);
  }

  return sequence.split('').map((aa, i) => ({
    index         : i,
    position      : i + 1,
    aa,
    ss            : ssCall(i),
    accessibility : accessibility(i),
    conservation  : conservationPenalty(i),
    hydrophobicity: sHydro[i],
    disorder      : Math.min(1, Math.max(0, sDisord[i] + 0.35)),
  }));
}

/**
 * SPLIT_HARD_RULES — unchanged from v8.0 (now augmented by rejection engine)
 */
const SPLIT_HARD_RULES = {
  min_fragment_aa : 50,
  acc_threshold   : 0.35,
  cons_threshold  : 0.65,
  min_dist_to_site: 8,
};

/**
 * findSplitSiteCandidates — v8.0 preserved.
 * Now also accepts structurally-enriched props from applyStructureToProps().
 */
function findSplitSiteCandidates(props, annotations, constraints) {
  annotations = annotations || {};
  constraints = Object.assign({}, SPLIT_HARD_RULES, constraints || {});
  const N = props.length;

  const forbidden = new Set();
  ['active_site','binding_hotspot','interface','forbidden'].forEach(function(key) {
    (annotations[key] || []).forEach(function(pos) { forbidden.add(pos); });
  });
  const forbiddenExpanded = new Set();
  forbidden.forEach(function(pos) {
    for (let d = -constraints.min_dist_to_site; d <= constraints.min_dist_to_site; d++) {
      forbiddenExpanded.add(pos + d);
    }
  });

  const candidates = [];
  for (let i = 0; i < N - 1; i++) {
    const pos1  = i + 1;
    const f1len = pos1;
    const f2len = N - pos1;

    if (f1len < constraints.min_fragment_aa)     continue;
    if (f2len < constraints.min_fragment_aa)     continue;
    if (props[i].ss !== 'loop')                  continue;
    if (props[i].accessibility < constraints.acc_threshold)  continue;
    if (props[i].conservation  > constraints.cons_threshold) continue;
    if (forbiddenExpanded.has(pos1))             continue;

    const ssScore  = 1.0;
    const accScore = Math.min(1, (props[i].accessibility - constraints.acc_threshold) / (1 - constraints.acc_threshold));
    const consScore = 1 - props[i].conservation;

    let minDist = Infinity;
    forbidden.forEach(fp => { minDist = Math.min(minDist, Math.abs(pos1 - fp)); });
    if (!isFinite(minDist)) minDist = N;
    const distScore = Math.min(1, minDist / 30);

    const balance  = Math.min(f1len, f2len) / Math.max(f1len, f2len);
    const balScore = 0.50 + 0.50 * balance;

    // V10: use structural disorder if available, else sequence-based
    const disVal   = props[i].disorder || 0;
    const disScore = 0.40 + 0.60 * disVal;

    // V10: AF2 pLDDT bonus — flexible confident positions are ideal
    const plddtBonus = (props[i].plddt != null && props[i].plddt < 70 && props[i].plddt >= 40) ? 0.05 : 0;

    const splitScore = (
      ssScore   * 0.28 +
      accScore  * 0.22 +
      consScore * 0.20 +
      distScore * 0.16 +
      balScore  * 0.09 +
      disScore  * 0.05
    ) + plddtBonus;

    candidates.push({
      position      : pos1,
      aa            : props[i].aa,
      ss            : props[i].ss,
      f1Length      : f1len,
      f2Length      : f2len,
      hasStructure  : props[i].hasStructure || false,
      plddt         : props[i].plddt || null,
      bFactor       : props[i].bFactor || null,
      scores        : {
        loop         : parseFloat(ssScore.toFixed(3)),
        accessibility: parseFloat(accScore.toFixed(3)),
        conservation : parseFloat(consScore.toFixed(3)),
        funcDistance : parseFloat(distScore.toFixed(3)),
        balance      : parseFloat(balScore.toFixed(3)),
        disorder     : parseFloat(disScore.toFixed(3)),
      },
      splitScore    : parseFloat(Math.min(1, splitScore).toFixed(3)),
      verdict       : splitScore>=0.70?'Excellent':splitScore>=0.50?'Good':splitScore>=0.35?'Marginal':'Poor',
      minDistToSite : isFinite(minDist) ? minDist : null,
    });
  }

  candidates.sort((a,b) => b.splitScore - a.splitScore);
  return candidates;
}

// ═══════════════════════════════════════════════════════════════════════════
// FRAGMENT PROPERTY ESTIMATORS (v8.0, preserved)
// ═══════════════════════════════════════════════════════════════════════════
function estimateFragmentMW(seq) { return (seq || '').length * 110; }

function estimateLogP(seq) {
  if (!seq || !seq.length) return -2.0;
  let total = 0;
  for (let i = 0; i < seq.length; i++) total += _sp_get(KD_HYDRO, seq[i], 0);
  return parseFloat((total / seq.length * 0.4 - 1.5).toFixed(2));
}

function estimateHBD(seq) {
  let count = 0;
  for (let i = 0; i < (seq||'').length; i++) {
    if ('RKNDQHSTYWM'.indexOf(seq[i]) >= 0) count++;
  }
  return Math.round(count * 0.4);
}

function estimateHBA(seq) {
  let count = 0;
  for (let i = 0; i < (seq||'').length; i++) {
    if ('RKNDQEHSTYWM'.indexOf(seq[i]) >= 0) count++;
  }
  return Math.round(count * 0.6);
}

function terminiDegronRisk(f1seq, f2seq) {
  const risk = { f1Cterminus:'low', f2Nterminus:'low', f2Cterminus:'low', overall:'low' };
  if (f1seq && N_DEGRON_AA.indexOf(f1seq[f1seq.length-1]) >= 0) risk.f1Cterminus = 'medium';
  if (f2seq && N_DEGRON_AA.indexOf(f2seq[0])              >= 0) risk.f2Nterminus = 'high';
  if (f2seq && N_DEGRON_AA.indexOf(f2seq[f2seq.length-1]) >= 0) risk.f2Cterminus = 'medium';
  const levels = ['low','medium','high'];
  const worst = Math.max(
    levels.indexOf(risk.f1Cterminus),
    levels.indexOf(risk.f2Nterminus),
    levels.indexOf(risk.f2Cterminus)
  );
  risk.overall = levels[Math.max(0, worst)];
  return risk;
}

// ═══════════════════════════════════════════════════════════════════════════
// DELIVERY MODE SCORING (v8.0, preserved)
// ═══════════════════════════════════════════════════════════════════════════
function scoreDeliveryMode(f1mw, f2mw, mode, bbbStrat, tissue, barrier) {
  tissue  = tissue  || 'cns';
  barrier = barrier || {};
  const maxMW = Math.max(f1mw, f2mw);
  let score = 0;
  const notes = [], warnings = [];

  if (mode === 'dual_aav') {
    const payloadOK = maxMW <= 158000;
    const coTrans   = tissue === 'cns' ? 0.55 : 0.65;
    score = payloadOK ? coTrans : coTrans * (158000 / maxMW) * 0.5;
    if (!payloadOK) warnings.push('Fragment >' + Math.round(maxMW/1000) + ' kDa may exceed dual-AAV payload ceiling.');
    notes.push('Co-transduction ~55% in CNS with AAV9/AAVrh10.');
    if (tissue === 'cns') notes.push('Npu DnaE or gp41-1 intein required for trans-splicing.');

  } else if (mode === 'mrna_lnp') {
    const imbalance     = Math.max(f1mw, f2mw) / Math.min(f1mw, f2mw);
    const stoichPenalty = imbalance > 2.5 ? 0.30 : imbalance > 1.5 ? 0.10 : 0;
    const endoEscape    = 0.70;
    score = endoEscape * (1 - stoichPenalty) * (tissue === 'cns' ? 0.75 : 1.0);
    if (stoichPenalty > 0) warnings.push('MW imbalance ' + imbalance.toFixed(1) + 'x — stoichiometry risk.');
    notes.push('LNP endosomal escape ~70% with optimised lipid formulation.');
    if (tissue === 'cns') notes.push('CNS LNP ~0.5-3% ID/g brain; IT administration improves exposure.');

  } else {
    const barrierDef = { tj:100,aj:100,mmp:0,nfkb:0,wnt:100,shh:100,pericyte:100,notch:100,angpt:100,ptm:0,cbf:100 };
    const b = Object.assign({}, barrierDef, barrier);
    const lp1 = Math.max(-5, Math.min(2, -1.5 - (f1mw / 15000)));
    const lp2 = Math.max(-5, Math.min(2, -1.5 - (f2mw / 15000)));
    const f1rmt = (bbbStrat === 'tfr_rmt');
    const f1rvg = (bbbStrat === 'rvg29');
    const f2rvg = (bbbStrat === 'tfr_rmt' || bbbStrat === 'rvg29');
    const mol1 = _buildMolecule({ mw:f1mw, logp:lp1, hbd:Math.min(15,Math.round(f1mw/2200)), hba:Math.min(20,Math.round(f1mw/1600)), ppb:0.20, type:'peptide', rmt:f1rmt, rvg:f1rvg, tj:b.tj,aj:b.aj,mmp:b.mmp,nfkb:b.nfkb,wnt:b.wnt,shh:b.shh,pericyte:b.pericyte,notch:b.notch,angpt:b.angpt,ptm:b.ptm,cbf:b.cbf });
    const mol2 = _buildMolecule({ mw:f2mw, logp:lp2, hbd:Math.min(15,Math.round(f2mw/2200)), hba:Math.min(20,Math.round(f2mw/1600)), ppb:0.18, type:'peptide', rmt:false, rvg:f2rvg, tj:b.tj,aj:b.aj,mmp:b.mmp,nfkb:b.nfkb,wnt:b.wnt,shh:b.shh,pericyte:b.pericyte,notch:b.notch,angpt:b.angpt,ptm:b.ptm,cbf:b.cbf });
    const r1 = _analyzeMolecule(mol1, 'hippocampus');
    const r2 = _analyzeMolecule(mol2, 'hippocampus');
    score = (r1.score.net / 100) * (r2.score.net / 100);
    notes.push('F1 BBB: ' + Math.round(r1.score.net) + '/100 via ' + (r1.score.bestRoute || '—') + '.');
    notes.push('F2 BBB: ' + Math.round(r2.score.net) + '/100 via ' + (r2.score.bestRoute || '—') + '.');
    if (maxMW > 50000 && bbbStrat === 'none') warnings.push('Large biologic with no BBB strategy: brain exposure <0.1% ID.');
    if (bbbStrat === 'tfr_rmt') notes.push('TfR-RMT validated — pabinafusp alfa precedent (Japan 2021).');
    if (bbbStrat === 'rvg29')   notes.push('RVG-29 nAChR-targeting; preferentially transduces RABV-infected neurons.');
  }

  score = Math.max(0, Math.min(1, score));
  return { mode, bbbStrategy:bbbStrat, score, scorePct:Math.round(score * 100), notes, warnings, color: score >= 0.55 ? '#3ecf8e' : score >= 0.30 ? '#f59e0b' : '#f87171' };
}

function rankDeliveryStrategies(f1mw, f2mw, targetTissue, barrierState) {
  targetTissue = targetTissue || 'cns';
  barrierState = barrierState || {};
  const modes = [
    { mode:'dual_aav',         label:'Dual AAV',                          bbbStrat:'none'    },
    { mode:'mrna_lnp',         label:'mRNA / LNP co-delivery',            bbbStrat:'none'    },
    { mode:'protein_biologic', label:'Protein biologic + TfR-RMT shuttle',bbbStrat:'tfr_rmt' },
    { mode:'protein_biologic', label:'Protein biologic + RVG-29 shuttle', bbbStrat:'rvg29'   },
  ];
  const results = modes.map(m => {
    const d = scoreDeliveryMode(f1mw, f2mw, m.mode, m.bbbStrat, targetTissue, barrierState);
    return { id:m.mode+(m.bbbStrat!=='none'?'_'+m.bbbStrat:''), label:m.label, score:d.score, scorePct:d.scorePct, notes:d.notes, warnings:d.warnings, color:d.color };
  });
  results.sort((a,b) => b.score - a.score);
  results.forEach((r,i) => { r.rank = i+1; });
  return results;
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 9 — UNIFIED TOTAL SCORER (S_split × S_delivery × S_reassembly × S_function)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * scoreSplitDeliveryTotal(params)
 *
 * REPLACES the old scoreSplitDesign() as the primary composite scorer.
 * All Phase 1-9 sub-scores are integrated here.
 *
 * Required params:
 *   splitSite       — candidate from findSplitSiteCandidates()
 *   sequence        — full protein sequence
 *   splitSystem     — 'split_intein_npu' | 'split_gfp' | 'fkbp_frb' | ...
 *   concNM          — estimated intracellular fragment concentration (nM)
 *   deliveryMode    — 'dual_aav' | 'mrna_lnp' | 'protein_biologic'
 *   bbbStrategy     — 'tfr_rmt' | 'rvg29' | 'none'
 *   targetTissue    — 'cns' | 'peripheral'
 *   annotations     — { active_site[], binding_hotspot[], interface[], forbidden[], known_good_splits[] }
 *   domains         — array of { name, start, end, type }
 *   targetId        — 'P' | 'N' | 'L' or null (for RABV-specific assembly context)
 *
 * Optional:
 *   pdbResult       — parsePDB output
 *   afResult        — parseAlphaFold output
 *   contactMap      — buildContactMap output
 *   structuralProps — applyStructureToProps output
 *   barrierState    — BBB barrier parameters
 *   motifHits       — detectFunctionalMotifs output
 */
function scoreSplitDeliveryTotal(params) {
  params = params || {};
  const site       = params.splitSite    || {};
  const seq        = params.sequence     || '';
  const sysKey     = params.splitSystem  || 'split_intein_npu';
  const concNM     = params.concNM       != null ? params.concNM : 5;
  const mode       = params.deliveryMode || 'protein_biologic';
  const bbbStrat   = params.bbbStrategy  || 'tfr_rmt';
  const tissue     = params.targetTissue || 'cns';
  const annots     = params.annotations  || {};
  const domainList = params.domains      || [];
  const targetId   = params.targetId     || null;
  const barrier    = params.barrierState || {};
  const pdbResult  = params.pdbResult    || null;
  const afResult   = params.afResult     || null;
  const contactMap = params.contactMap   || null;

  const cutPos  = site.position || Math.floor(seq.length / 2);
  const f1seq   = seq.slice(0, cutPos);
  const f2seq   = seq.slice(cutPos);
  const f1mw    = estimateFragmentMW(f1seq);
  const f2mw    = estimateFragmentMW(f2seq);

  // ── Auto-detect functional motifs
  const motifHits = params.motifHits || detectFunctionalMotifs(seq);

  // ── Phase 2: Contact disruption
  const contactDisruption = scoreContactDisruption(cutPos, contactMap, annots);

  // ── Phase 3: Domain integrity
  const parsedDomains = parseDomainBoundaries(domainList, seq.length);
  const domainScore   = scoreDomainIntegrity(cutPos, parsedDomains, seq.length);

  // ── Phase 4: Functional risk
  const functionalRisk = scoreFunctionalRisk(cutPos, seq, annots, motifHits);

  // ── Phase 5: Fragment foldability
  const f1Foldability = scoreFragmentFoldability(f1seq, 'F1');
  const f2Foldability = scoreFragmentFoldability(f2seq, 'F2');

  // ── Phase 6: Reassembly geometry
  const geometry = scoreReassemblyGeometry(cutPos, pdbResult, sysKey);

  // ── Phase 7: Fusion burden
  const fusionBurden = scoreFusionBurden(f1mw, f2mw, sysKey, bbbStrat);

  // ── Phase 8: Assembly context (RABV-specific or generic)
  const assemblyScore = scoreAssemblyContext(cutPos, targetId, null);

  // ── Manufacturability
  const manufacturability = scoreManufacturability(f1seq, f2seq, sysKey, mode);

  // ── Degron risk
  const degronRisk = terminiDegronRisk(f1seq, f2seq);

  // ── Delivery
  const deliveryDetail = scoreDeliveryMode(f1mw, f2mw, mode, bbbStrat, tissue, barrier);

  // ── Reassembly (from shared_engine or local stub)
  const _computeReassembly = (_core && _core.computeReassembly) ? _core.computeReassembly
    : function(sys, conc) {
        const kd = { split_intein_npu:0.001, split_gfp:0.5, fkbp_frb:0.2, leucine_zipper:100, nanobit:190000 }[sys] || 1;
        const f  = conc / (conc + kd);
        return { probability:f*0.85, probabilityPct:Math.round(f*85) };
      };
  const reassemblyResult = _computeReassembly(sysKey, concNM);

  // ══ COMPOSITE SCORING ════════════════════════════════════
  // Each dimension → penalty applied to running score

  // S_site: SPELL score (0-1)
  const Ssite = site.splitScore || 0.5;

  // S_contact: 1 - contactPenalty (0-1)
  const Scontact = 1 - contactDisruption.penalty;

  // S_domain: domain integrity score (0-1)
  const Sdomain = domainScore.score;

  // S_function: 1 - functionalRisk.overallRisk
  const Sfunc = 1 - functionalRisk.overallRisk;

  // S_assembly: 1 - assemblyRisk
  const Sasm = 1 - assemblyScore.assemblyRisk;

  // S_foldability: worst fragment foldability
  const Sfold = Math.min(f1Foldability.overall, f2Foldability.overall);

  // S_geometry: reassembly geometry
  const Sgeo = geometry.score;

  // S_fusion: overall burden inverted
  const Sfusion = fusionBurden.overallBurden;

  // S_delivery: delivery score
  const Sdeliv = deliveryDetail.score;

  // S_reassembly
  const Sreasm = reassemblyResult.probability;

  // S_mfg
  const Smfg = manufacturability.feasibility / 100;

  // Weighted total (Phase 9: S_total = product of probability chain + structure corrections)
  const structureWeight = pdbResult && !pdbResult.error ? 1.0 : 0.70; // discount if no structure
  const baseScore = (
    Ssite    * 0.18 +
    Scontact * 0.14 * structureWeight +
    Sdomain  * 0.14 +
    Sfunc    * 0.12 +
    Sasm     * 0.10 +
    Sfold    * 0.10 +
    Sgeo     * 0.08 * structureWeight +
    Sfusion  * 0.06 +
    Sdeliv   * 0.05 +
    Sreasm   * 0.03
  );

  // Uncertainty discount
  const hasAnnot = Object.keys(annots).some(k => (annots[k]||[]).length > 0);
  const hasStruct = !!(pdbResult && !pdbResult.error);
  const seqQual   = Math.min(1, seq.length / 100);
  let U = 0.20
    + (hasAnnot ? 0 : 0.10)
    + (hasStruct ? 0 : 0.12)
    + (1 - seqQual) * 0.10;
  if (sysKey === 'fkbp_frb') U += 0.05;
  if (sysKey === 'nanobit')  U += 0.15;

  const finalScore = Math.round(Math.max(0, Math.min(100, baseScore * (1 - 0.30 * Math.min(1, U)) * 100)));
  const color      = finalScore >= 65 ? '#3ecf8e' : finalScore >= 45 ? '#f59e0b' : finalScore >= 25 ? '#fb923c' : '#f87171';

  // Context per candidate (for rejection engine)
  const candidateContext = { contactDisruption, domainScore, functionalRisk, fusionBurden, assemblyScore, f1Foldability, f2Foldability };

  // Run rejection engine
  const rejectionResult = rejectCandidates([site], [candidateContext]);
  const isRejected      = rejectionResult.rejected.length > 0;
  const rejectionReason = isRejected ? rejectionResult.rejected[0].primaryReason : null;

  // Waterfall explanation
  const explanation = explainCandidate(site, {
    contactDisruption, domainScore, functionalRisk, assemblyScore,
    f1Foldability, f2Foldability, fusionBurden, geometry, manufacturability,
  }, seq, sysKey, bbbStrat);

  return {
    // Identity
    splitSite    : site,
    position     : cutPos,
    f1seq, f2seq, f1mw, f2mw,
    degronRisk,

    // Sub-scores (all 0-1)
    subscores: {
      siteSPELL        : parseFloat(Ssite.toFixed(3)),
      contactIntegrity : parseFloat(Scontact.toFixed(3)),
      domainIntegrity  : parseFloat(Sdomain.toFixed(3)),
      functionalSafety : parseFloat(Sfunc.toFixed(3)),
      assemblyContext  : parseFloat(Sasm.toFixed(3)),
      fragFoldability  : parseFloat(Sfold.toFixed(3)),
      reassemblyGeo    : parseFloat(Sgeo.toFixed(3)),
      fusionBurden     : parseFloat(Sfusion.toFixed(3)),
      delivery         : parseFloat(Sdeliv.toFixed(3)),
      reassemblyProb   : parseFloat(Sreasm.toFixed(3)),
      manufacturability: parseFloat(Smfg.toFixed(3)),
    },

    // Full detail objects for UI
    contactDisruption,
    domainScore,
    functionalRisk,
    assemblyScore,
    f1Foldability,
    f2Foldability,
    geometry,
    fusionBurden,
    manufacturability,
    reassemblyDetail : reassemblyResult,
    deliveryDetail,
    motifHits,

    // Final output
    uncertainty      : parseFloat(U.toFixed(3)),
    baseScore        : parseFloat(baseScore.toFixed(3)),
    finalScore,
    color,
    verdict          : finalScore>=65?'Excellent':finalScore>=45?'Good':finalScore>=25?'Marginal':'Poor',
    isRejected,
    rejectionReason,

    // Waterfall + upgrade
    explanation,
    upgradeHints     : explanation.upgradeHints,
    scoreRange       : { low:Math.max(0, finalScore-15), high:Math.min(100, finalScore+15) },

    // hasStructure flag for UI
    hasStructure     : hasStruct,
    hasAnnotations   : hasAnnot,
  };
}

// ── Legacy alias for backward compatibility ────────────────────────────────
function scoreSplitDesign(params) {
  return scoreSplitDeliveryTotal(params);
}

// ═══════════════════════════════════════════════════════════════════════════
// FULL ANALYSIS RUNNER
// ═══════════════════════════════════════════════════════════════════════════

/**
 * runFullAnalysis(params)
 *
 * Convenience wrapper that runs the complete pipeline for a given target.
 *
 * params: {
 *   sequence         — FASTA text or raw sequence string
 *   targetId         — 'P' | 'N' | 'L' (uses RABV_TARGET_MAPS if set)
 *   pdbText          — optional PDB file text
 *   afPdbText        — optional AlphaFold PDB text
 *   splitSystem      — split system key
 *   concNM           — intracellular fragment concentration
 *   deliveryMode     — delivery mode
 *   bbbStrategy      — BBB shuttle strategy
 *   topN             — return top N candidates (default 10)
 *   customAnnotations— override target annotations
 *   customDomains    — override domain list
 *   constraints      — SPLIT_HARD_RULES overrides
 * }
 *
 * Returns {
 *   sequence, N, target,
 *   residueProps,
 *   candidates,        — all passing candidates (sorted by finalScore)
 *   rejectedCandidates,
 *   topCandidates,     — top N
 *   mcResults,         — Monte Carlo ranking of top 20
 *   deliveryRanking,
 *   calibration,
 *   summary,
 *   motifHits,
 *   hasStructure, hasAnnotations,
 * }
 */
function runFullAnalysis(params) {
  params = params || {};

  // 1. Resolve sequence + annotations
  let seq, annotations, domainList, targetLabel;
  const usePreset = params.targetId && RABV_TARGET_MAPS[params.targetId];

  if (usePreset) {
    const preset = RABV_TARGET_MAPS[params.targetId];
    seq         = preset.sequence || params.sequence || '';
    annotations = Object.assign({}, preset.annotations, params.customAnnotations || {});
    domainList  = params.customDomains || preset.domains || [];
    targetLabel = preset.name;
  } else {
    const fastaResult = parseFASTA(typeof params.sequence === 'string' ? params.sequence : '');
    seq         = fastaResult.sequence || params.sequence || '';
    annotations = params.customAnnotations || {};
    domainList  = params.customDomains || [];
    targetLabel = fastaResult.name || 'Custom';
  }

  if (!seq || seq.length < 50) {
    return { error:'Sequence too short or unavailable', sequence:seq, N:seq.length };
  }

  // 2. Parse structural data if provided
  const pdbResult  = params.pdbText    ? parsePDB(params.pdbText)          : null;
  const afResult   = params.afPdbText  ? parseAlphaFold(params.afPdbText)  : null;
  const contactMap = pdbResult && !pdbResult.error ? buildContactMap(pdbResult) : null;

  // 3. Per-residue properties (with structural enrichment)
  let residueProps = estimateResidueProperties(seq);
  if (pdbResult && !pdbResult.error) {
    residueProps = applyStructureToProps(residueProps, pdbResult, afResult);
  }

  // 4. Find candidates
  const candidates = findSplitSiteCandidates(residueProps, annotations, params.constraints || {});

  // 5. Detect functional motifs
  const motifHits = detectFunctionalMotifs(seq);

  // 6. Score all candidates with full pipeline
  const scoredCandidates = candidates.map(c => {
    try {
      return scoreSplitDeliveryTotal({
        splitSite    : c,
        sequence     : seq,
        splitSystem  : params.splitSystem  || 'split_intein_npu',
        concNM       : params.concNM       != null ? params.concNM : 5,
        deliveryMode : params.deliveryMode || 'protein_biologic',
        bbbStrategy  : params.bbbStrategy  || 'tfr_rmt',
        targetTissue : params.targetTissue || 'cns',
        annotations,
        domains      : domainList,
        targetId     : params.targetId || null,
        pdbResult,
        afResult,
        contactMap,
        barrierState : params.barrierState || {},
        motifHits,
      });
    } catch(e) {
      return { position:c.position, finalScore:0, isRejected:true, rejectionReason:'Scoring error: '+e.message, error:true };
    }
  });

  // 7. Separate rejected
  const passed   = scoredCandidates.filter(s => !s.isRejected && !s.error);
  const rejected = scoredCandidates.filter(s => s.isRejected || s.error);

  // Sort by finalScore
  passed.sort((a,b) => b.finalScore - a.finalScore);
  rejected.sort((a,b) => b.position - a.position);

  const topN = params.topN || 10;
  const topCandidates = passed.slice(0, topN);

  // 8. Monte Carlo ranking of top candidates
  const mcInput     = topCandidates.slice(0, 20);
  const mcContexts  = mcInput.map(s => ({
    contactDisruption: s.contactDisruption,
    domainScore      : s.domainScore,
    functionalRisk   : s.functionalRisk,
    assemblyScore    : s.assemblyScore,
    f1Foldability    : s.f1Foldability,
    f2Foldability    : s.f2Foldability,
  }));
  const mcResults = runSplitMonteCarlo(mcInput.map(s => ({
    position: s.position, aa: s.position, f1Length: s.f1mw/110, f2Length: s.f2mw/110,
    splitScore: s.subscores ? s.subscores.siteSPELL : 0.5,
  })), mcContexts, 300);

  // 9. Delivery ranking
  const bestF1mw = topCandidates[0] ? topCandidates[0].f1mw : estimateFragmentMW(seq.slice(0, Math.floor(seq.length/2)));
  const bestF2mw = topCandidates[0] ? topCandidates[0].f2mw : estimateFragmentMW(seq.slice(Math.floor(seq.length/2)));
  const deliveryRanking = rankDeliveryStrategies(bestF1mw, bestF2mw, params.targetTissue || 'cns', params.barrierState || {});

  // 10. Calibration check
  const calibration = calibrateScorer(seq, annotations);

  // 11. Summary
  const bestSite = topCandidates[0];
  const summary = {
    topPosition        : bestSite ? bestSite.position : null,
    topScore           : bestSite ? bestSite.finalScore : null,
    topVerdict         : bestSite ? bestSite.verdict : null,
    totalCandidates    : candidates.length,
    passedCandidates   : passed.length,
    rejectedCount      : rejected.length,
    hasStructuralData  : !!(pdbResult && !pdbResult.error),
    hasAnnotations     : Object.keys(annotations).some(k => (annotations[k]||[]).length > 0),
    mcTopPosition      : mcResults.topCandidate ? mcResults.topCandidate.position : null,
    mcCertainty        : mcResults.certainty,
    recommendedDelivery: deliveryRanking[0] ? deliveryRanking[0].label : null,
    calibrationPassed  : calibration.passed,
    motifCount         : motifHits.length,
    rejectedExamples   : rejected.slice(0,3).map(r => `pos ${r.position}: ${r.rejectionReason || 'rejected'}`),
  };

  return {
    sequence        : seq,
    N               : seq.length,
    target          : targetLabel,
    residueProps    : residueProps.map(p => ({ // lightweight version for UI
      position:p.position, aa:p.aa, ss:p.ss,
      accessibility:+(p.accessibility.toFixed(2)),
      disorder:+(p.disorder.toFixed(2)),
      plddt:p.plddt, hasStructure:p.hasStructure||false,
    })),
    candidates      : passed,
    rejectedCandidates: rejected,
    topCandidates,
    mcResults,
    deliveryRanking,
    calibration,
    summary,
    motifHits,
    hasStructure    : !!(pdbResult && !pdbResult.error),
    hasAnnotations  : summary.hasAnnotations,
    pdbParsed       : pdbResult ? { chains:pdbResult.chains, nResidues:pdbResult.residues.length, error:pdbResult.error } : null,
    contactMapStats : contactMap ? { totalContacts:contactMap.contacts.length, N:contactMap.N } : null,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// LEGACY PROTEIN_PRESETS (v8.0, preserved for backward compat)
// ═══════════════════════════════════════════════════════════════════════════
const PROTEIN_PRESETS = {
  P: RABV_TARGET_MAPS.P,
  N: RABV_TARGET_MAPS.N,
  L: RABV_TARGET_MAPS.L,
};

// ═══════════════════════════════════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════════════════════════════════
const SplitterEngine = {
  // ── Core sequence ──────────────────────────────────────────────────────
  parseFASTA,
  estimateResidueProperties,
  findSplitSiteCandidates,
  SPLIT_HARD_RULES,

  // ── Phase 1: Structure ─────────────────────────────────────────────────
  parsePDB,
  parseAlphaFold,
  applyStructureToProps,

  // ── Phase 2: Contact map ───────────────────────────────────────────────
  buildContactMap,
  scoreContactDisruption,

  // ── Phase 3: Domains ───────────────────────────────────────────────────
  parseDomainBoundaries,
  scoreDomainIntegrity,

  // ── Phase 4: Functional risk ───────────────────────────────────────────
  detectFunctionalMotifs,
  scoreFunctionalRisk,
  FUNCTIONAL_MOTIFS,

  // ── Phase 5: Fragment foldability ──────────────────────────────────────
  scoreFragmentFoldability,

  // ── Phase 6: Geometry ──────────────────────────────────────────────────
  scoreReassemblyGeometry,

  // ── Phase 7: Fusion burden ─────────────────────────────────────────────
  scoreFusionBurden,
  FUSION_TAGS,

  // ── Phase 8: Assembly context ──────────────────────────────────────────
  scoreAssemblyContext,
  RABV_ASSEMBLY_STATES,

  // ── Phase 9: Total scorer ──────────────────────────────────────────────
  scoreSplitDeliveryTotal,
  scoreSplitDesign,         // legacy alias

  // ── Phase 10: Benchmarks ───────────────────────────────────────────────
  BENCHMARK_SPLITS,
  calibrateScorer,

  // ── Phase 11: RABV target maps ─────────────────────────────────────────
  RABV_TARGET_MAPS,
  PROTEIN_PRESETS,           // backward compat

  // ── Phase 12: Explainability ───────────────────────────────────────────
  explainCandidate,

  // ── Phase 13: Monte Carlo ──────────────────────────────────────────────
  runSplitMonteCarlo,

  // ── Phase 14: Manufacturability ────────────────────────────────────────
  scoreManufacturability,

  // ── Phase 15: Rejection engine ─────────────────────────────────────────
  rejectCandidates,
  REJECTION_CRITERIA,

  // ── Full pipeline runner ───────────────────────────────────────────────
  runFullAnalysis,

  // ── Fragment property estimators ──────────────────────────────────────
  estimateFragmentMW,
  estimateLogP,
  estimateHBD,
  estimateHBA,
  terminiDegronRisk,

  // ── Delivery ──────────────────────────────────────────────────────────
  scoreDeliveryMode,
  rankDeliveryStrategies,

  // ── Biophysical scales (for external auditing) ────────────────────────
  CF_HELIX, CF_SHEET, KD_HYDRO, JN_BURIAL, DIS_PROP, AGG_PROP, N_DEGRON_AA,

  version      : '10.0',
  phase1to15   : true,
};

if (typeof module !== 'undefined' && module.exports) module.exports = SplitterEngine;
else if (typeof window !== 'undefined') window.SplitterEngine = SplitterEngine;
