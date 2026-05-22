"""
splitter_core.py — NeuroViral Lab Protein Splitter Engine  v1.0
Python port of splitter_engine.js v10.0 (all 15 phases)

Architecture mirrors bbb_core.py and rabv_core.py exactly:
  - Dataclass input/output schemas
  - All 15 phases implemented
  - Domain ownership: Protein Engineer / Bioinformatician
  - FastAPI-compatible: called by splitter_main.py on port 8002

Scientific citations:
  Phase 1  PDB parsing:  Berman et al. Nucleic Acids Res 2000
  Phase 2  Contact maps: Grana et al. Nucleic Acids Res 2002 (CASP)
  Phase 3  Domain score: Apic et al. J Mol Biol 2001
  Phase 4  Motif detect: Gasteiger et al. Proteomics 2005
  Phase 5  Aggregation:  Fernandez-Escamilla et al. Nat Biotechnol 2004 (TANGO)
  Phase 5  Foldability:  Garbuzynskiy et al. Bioinformatics 2010 (FoldAmyloid)
  Phase 6  Geometry:     Bhaskara & Bhattacharyya, PLoS ONE 2011
  Phase 8  Oligomers:    Levy et al. Structure 2006 (PDB assembly)
  Phase 10 Benchmarks:   Cabantous 2005 (split-GFP); Dixon 2016 (split-Luc);
                         Truong 2015 (split-Cas9); Mootz 2003 (split-TEV)
  Phase 11 RABV maps:    Ribeiro 2009 (P LC8); Wiltzer 2014 (P TBK1);
                         Scrima 2008 (N structure); Rahmati 2025 (Pep2)
  Phase 13 Monte Carlo:  same noise model as shared_engine.js v9.0
           TRIPARTITE_F3_COND_COLOC = 0.22 (Cabantous 2013, Kerppola 2006)

DO NOT include BBB physics here — that belongs in bbb_core.py.
DO NOT include RABV kinetics here — that belongs in rabv_core.py.
"""

import math
import re
import time as _time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# BIOPHYSICAL SCALES
# ═══════════════════════════════════════════════════════════════════════════

CF_HELIX: Dict[str, float] = {
    'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,
    'G':0.57,'H':1.00,'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,
    'P':0.57,'S':0.77,'T':0.83,'W':1.08,'Y':0.69,'V':1.06,
}
CF_SHEET: Dict[str, float] = {
    'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,
    'G':0.75,'H':0.87,'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,
    'P':0.55,'S':0.75,'T':1.19,'W':1.37,'Y':1.47,'V':1.70,
}
KD_HYDRO: Dict[str, float] = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,
    'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,
    'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,
}
JN_BURIAL: Dict[str, float] = {
    'A':0.3,'R':-1.4,'N':-0.5,'D':-0.6,'C':0.9,'Q':-0.7,'E':-0.7,
    'G':0.3,'H':-0.1,'I':0.7,'L':0.5,'K':-1.8,'M':0.4,'F':0.5,
    'P':-0.3,'S':-0.1,'T':0.0,'W':0.3,'Y':-0.4,'V':0.6,
}
DIS_PROP: Dict[str, float] = {
    'A':0.06,'R':0.18,'N':0.15,'D':0.19,'C':-0.02,'Q':0.18,'E':0.24,
    'G':0.17,'H':0.05,'I':-0.12,'L':-0.11,'K':0.26,'M':-0.01,'F':-0.15,
    'P':0.23,'S':0.14,'T':0.09,'W':-0.13,'Y':-0.08,'V':-0.10,
}
# TANGO-inspired aggregation propensity (Fernandez-Escamilla 2004)
AGG_PROP: Dict[str, float] = {
    'A':0.0,'R':-2.0,'N':-1.5,'D':-2.5,'C':0.5,'Q':-1.0,'E':-2.5,
    'G':-0.5,'H':-1.5,'I':2.5,'L':2.0,'K':-2.0,'M':0.5,'F':3.0,
    'P':-3.0,'S':-0.5,'T':-0.5,'W':2.5,'Y':1.5,'V':2.5,
}
N_DEGRON_AA = set('RKHFWYLIED')

# Instability dipeptide scores (Gasteiger 2005, PEST-like)
INSTABILITY_DIPEPTIDES: Dict[str, int] = {
    'DP':0,'DG':0,'DS':0,'DD':1,'DE':1,'DT':0,
    'NP':1,'NS':0,'NG':1,'ND':1,'NQ':0,'NH':0,
}

def _sp(scale: Dict, ch: str, default: float = 0.0) -> float:
    return scale.get(ch.upper() if ch else '', default)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — STRUCTURE PARSING
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PDBResidue:
    res_seq: int
    chain: str
    res_name: str
    ca: Optional[Tuple[float, float, float]] = None  # (x,y,z)
    cb: Optional[Tuple[float, float, float]] = None
    b_factor: float = 20.0

@dataclass
class PDBResult:
    residues: List[PDBResidue] = field(default_factory=list)
    chains: List[str] = field(default_factory=list)
    error: Optional[str] = None

def parse_pdb(pdb_text: str) -> PDBResult:
    """
    Parse PDB ATOM records.  Extracts per-residue Cα/Cβ coords + B-factors.
    B-factor column is used as-is (AF2 models store pLDDT there).
    Ref: Berman et al. Nucleic Acids Res 2000.
    """
    if not pdb_text or not pdb_text.strip():
        return PDBResult(error='Empty PDB input')

    residue_map: Dict[str, PDBResidue] = {}
    chain_set: set = set()

    for line in pdb_text.splitlines():
        rec = line[:6].strip()
        if rec not in ('ATOM', 'HETATM'):
            continue
        if rec == 'HETATM':
            continue
        try:
            name     = line[12:16].strip()
            res_name = line[17:20].strip()
            chain    = line[21:22].strip() or 'A'
            res_seq  = int(line[22:26].strip())
            x        = float(line[30:38].strip())
            y        = float(line[38:46].strip())
            z        = float(line[46:54].strip())
            b_factor = float(line[60:66].strip()) if len(line) >= 66 else 20.0
        except (ValueError, IndexError):
            continue

        chain_set.add(chain)
        key = f'{chain}:{res_seq}'
        if key not in residue_map:
            residue_map[key] = PDBResidue(res_seq=res_seq, chain=chain, res_name=res_name, b_factor=b_factor)
        res = residue_map[key]
        if name == 'CA':
            res.ca = (x, y, z)
            res.b_factor = b_factor
        elif name == 'CB':
            res.cb = (x, y, z)

    residues = [r for r in residue_map.values() if r.ca is not None]
    residues.sort(key=lambda r: r.res_seq)

    if not residues:
        return PDBResult(error='No ATOM records with Cα found')

    return PDBResult(residues=residues, chains=sorted(chain_set))


def parse_alphafold(pdb_text: str) -> Dict:
    """
    AlphaFold models store pLDDT in the B-factor column.
    pLDDT >90 = very high confidence (ordered)
    pLDDT 70-90 = high confidence
    pLDDT 50-70 = low confidence (flexible)
    pLDDT <50   = very low (disordered)
    """
    result = parse_pdb(pdb_text)
    if result.error:
        return {'plddt': [], 'error': result.error}

    plddt = []
    for r in result.residues:
        score = r.b_factor
        plddt.append({
            'res_seq'  : r.res_seq,
            'plddt'    : score,
            'ordered'  : score >= 70,
            'flexible' : score < 50,
        })
    return {'plddt': plddt, 'residues': result.residues, 'error': None}


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — CONTACT MAP DISRUPTION
# ═══════════════════════════════════════════════════════════════════════════

def build_contact_map(pdb_result: PDBResult, cutoff_angstrom: float = 12.0) -> Optional[Dict]:
    """
    Build Cβ-Cβ (Cα for Gly) distance matrix.
    Weight tiers:
      surface contact (both exposed, dist 8-12Å) = 0.5
      core packing   (both buried, dist <8Å)      = 2.0
      edge           (one buried, one exposed)     = 1.2
    Ref: Grana et al. Nucleic Acids Res 2002.
    """
    if not pdb_result or pdb_result.error or not pdb_result.residues:
        return None

    residues = pdb_result.residues
    N = len(residues)
    avg_b = sum(r.b_factor for r in residues) / max(1, N)

    def get_coord(r: PDBResidue):
        return r.cb if r.cb else r.ca

    def dist3(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    def is_buried(r: PDBResidue):
        return r.b_factor < avg_b * 0.7

    contacts = []
    for i in range(N - 1):
        ci = get_coord(residues[i])
        if ci is None:
            continue
        for j in range(i + 4, N):  # skip i+1..i+3 (covalent bonds)
            cj = get_coord(residues[j])
            if cj is None:
                continue
            d = dist3(ci, cj)
            if d > cutoff_angstrom:
                continue
            bi = is_buried(residues[i])
            bj = is_buried(residues[j])
            if bi and bj and d < 8.0:
                w = 2.0
            elif not bi and not bj:
                w = 0.5
            else:
                w = 1.2
            contacts.append({
                'i': residues[i].res_seq,
                'j': residues[j].res_seq,
                'dist': round(d, 2),
                'weight': w,
                'both_buried': bi and bj,
            })

    return {'contacts': contacts, 'N': N, 'avg_b': round(avg_b, 2)}


def score_contact_disruption(cut_position: int, contact_map: Optional[Dict],
                              annotations: Optional[Dict] = None) -> Dict:
    """
    Counts contacts severed at cut_position (i <= cut AND j > cut).
    Amplifies weight when a functional residue is involved (×2.5).
    Penalty categories: negligible/mild/moderate/severe/critical.
    """
    if not contact_map:
        return {
            'penalty': 0.0, 'contact_penalty': 0.0,
            'broken': [], 'weighted_loss': 0.0,
            'broken_count': 0, 'category': 'no_structure',
            'note': 'No PDB data — contact disruption not calculated',
        }

    annotations = annotations or {}
    active_sites = set(
        (annotations.get('active_site') or []) +
        (annotations.get('binding_hotspot') or []) +
        (annotations.get('interface') or []) +
        (annotations.get('forbidden') or [])
    )

    broken = []
    weighted_loss = 0.0

    for c in contact_map['contacts']:
        if c['i'] <= cut_position < c['j']:
            w = c['weight']
            if c['i'] in active_sites or c['j'] in active_sites:
                w *= 2.5
            broken.append({**c, 'effective_weight': round(w, 2)})
            weighted_loss += w

    # Penalty mapping
    if weighted_loss < 2:
        penalty = weighted_loss / 40
        category = 'negligible'
    elif weighted_loss < 6:
        penalty = 0.05 + (weighted_loss - 2) / 20
        category = 'mild'
    elif weighted_loss < 12:
        penalty = 0.20 + (weighted_loss - 6) / 20
        category = 'moderate'
    elif weighted_loss < 20:
        penalty = 0.45 + (weighted_loss - 12) / 28
        category = 'severe'
    else:
        penalty = min(1.0, 0.75 + (weighted_loss - 20) / 80)
        category = 'critical'

    broken.sort(key=lambda x: -x['effective_weight'])

    return {
        'penalty'       : round(penalty, 4),
        'contact_penalty': round(penalty, 4),
        'broken'        : broken[:10],
        'weighted_loss' : round(weighted_loss, 2),
        'broken_count'  : len(broken),
        'category'      : category,
        'note'          : f'{len(broken)} contacts broken (wt.loss={weighted_loss:.1f}) — {category}',
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — DOMAIN INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

def parse_domain_boundaries(domain_list: List[Dict], sequence_length: int) -> List[Dict]:
    """Fill gaps between domains with linker entries."""
    if not domain_list:
        return []
    sorted_domains = sorted(domain_list, key=lambda d: d['start'])
    result = []
    prev = 1
    for d in sorted_domains:
        if d['start'] > prev + 3:
            result.append({'name': 'Linker', 'start': prev, 'end': d['start'] - 1,
                           'type': 'linker', 'is_linker': True})
        result.append({**d, 'is_linker': False})
        prev = d['end'] + 1
    if prev <= sequence_length:
        result.append({'name': 'C-terminal region', 'start': prev, 'end': sequence_length,
                       'type': 'linker', 'is_linker': True})
    return result


def score_domain_integrity(cut_position: int, domains: List[Dict],
                           sequence_length: int) -> Dict:
    """
    Score: 0-1 (higher = better).
    inter-domain linker = 0.95, catalytic cut = 0.05.
    Ref: Apic et al. J Mol Biol 2001.
    """
    if not domains:
        return {
            'score': 0.50, 'classification': 'unknown',
            'explanation': 'No domain boundaries provided — default score applied',
            'penalty': 0.50,
        }

    cut_domain = None
    for d in domains:
        if d['start'] <= cut_position <= d['end']:
            cut_domain = d
            break

    if cut_domain is None:
        return {
            'score': 0.95, 'classification': 'inter_domain_linker',
            'explanation': 'Cut falls between annotated domains — optimal position',
            'penalty': 0.05,
        }

    if cut_domain.get('is_linker') or cut_domain.get('type') in ('linker', 'disordered'):
        linker_len = cut_domain['end'] - cut_domain['start'] + 1
        if linker_len >= 10:
            return {
                'score': 0.90, 'classification': 'long_flexible_linker',
                'explanation': f'Cut in long linker ({linker_len}aa) between domains — excellent choice',
                'penalty': 0.10,
            }
        return {
            'score': 0.70, 'classification': 'short_linker',
            'explanation': f'Cut in short linker ({linker_len}aa) — acceptable but may constrain geometry',
            'penalty': 0.30,
        }

    t = cut_domain.get('type', 'scaffold')
    name = cut_domain.get('name', '')
    if t == 'catalytic':
        return {
            'score': 0.05, 'classification': 'catalytic_domain',
            'explanation': f'CRITICAL: Cut inside catalytic domain "{name}" — will destroy enzymatic function',
            'penalty': 0.95,
        }
    if t == 'binding':
        return {
            'score': 0.12, 'classification': 'interface_crossing',
            'explanation': f'SEVERE: Cut crosses binding domain "{name}" — will disrupt recognition',
            'penalty': 0.88,
        }
    if t == 'scaffold':
        return {
            'score': 0.25, 'classification': 'intra_domain_scaffold',
            'explanation': f'BAD: Cut inside structural scaffold domain "{name}" — likely to destabilize fold',
            'penalty': 0.75,
        }
    return {
        'score': 0.60, 'classification': 'intra_domain_flexible',
        'explanation': f'Cut inside domain "{name}" which contains flexible regions — borderline',
        'penalty': 0.40,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — FUNCTIONAL MOTIF DETECTION
# ═══════════════════════════════════════════════════════════════════════════

FUNCTIONAL_MOTIFS = [
    {'name': 'Phosphorylation (CK2)',   'pattern': r'[ST]..E',    'risk': 'indirect',  'weight': 0.4},
    {'name': 'Phosphorylation (PKA)',   'pattern': r'RR.S',       'risk': 'direct',    'weight': 0.6},
    {'name': 'NLS (classical)',         'pattern': r'KK[RK]K',    'risk': 'indirect',  'weight': 0.5},
    {'name': 'RdRp GDD motif',          'pattern': r'GDD',        'risk': 'catalytic', 'weight': 1.0},
    {'name': 'RdRp GDN motif (RABV)',   'pattern': r'GDN',        'risk': 'catalytic', 'weight': 1.0},
    {'name': 'Dynein LC8 motif',        'pattern': r'[KR]..T.Q',  'risk': 'direct',    'weight': 0.8},
    {'name': 'Coiled-coil heptad',      'pattern': r'[LI].{3}[LI].{3}[LI]', 'risk': 'indirect', 'weight': 0.5},
    {'name': 'Cys-rich Zn finger',      'pattern': r'C.{2,4}C.{3}C.{2,4}C', 'risk': 'direct', 'weight': 0.7},
    {'name': 'N-glycosylation',         'pattern': r'N[^P][ST]',  'risk': 'indirect',  'weight': 0.3},
    {'name': 'DEAD-box helicase',       'pattern': r'DEAD',       'risk': 'catalytic', 'weight': 1.0},
    {'name': 'Walker-A P-loop',         'pattern': r'G.{4}GK[ST]','risk': 'catalytic', 'weight': 0.9},
]

def detect_functional_motifs(sequence: str) -> List[Dict]:
    """Auto-detect functional motifs from sequence. Ref: Gasteiger 2005."""
    if not sequence:
        return []
    hits = []
    for m in FUNCTIONAL_MOTIFS:
        positions = []
        for match in re.finditer(m['pattern'], sequence):
            for k in range(len(match.group(0))):
                positions.append(match.start() + k + 1)  # 1-indexed
        if positions:
            hits.append({
                'motif': m['name'],
                'positions': sorted(set(positions)),
                'risk': m['risk'],
                'weight': m['weight'],
            })
    return hits


def score_functional_risk(cut_position: int, sequence: str,
                          annotations: Optional[Dict] = None,
                          motif_hits: Optional[List[Dict]] = None) -> Dict:
    """
    Direct risk:   cut destroys functional residues at/near cut point
    Indirect risk: cut severs structural scaffold supporting distant function
    Proximity radius = 8aa.
    """
    annotations = annotations or {}
    motif_hits  = motif_hits or []
    PROX = 8

    direct_sites   = set(
        (annotations.get('active_site') or []) +
        (annotations.get('binding_hotspot') or []) +
        (annotations.get('forbidden') or [])
    )
    indirect_sites = set(annotations.get('interface') or [])

    for m in motif_hits:
        for p in m['positions']:
            if m['risk'] in ('catalytic', 'direct'):
                direct_sites.add(p)
            else:
                indirect_sites.add(p)

    direct_risk   = 0.0
    indirect_risk = 0.0
    reasons       = []

    for s in direct_sites:
        d = abs(cut_position - s)
        if d <= PROX:
            contrib = max(0, 1 - d / PROX)
            direct_risk = max(direct_risk, contrib)
            reasons.append(f'Direct site at pos {s} (d={d}) — risk +{int(contrib*100)}%')

    for s in indirect_sites:
        d = abs(cut_position - s)
        if d <= PROX * 2:
            contrib = max(0, 1 - d / (PROX * 2)) * 0.6
            indirect_risk = max(indirect_risk, contrib)
            if contrib > 0.1:
                reasons.append(f'Interface site at pos {s} (d={d}) — indirect risk +{int(contrib*100)}%')

    overall_risk = max(direct_risk, indirect_risk * 0.7)
    return {
        'direct_risk'          : round(direct_risk, 3),
        'indirect_risk'        : round(indirect_risk, 3),
        'overall_risk'         : round(overall_risk, 3),
        'functional_penalty'   : round(overall_risk, 3),
        'reasons'              : reasons,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5 — FRAGMENT FOLDABILITY
# ═══════════════════════════════════════════════════════════════════════════

def _instability_index(seq: str) -> float:
    if not seq or len(seq) < 2:
        return 40.0
    score = sum(INSTABILITY_DIPEPTIDES.get(seq[i]+seq[i+1], 0) for i in range(len(seq)-1))
    return min(100.0, (score / max(1, len(seq)-1)) * 60 + 30)


def score_fragment_foldability(seq: str, label: str = 'Fragment') -> Dict:
    """
    Per-fragment foldability assessment.
    Ref: Fernandez-Escamilla 2004 (TANGO); Garbuzynskiy 2010 (FoldAmyloid).
    """
    if not seq or len(seq) < 10:
        return {
            'label': label, 'too_short': True, 'foldability': 0,
            'aggregation_risk': 1.0, 'solubility': 0.0, 'overall': 0.0,
            'overall_pct': 0, 'flags': ['Sequence too short (<10 aa)'], 'grade': 'D',
        }

    N = len(seq)
    flags = []

    # 1. Secondary structure propensity (Chou-Fasman)
    helix_sum = sum(_sp(CF_HELIX, aa, 1.0) for aa in seq)
    sheet_sum = sum(_sp(CF_SHEET, aa, 1.0) for aa in seq)
    dis_sum   = sum(_sp(DIS_PROP, aa, 0.0) for aa in seq)
    agg_sum   = sum(_sp(AGG_PROP, aa, 0.0) for aa in seq)

    ordered_ss = (helix_sum + sheet_sum) / (2 * N)
    disordered_frac = min(1.0, max(0.0, dis_sum / N + 0.35))
    intrinsic_foldability = max(0.0, min(1.0,
        ordered_ss * 0.6 * (1 - disordered_frac * 0.4)
    ))

    # 2. Hydrophobic patch
    max_hydro_run = 0
    current_run   = 0
    for aa in seq:
        if _sp(KD_HYDRO, aa, 0) > 1.5:
            current_run += 1
            max_hydro_run = max(max_hydro_run, current_run)
        else:
            current_run = 0
    if max_hydro_run > 6:
        flags.append(f'Hydrophobic patch {max_hydro_run}aa long — high aggregation risk')

    # 3. Aggregation risk
    agg_risk_raw  = agg_sum / N
    aggregation_risk = max(0.0, min(1.0, (agg_risk_raw + 1) / 5))

    # 4. Net charge
    net_charge = sum(1 if aa in 'KR' else -1 if aa in 'DE' else 0 for aa in seq)
    charge_per_res = net_charge / N
    charge_ok = abs(charge_per_res) < 0.15
    if not charge_ok:
        flags.append(f'Extreme net charge ({net_charge:+d}) — may cause solubility issues')

    # 5. Predicted solubility
    predicted_solubility = max(0.0, min(1.0,
        1 - aggregation_risk * 0.5 - min(0.4, max_hydro_run / 20)
    ))

    # 6. Cysteine burden
    cys_count = seq.count('C')
    if cys_count > 2:
        flags.append(f'{cys_count} cysteines — high misfolding risk in reducing environment')
    cys_penalty = min(0.4, cys_count * 0.08)

    # 7. Protease sensitivity (dibasic sites)
    protease_hits = len(re.findall(r'[KR]{2}', seq))
    if protease_hits > 3:
        flags.append(f'{protease_hits} dibasic protease sites — CNS protease sensitivity')
    protease_sensitivity = min(1.0, protease_hits * 0.15)

    # 8. Instability index
    instability_idx = _instability_index(seq)
    if instability_idx > 55:
        flags.append(f'Instability index {instability_idx:.0f} (>55 = unstable)')

    overall = max(0.0, min(1.0,
        intrinsic_foldability
        * (1 - aggregation_risk * 0.35)
        * (1 - cys_penalty)
        * (1 - protease_sensitivity * 0.20)
        * (0.75 if instability_idx > 55 else 1.0)
    ))

    grade = 'A' if overall >= 0.70 else 'B' if overall >= 0.50 else 'C' if overall >= 0.30 else 'D'

    return {
        'label'                : label,
        'length'               : N,
        'intrinsic_foldability': round(intrinsic_foldability, 3),
        'disorder_fraction'    : round(disordered_frac, 3),
        'hydrophobic_patch'    : max_hydro_run,
        'net_charge'           : net_charge,
        'charge_ok'            : charge_ok,
        'aggregation_risk'     : round(aggregation_risk, 3),
        'predicted_solubility' : round(predicted_solubility, 3),
        'cysteine_burden'      : cys_count,
        'protease_sensitivity' : round(protease_sensitivity, 3),
        'instability_index'    : round(instability_idx, 1),
        'overall'              : round(overall, 3),
        'overall_pct'          : round(overall * 100),
        'flags'                : flags,
        'grade'                : grade,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6 — REASSEMBLY GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

SPLIT_SYSTEM_REACH = {
    'split_intein_npu': {'reach_A': 20, 'optimal_A': 10, 'max_A': 35},
    'split_gfp':        {'reach_A': 40, 'optimal_A': 30, 'max_A': 50},
    'fkbp_frb':         {'reach_A': 50, 'optimal_A': 40, 'max_A': 60},
    'leucine_zipper':   {'reach_A': 30, 'optimal_A': 25, 'max_A': 45},
    'nanobit':          {'reach_A': 30, 'optimal_A': 25, 'max_A': 45},
}

def score_reassembly_geometry(cut_position: int,
                               pdb_result: Optional[PDBResult],
                               split_system: str = 'split_intein_npu') -> Dict:
    """
    Estimates terminal accessibility using Cα distance at cut point.
    Ref: Bhaskara & Bhattacharyya PLoS ONE 2011.
    """
    if not pdb_result or pdb_result.error or not pdb_result.residues:
        return {
            'score': 0.60, 'terminal_distance_a': None,
            'stereo_compatible': None,
            'linker_reach': SPLIT_SYSTEM_REACH.get(split_system, {}).get('reach_A', '?'),
            'notes': ['No PDB data — geometry not calculable, neutral score applied'],
        }

    residues = pdb_result.residues
    f1_last  = next((r for r in residues if r.res_seq == cut_position), None)
    f2_first = next((r for r in residues if r.res_seq == cut_position + 1), None)

    if not f1_last or not f2_first or not f1_last.ca or not f2_first.ca:
        return {
            'score': 0.55, 'terminal_distance_a': None,
            'notes': ['Flanking residues not found in PDB — geometry unknown'],
        }

    a, b = f1_last.ca, f2_first.ca
    term_dist = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)
    params = SPLIT_SYSTEM_REACH.get(split_system, {'reach_A':25,'optimal_A':15,'max_A':40})

    if split_system == 'split_intein_npu':
        if term_dist < 10:   score, note = 0.95, 'Excellent — termini very close, ideal intein geometry'
        elif term_dist < 20: score, note = 0.80, 'Good — within intein reach'
        elif term_dist < 35: score, note = 0.55, 'Borderline — may need flexible linker extension'
        else:                score, note = 0.25, 'Poor — termini too far apart for trans-splicing'
    elif split_system == 'split_gfp':
        if term_dist < 30:   score, note = 0.85, 'Good — split-GFP tolerates moderate terminal distances'
        elif term_dist < 50: score, note = 0.65, 'Acceptable for split-GFP'
        else:                score, note = 0.35, 'May fail — termini too far for split-GFP assembly'
    elif split_system == 'fkbp_frb':
        if term_dist < 40:   score, note = 0.80, 'Good — FKBP-FRB tolerates distance'
        elif term_dist < 60: score, note = 0.60, 'Acceptable with optimized linker'
        else:                score, note = 0.35, 'Distant termini — long linkers needed'
    else:
        if term_dist < 25:   score, note = 0.75, 'Acceptable geometry'
        elif term_dist < 45: score, note = 0.55, 'Moderate geometry — add flexible linkers'
        else:                score, note = 0.30, 'Challenging geometry for non-covalent systems'

    return {
        'score'              : round(score, 3),
        'terminal_distance_a': round(term_dist, 1),
        'linker_reach'       : f'{params["reach_A"]}Å ({split_system})',
        'stereo_compatible'  : score >= 0.60,
        'stereo_note'        : note,
        'notes'              : [note, f'Terminal Cα distance: {term_dist:.1f}Å'],
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 7 — FUSION BURDEN MODEL
# ═══════════════════════════════════════════════════════════════════════════

FUSION_TAG_MW_DA = {
    'tfr_rmt_shuttle'  : 7000,
    'rvg29'            : 3300,
    'intein_n_npu'     : 11000,
    'intein_c_npu'     : 4000,
    'intein_n_gfp1_10' : 23000,
    'intein_c_gfp11'   : 1400,
    'fkbp'             : 12000,
    'frb'              : 11800,
    'linker_ggg_x3'    : 1000,
}
FUSION_TAG_NAMES = {
    'tfr_rmt_shuttle'  : 'TfR-RMT VHH shuttle',
    'rvg29'            : 'RVG-29 peptide',
    'intein_n_npu'     : 'Npu InteinN',
    'intein_c_npu'     : 'Npu InteinC',
    'intein_n_gfp1_10' : 'GFP1-10 fragment',
    'intein_c_gfp11'   : 'GFP11 peptide',
    'fkbp'             : 'FKBP12 tag',
    'frb'              : 'FRB domain',
    'linker_ggg_x3'    : '(GGGGS)×3 linker',
}

def score_fusion_burden(f1_mw: int, f2_mw: int,
                         split_system: str = 'split_intein_npu',
                         bbb_strategy: str = 'tfr_rmt') -> Dict:
    flags, warnings = [], []

    f1_tags, f2_tags = [], []

    # Split system tags
    if split_system == 'split_intein_npu':
        f1_tags.append('intein_n_npu')
        f2_tags.insert(0, 'intein_c_npu')
    elif split_system == 'split_gfp':
        f1_tags.append('intein_n_gfp1_10')
        f2_tags.insert(0, 'intein_c_gfp11')
    elif split_system == 'fkbp_frb':
        f1_tags.append('fkbp')
        f2_tags.insert(0, 'frb')

    # BBB shuttles
    if bbb_strategy == 'tfr_rmt':
        f1_tags.insert(0, 'tfr_rmt_shuttle')
        f1_tags.append('linker_ggg_x3')
    elif bbb_strategy == 'rvg29':
        f1_tags.insert(0, 'rvg29')
    f2_tags.append('rvg29')  # F2 always gets RVG-29

    f1_tag_mw = sum(FUSION_TAG_MW_DA.get(k, 0) for k in f1_tags)
    f2_tag_mw = sum(FUSION_TAG_MW_DA.get(k, 0) for k in f2_tags)
    f1_construct_mw = f1_mw + f1_tag_mw
    f2_construct_mw = f2_mw + f2_tag_mw

    def construct_score(mw):
        if mw <= 10000:  return 0.90
        if mw <= 15000:  return 0.75
        if mw <= 20000:  return 0.55
        if mw <= 30000:  return 0.35
        if mw <= 50000:  return 0.20
        return 0.08

    f1_score = construct_score(f1_construct_mw)
    f2_score = construct_score(f2_construct_mw)

    if split_system == 'split_intein_npu' and bbb_strategy == 'tfr_rmt':
        warnings.append('F1 has shuttle + inteinN — confirm (GGGGS)×3 linker prevents steric clash')
    if bbb_strategy == 'tfr_rmt' and f1_mw > 20000:
        warnings.append('F1 >20kDa with TfR shuttle — large cargo may reduce TfR binding efficiency')
    max_mw = max(f1_construct_mw, f2_construct_mw)
    if max_mw > 40000:
        flags.append(f'Max construct {round(max_mw/1000)}kDa — protein biologic delivery unlikely; use dual-AAV')

    overall = (f1_score + f2_score) / 2

    return {
        'f1_tags'          : [FUSION_TAG_NAMES.get(k, k) for k in f1_tags],
        'f2_tags'          : [FUSION_TAG_NAMES.get(k, k) for k in f2_tags],
        'f1_construct_mw'  : f1_construct_mw,
        'f2_construct_mw'  : f2_construct_mw,
        'f1_fusion_score'  : round(f1_score, 3),
        'f2_fusion_score'  : round(f2_score, 3),
        'overall_burden'   : round(overall, 3),
        'overall_burden_pct': round(overall * 100),
        'flags'            : flags,
        'warnings'         : warnings,
        'recommendation'   : (
            'Construct size manageable — proceed with protein biologic delivery' if overall >= 0.65
            else 'Construct large — consider dual-AAV or reduce fragment/tag size' if overall >= 0.40
            else 'Construct too large for protein delivery — dual-AAV or mRNA-LNP required'
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 8 — OLIGOMERIZATION / ASSEMBLY CONTEXT
# ═══════════════════════════════════════════════════════════════════════════

RABV_ASSEMBLY_STATES = {
    'P': {
        'state': 'dimer',
        'dimer_interface': list(range(189, 298)),
        'safe_zone': {'start': 100, 'end': 180, 'reason': 'Flexible linker between N-domain and dimerization domain'},
        'dimer_region': {'start': 189, 'end': 297},
        'monomer_region': {'start': 1, 'end': 130},
        'note': 'P constitutive dimer via C-terminal domain. N-domain (aa 1-130) is monomeric and disordered. Safe cut zone: aa 100-180.',
    },
    'N': {
        'state': '11-mer',
        'oligomer_interface': list(range(1, 23)) + list(range(377, 451)),
        'monomer_core': {'start': 23, 'end': 376},
        'rna_groove': {'start': 246, 'end': 275},
        'note': 'N forms 11-mer helical rings (RCSB 8FFR). N-arm (1-22) and C-arm (377-450) are oligomerization arms — MUST NOT be cut.',
    },
    'L': {
        'state': 'monomer_complex',
        'domains': [
            {'name': 'RdRp core',      'start': 1,    'end': 540,  'type': 'catalytic'},
            {'name': 'Linker 1',       'start': 541,  'end': 560,  'type': 'linker'},
            {'name': 'Capping domain', 'start': 561,  'end': 900,  'type': 'catalytic'},
            {'name': 'Linker 2',       'start': 901,  'end': 920,  'type': 'linker'},
            {'name': 'MTase domain',   'start': 921,  'end': 1700, 'type': 'catalytic'},
            {'name': 'Linker 3',       'start': 1701, 'end': 1720, 'type': 'linker'},
            {'name': 'C-terminal',     'start': 1721, 'end': 2142, 'type': 'scaffold'},
        ],
        'note': 'L monomeric but complexes with P. Only three safe linkers: ~aa 541-560, 901-920, 1701-1720.',
    },
}


def score_assembly_context(cut_position: int, target_id: Optional[str] = None,
                            custom_assembly: Optional[Dict] = None) -> Dict:
    """
    Score: assemblyRisk 0-1 (lower = better).
    Ref: Levy et al. Structure 2006 (PDB assembly analysis).
    """
    asm = custom_assembly or RABV_ASSEMBLY_STATES.get(target_id or '', None)
    if not asm:
        return {
            'assembly_risk': 0.10, 'assembly_class': 'unknown',
            'state': 'unknown',
            'notes': ['No assembly data — default low risk assumed'],
            'recommendation': 'No assembly data available',
        }

    notes = []
    assembly_risk  = 0.10
    assembly_class = 'monomer_safe'
    state = asm.get('state', 'unknown')

    if state == 'dimer':
        dimer_face = asm.get('dimer_interface', [])
        in_dimer = any(abs(p - cut_position) <= 5 for p in dimer_face)
        safe = asm.get('safe_zone', {})
        dimer_reg = asm.get('dimer_region', {})
        if in_dimer:
            assembly_risk = 0.90
            assembly_class = 'dimer_interface_break'
            notes.append('CRITICAL: Cut is within dimer interface — will destroy dimerization')
        elif safe and safe['start'] <= cut_position <= safe['end']:
            assembly_risk = 0.08
            assembly_class = 'safe_zone'
            notes.append(f'Safe zone ({safe["reason"]}) — low assembly risk')
        elif dimer_reg and cut_position >= dimer_reg['start']:
            assembly_risk = 0.75
            assembly_class = 'dimer_region'
            notes.append(f'In dimerization domain (aa {dimer_reg["start"]}-{dimer_reg["end"]}) — high risk')

    elif state == '11-mer':
        oligo_face = asm.get('oligomer_interface', [])
        in_oligo = any(abs(p - cut_position) <= 8 for p in oligo_face)
        core = asm.get('monomer_core', {})
        rna = asm.get('rna_groove', {})
        if in_oligo:
            assembly_risk = 0.88
            assembly_class = 'oligomer_interface_break'
            notes.append('CRITICAL: Cut near oligomerization surface — likely destroys 11-mer assembly')
        elif core and core['start'] <= cut_position <= core['end']:
            if rna and rna['start'] <= cut_position <= rna['end']:
                assembly_risk = 0.70
                assembly_class = 'rna_groove_region'
                notes.append('In RNA binding groove — high functional risk')
            else:
                assembly_risk = 0.15
                assembly_class = 'monomer_core_safe'
                notes.append('In monomer core outside N/C-arms — oligomerization interfaces preserved')

    elif state == 'monomer_complex':
        domains = asm.get('domains', [])
        cut_dom = next((d for d in domains if d['start'] <= cut_position <= d['end']), None)
        if cut_dom:
            if cut_dom['type'] == 'linker':
                assembly_risk = 0.05
                assembly_class = 'domain_linker'
                notes.append(f'In safe interdomain linker "{cut_dom["name"]}"')
            elif cut_dom['type'] == 'catalytic':
                assembly_risk = 0.95
                assembly_class = 'catalytic_domain_break'
                notes.append(f'CRITICAL: In catalytic domain "{cut_dom["name"]}" — will destroy enzymatic activity')
            else:
                assembly_risk = 0.50
                assembly_class = 'scaffold_domain'
                notes.append(f'In scaffold domain "{cut_dom["name"]}" — structural risk')

    notes.append(f'Assembly state: {state}')
    return {
        'assembly_risk'   : round(assembly_risk, 3),
        'assembly_class'  : assembly_class,
        'state'           : state,
        'notes'           : notes,
        'recommendation'  : (
            'Proceed — assembly context favorable' if assembly_risk < 0.20
            else 'Caution — assembly context moderately unfavorable' if assembly_risk < 0.55
            else 'REJECT — cut destroys critical assembly interface'
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 10 — BENCHMARK CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

BENCHMARK_SPLITS = [
    {'protein': 'GFP',              'position': 214, 'outcome': 'success', 'note': 'Split-GFP 1-10/11 (Cabantous 2005)'},
    {'protein': 'GFP',              'position': 157, 'outcome': 'success', 'note': 'Alternative split-GFP site'},
    {'protein': 'Firefly luciferase','position': 437, 'outcome': 'success', 'note': 'Split-Luc (Dixon 2016)'},
    {'protein': 'Cas9',             'position': 573, 'outcome': 'success', 'note': 'Split-Cas9 dual-AAV (Truong 2015)'},
    {'protein': 'Cas9',             'position': 637, 'outcome': 'success', 'note': 'Split-Cas9 alternative (Chew 2016)'},
    {'protein': 'TEV protease',     'position': 118, 'outcome': 'success', 'note': 'Reconstituted TEV (Mootz 2003)'},
    {'protein': 'GFP',              'position': 100, 'outcome': 'failure', 'note': 'Core β-barrel — buried'},
    {'protein': 'GFP',              'position': 180, 'outcome': 'failure', 'note': 'Sheet interior — aggregation'},
    {'protein': 'DHFR',             'position': 1,   'outcome': 'failure', 'note': 'N-terminus — F1 too short'},
]


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 11 — RABV TARGET MAPS
# ═══════════════════════════════════════════════════════════════════════════

RABV_TARGET_MAPS = {
    'P': {
        'id': 'P', 'name': 'RABV Phosphoprotein (P)',
        'short_name': 'P protein', 'mw_kda': 33, 'length': 297,
        'uniprot_id': 'P16285', 'pdb': '7C20,3OA1',
        'sequence': 'MDADKIVFKVNNQVVSLKPEIIVDQYEYKYPAIKDLKKPCITLGQVDNKTYVDKQLQNFEKGVTIDFDLASSRLSERNFISFTDLEYNSSPFVTTPTVSIQEQRLDSITEDPGSSGTTESTDISRLNDALRRNMEEVLAQIRPAEDPTPNRAAQQPEMEWSRALNTIYLNQQNLRIQKQVSETEGIEGLAQDDSVTIAQNNTQTKVVNDSGLNYMKSNVKQVKKMADEFERNLKESQRPVHFLNFGTLNLSIVREKQKTLHSANVKDQYEMESLFHSTPGVSTQRRGDLNQSYRRILDPNAKMTVDLNQTVSTTQEAYRQIMFNLAK',
        'domains': [
            {'name': 'N-terminal IDR',         'start': 1,   'end': 50,  'type': 'disordered'},
            {'name': 'N-terminal domain',       'start': 51,  'end': 130, 'type': 'scaffold'},
            {'name': 'Flexible linker',         'start': 131, 'end': 172, 'type': 'linker'},
            {'name': 'Oligomerization domain',  'start': 173, 'end': 202, 'type': 'scaffold'},
            {'name': 'Dimerization domain',     'start': 203, 'end': 297, 'type': 'binding'},
        ],
        'annotations': {
            'active_site'     : [],
            'binding_hotspot' : [218,219,220,221,222,223,224,225],
            'interface'       : [179,189,195,202,215,240,260,275,285],
            'forbidden'       : [179,218,219,220,221,222,223,224,225],
            'known_good_splits': [100,120,140,155,160,165],
        },
        'note': 'Best split zone: aa 100-165 (linker between N-domain and oligomerization domain). LC8 binding (aa 218-225) via Pep2 precedent; TBK1 interface (Ser179) via split-nanobody. Refs: Ribeiro 2009; Wiltzer 2014; Rahmati 2025.',
    },
    'N': {
        'id': 'N', 'name': 'RABV Nucleoprotein (N)',
        'short_name': 'N protein', 'mw_kda': 47, 'length': 450,
        'uniprot_id': 'P06025', 'pdb': '8FFR',
        'sequence': 'MDADKIVFKVNNQVVSLKPEIIVKMDVNPKDEVLNKLNELKQRLEEMGDPEEQVVMAIPSWQHLYQKSTMGPQHPNPHLSYMVDVLQPPQPDNHNDRDRQHYENNQEFWKEHLDRLRLEQGGDQATNLRKVLNGLRQFAIGNDVTPFNRFVDGEEALVLKKNMEIAHFGTPFQHINDTKKDEYEFLSNKNMDDPQVFLMDQQLEQKLLEAQPTLELTLAIHKLRNVSSDNKGYSIQDTDNRGEGIQKFLKRMIMQMNDNHSDKVAEGIASCLLDLKDKIIEQINKLLDSDFVTKKQLITPKIPAIAQAAALDGPYQLKSKNPNLATILNAIQLTVKMSEDLKLQRYAQNVKQLIDLKMEQESGPKIDTIEQINQENIKKMANDMVNRQKSMTEKVTMRHTREKQQVVPVKQALVVSHYENMDPIIAEEGDNMIDFQHPYNSSLFKQDAIILRVQQLMNPQLQEFLQSSQERLA',
        'domains': [
            {'name': 'N-arm',           'start': 1,   'end': 22,  'type': 'binding'},
            {'name': 'N-terminal lobe', 'start': 23,  'end': 220, 'type': 'scaffold'},
            {'name': 'Central linker',  'start': 221, 'end': 260, 'type': 'linker'},
            {'name': 'C-terminal lobe', 'start': 261, 'end': 375, 'type': 'scaffold'},
            {'name': 'C-arm',           'start': 376, 'end': 450, 'type': 'binding'},
        ],
        'annotations': {
            'active_site'     : [],
            'binding_hotspot' : [246,247,248,249,250,251,252,253,254,255],
            'interface'       : [1,2,3,4,5,6,7,8,101,102,103,370,371,372,373,374,375,376,420],
            'forbidden'       : [1,2,3,4,5,101,102,103,246,247,248,249,250,251,370,371,372,376],
            'known_good_splits': [200,220,235,245,255,265],
        },
        'note': '11-mer helical nucleocapsid (RCSB 8FFR). N-arm (1-22) and C-arm (376-450) MUST NOT be cut. Central linker (aa 221-260) is the only safe zone. Refs: Scrima 2008; Green 2014; Blondel 2012.',
    },
    'L': {
        'id': 'L', 'name': 'RABV L Protein (RdRp)',
        'short_name': 'L protein (RdRp)', 'mw_kda': 240, 'length': 2142,
        'uniprot_id': 'P06029', 'pdb': '—',
        'sequence': None,
        'domains': RABV_ASSEMBLY_STATES['L']['domains'],
        'annotations': {
            'active_site'     : [831,832,833],
            'binding_hotspot' : [740,741,742,743,744,745],
            'interface'       : [1,2,3,4,5,6,7,8,2138,2139,2140,2141,2142],
            'forbidden'       : [829,830,831,832,833,834,835],
            'known_good_splits': [545,555,905,915,1705,1715],
        },
        'high_uncertainty': True,
        'note': 'Only three safe interdomain linkers: ~aa 541-560, 901-920, 1701-1720. All fragments >60kDa — dual-AAV gene delivery ONLY. L structure largely unresolved (partial cryo-EM: Loureiro 2023). Dominant-negative approach (Architecture C) is theoretical.',
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 14 — MANUFACTURABILITY
# ═══════════════════════════════════════════════════════════════════════════

def score_manufacturability(f1_seq: str, f2_seq: str,
                             split_system: str = 'split_intein_npu',
                             delivery_mode: str = 'protein_biologic') -> Dict:
    """
    Full manufacturability assessment:
    expression, purification, cysteine, instability, repeats,
    conjugate complexity, aggregation, shelf-life, GMP cost index.
    """
    f1_fold = score_fragment_foldability(f1_seq or '', 'F1')
    f2_fold = score_fragment_foldability(f2_seq or '', 'F2')

    flags, notes = [], []
    scores = {}

    # 1. Expression ease
    avg_fold = (f1_fold['overall'] + f2_fold['overall']) / 2
    scores['expression_ease'] = round(avg_fold * 100)
    if avg_fold < 0.50:
        flags.append('Poor fragment foldability — expression yield may be low')

    # 2. Purification ease
    f1_charge = abs(f1_fold['net_charge'])
    f2_charge = abs(f2_fold['net_charge'])
    charge_score = min(1.0, (f1_charge + f2_charge) / 10 + 0.5)
    hydro_pen = min(0.4, (f1_fold['hydrophobic_patch'] + f2_fold['hydrophobic_patch']) / 30)
    scores['purification_ease'] = round(charge_score * (1 - hydro_pen) * 100)

    # 3. Cysteine risk
    total_cys = f1_fold['cysteine_burden'] + f2_fold['cysteine_burden']
    scores['cysteine_risk'] = max(0, round(100 - total_cys * 12))
    if total_cys > 4:
        flags.append(f'{total_cys} cysteines total — aggregation/disulfide risk')

    # 4. Instability risk
    avg_instability = (f1_fold['instability_index'] + f2_fold['instability_index']) / 2
    scores['instability_risk'] = max(0, round(100 - max(0, avg_instability - 30) * 1.5))
    if avg_instability > 55:
        flags.append(f'Avg instability index {avg_instability:.0f} — sequence optimization recommended')

    # 5. Repeat risk (simplified tandem dipeptide)
    def count_repeats(seq):
        n = 0
        for i in range(len(seq or '') - 3):
            if seq[i] == seq[i+2] and seq[i+1] == seq[i+3]:
                n += 1
        return n
    total_repeats = count_repeats(f1_seq) + count_repeats(f2_seq)
    scores['repeat_risk'] = max(0, round(100 - total_repeats * 3))
    if total_repeats > 10:
        flags.append(f'{total_repeats} internal repeats — cloning instability risk')

    # 6. Conjugate complexity
    sys_steps = {'split_intein_npu':2,'split_gfp':2,'fkbp_frb':3,'leucine_zipper':1,'nanobit':1}.get(split_system, 2)
    deliv_steps = 2 if delivery_mode == 'protein_biologic' else 0
    conj_steps = sys_steps + deliv_steps
    scores['conjugate_complexity'] = max(0, round(100 - conj_steps * 12))
    if conj_steps > 4:
        flags.append(f'{conj_steps} conjugation steps — complex manufacturing')

    # 7. Aggregation risk
    avg_agg = (f1_fold['aggregation_risk'] + f2_fold['aggregation_risk']) / 2
    scores['aggregation_risk'] = round((1 - avg_agg) * 100)

    # 8. Shelf-life estimate
    is_covalent = split_system == 'split_intein_npu'
    scores['shelf_life'] = 75 if is_covalent else 50
    if not is_covalent:
        notes.append('Non-covalent assembly — shelf-life limited; intein system preferred')

    # Overall
    feasibility = round(
        scores['expression_ease']     * 0.20 +
        scores['purification_ease']   * 0.15 +
        scores['cysteine_risk']       * 0.15 +
        scores['instability_risk']    * 0.10 +
        scores['repeat_risk']         * 0.08 +
        scores['conjugate_complexity']* 0.12 +
        scores['aggregation_risk']    * 0.12 +
        scores['shelf_life']          * 0.08
    )
    grade = 'A' if feasibility >= 75 else 'B' if feasibility >= 60 else 'C' if feasibility >= 45 else 'D' if feasibility >= 30 else 'F'
    limiting = min(scores, key=scores.get)

    return {
        'feasibility'   : feasibility,
        'grade'         : grade,
        'sub_scores'    : scores,
        'limiting_factor': limiting,
        'flags'         : flags,
        'notes'         : notes,
        'f1_assessment' : {k: f1_fold[k] for k in ('overall','aggregation_risk','cysteine_burden','instability_index')},
        'f2_assessment' : {k: f2_fold[k] for k in ('overall','aggregation_risk','cysteine_burden','instability_index')},
        'cost_note'     : (
            f'Feasible for GMP development (grade {grade})' if feasibility >= 65
            else f'Significant manufacturing challenges (grade {grade})' if feasibility >= 45
            else f'Manufacturing bottleneck — redesign recommended (grade {grade})'
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SEQUENCE ENGINE — FASTA / RESIDUE PROPERTIES / CANDIDATE FINDER
# ═══════════════════════════════════════════════════════════════════════════

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

def parse_fasta(text: str) -> Dict:
    if not text or not text.strip():
        return {'error': 'Empty input', 'name': '', 'sequence': '', 'length': 0}
    lines = text.strip().splitlines()
    name, seq_lines = 'Protein', []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            name = line[1:].strip().split()[0] or 'Protein'
        else:
            seq_lines.append(''.join(c for c in line.upper() if c in VALID_AA))
    sequence = ''.join(seq_lines)
    if not sequence:
        return {'error': 'No valid amino acid sequence found', 'name': name, 'sequence': '', 'length': 0}
    return {'name': name, 'sequence': sequence, 'length': len(sequence), 'error': None}


def _smooth(arr: List[float], w: int) -> List[float]:
    N = len(arr)
    result = []
    for i in range(N):
        lo, hi = max(0, i-w), min(N-1, i+w)
        result.append(sum(arr[lo:hi+1]) / (hi - lo + 1))
    return result


def estimate_residue_properties(sequence: str) -> List[Dict]:
    """Sequence-based per-residue properties (Chou-Fasman + KD + Janin + disorder)."""
    N = len(sequence)
    W = 7

    raw_helix  = [_sp(CF_HELIX,  aa, 1.0) for aa in sequence]
    raw_sheet  = [_sp(CF_SHEET,  aa, 1.0) for aa in sequence]
    raw_hydro  = [_sp(KD_HYDRO,  aa, 0.0) for aa in sequence]
    raw_burial = [_sp(JN_BURIAL, aa, 0.0) for aa in sequence]
    raw_disord = [_sp(DIS_PROP,  aa, 0.0) for aa in sequence]

    s_helix  = _smooth(raw_helix,  W)
    s_sheet  = _smooth(raw_sheet,  W)
    s_hydro  = _smooth(raw_hydro,  W)
    s_burial = _smooth(raw_burial, W)
    s_disord = _smooth(raw_disord, W)

    def ss_call(i):
        if s_helix[i] > 1.05 and s_helix[i] >= s_sheet[i]: return 'helix'
        if s_sheet[i] > 1.05 and s_sheet[i] >= s_helix[i]: return 'sheet'
        return 'loop'

    def accessibility(i):
        raw = 0.50 - s_burial[i] * 0.30 - s_hydro[i] * 0.04
        return max(0.0, min(1.0, raw + 0.50))

    def conservation_penalty(i):
        lo, hi = max(0, i-W), min(N-1, i+W)
        window = sequence[lo:hi+1]
        freq: Dict[str, int] = {}
        for c in window:
            freq[c] = freq.get(c, 0) + 1
        total = len(window)
        H = 0.0
        for cnt in freq.values():
            p = cnt / total
            H -= p * math.log2(p)
        return max(0.0, 1 - H / 3.0)

    return [
        {
            'index'        : i,
            'position'     : i + 1,
            'aa'           : sequence[i],
            'ss'           : ss_call(i),
            'accessibility': round(accessibility(i), 3),
            'conservation' : round(conservation_penalty(i), 3),
            'hydrophobicity': round(s_hydro[i], 3),
            'disorder'     : round(min(1.0, max(0.0, s_disord[i] + 0.35)), 3),
            'has_structure': False,
            'plddt'        : None,
        }
        for i in range(N)
    ]


def apply_structure_to_props(props: List[Dict],
                              pdb_result: Optional[PDBResult],
                              af_result: Optional[Dict] = None) -> List[Dict]:
    """Override sequence heuristics with real structural data."""
    if not pdb_result or pdb_result.error or not pdb_result.residues:
        return props

    struct_map = {r.res_seq: r for r in pdb_result.residues}
    plddt_map  = {}
    if af_result and not af_result.get('error'):
        for p in af_result.get('plddt', []):
            plddt_map[p['res_seq']] = p

    enriched = []
    for prop in props:
        res = struct_map.get(prop['position'])
        af  = plddt_map.get(prop['position'])
        if not res or not res.ca:
            enriched.append(prop)
            continue

        b_norm = min(1.0, res.b_factor / 60)
        disorder = prop['disorder']
        if af:
            plddt = af['plddt']
            disorder = (0.85 if plddt < 50 else 0.55 if plddt < 70 else 0.20 if plddt < 90 else 0.05)
        else:
            disorder = max(prop['disorder'], min(0.95, b_norm * 0.8))

        accessibility = prop['accessibility']
        if res.cb and b_norm < 0.2:
            accessibility *= 0.5
        if not res.cb and prop['ss'] == 'loop':
            accessibility = max(prop['accessibility'], 0.50)

        enriched.append({
            **prop,
            'accessibility': round(accessibility, 3),
            'disorder'     : round(disorder, 3),
            'b_factor'     : res.b_factor,
            'has_structure': True,
            'plddt'        : af['plddt'] if af else None,
        })
    return enriched


SPLIT_HARD_RULES = {
    'min_fragment_aa'  : 50,
    'acc_threshold'    : 0.35,
    'cons_threshold'   : 0.65,
    'min_dist_to_site' : 8,
}


def find_split_site_candidates(props: List[Dict],
                                annotations: Optional[Dict] = None,
                                constraints: Optional[Dict] = None) -> List[Dict]:
    """SPELL-inspired split-site finder. Ref: Klingen 2014 (SPELL-derived logic)."""
    annotations = annotations or {}
    rules = {**SPLIT_HARD_RULES, **(constraints or {})}
    N = len(props)

    forbidden: set = set()
    for key in ('active_site', 'binding_hotspot', 'interface', 'forbidden'):
        for pos in (annotations.get(key) or []):
            forbidden.add(pos)
    forbidden_expanded: set = set()
    for pos in forbidden:
        for d in range(-rules['min_dist_to_site'], rules['min_dist_to_site'] + 1):
            forbidden_expanded.add(pos + d)

    candidates = []
    for i in range(N - 1):
        pos1  = i + 1
        f1len = pos1
        f2len = N - pos1

        if f1len < rules['min_fragment_aa']: continue
        if f2len < rules['min_fragment_aa']: continue
        if props[i]['ss'] != 'loop':         continue
        if props[i]['accessibility'] < rules['acc_threshold']:  continue
        if props[i]['conservation']  > rules['cons_threshold']: continue
        if pos1 in forbidden_expanded:       continue

        ss_score   = 1.0
        acc_score  = min(1.0, (props[i]['accessibility'] - rules['acc_threshold'])
                         / (1 - rules['acc_threshold']))
        cons_score = 1 - props[i]['conservation']

        min_dist = min((abs(pos1 - fp) for fp in forbidden), default=N)
        dist_score = min(1.0, min_dist / 30)

        balance   = min(f1len, f2len) / max(f1len, f2len)
        bal_score = 0.50 + 0.50 * balance
        dis_score = 0.40 + 0.60 * props[i]['disorder']

        # AF2 pLDDT bonus
        plddt_bonus = 0.05 if (props[i].get('plddt') and 40 <= props[i]['plddt'] < 70) else 0.0

        split_score = (
            ss_score   * 0.28 +
            acc_score  * 0.22 +
            cons_score * 0.20 +
            dist_score * 0.16 +
            bal_score  * 0.09 +
            dis_score  * 0.05
        ) + plddt_bonus

        split_score = min(1.0, split_score)
        verdict = ('Excellent' if split_score >= 0.70 else 'Good' if split_score >= 0.50
                   else 'Marginal' if split_score >= 0.35 else 'Poor')

        candidates.append({
            'position'      : pos1,
            'aa'            : props[i]['aa'],
            'ss'            : props[i]['ss'],
            'f1_length'     : f1len,
            'f2_length'     : f2len,
            'has_structure' : props[i].get('has_structure', False),
            'plddt'         : props[i].get('plddt'),
            'b_factor'      : props[i].get('b_factor'),
            'scores'        : {
                'loop'         : round(ss_score, 3),
                'accessibility': round(acc_score, 3),
                'conservation' : round(cons_score, 3),
                'func_distance': round(dist_score, 3),
                'balance'      : round(bal_score, 3),
                'disorder'     : round(dis_score, 3),
            },
            'split_score'   : round(split_score, 3),
            'verdict'       : verdict,
            'min_dist_to_site': min_dist if min_dist < N else None,
        })

    candidates.sort(key=lambda x: -x['split_score'])
    return candidates


def estimate_fragment_mw(seq: str) -> int:
    return len(seq or '') * 110


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 9 — UNIFIED TOTAL SCORER
# ═══════════════════════════════════════════════════════════════════════════

def score_split_delivery_total(
    split_site: Dict,
    sequence: str,
    split_system: str = 'split_intein_npu',
    conc_nm: float = 5.0,
    delivery_mode: str = 'protein_biologic',
    bbb_strategy: str = 'tfr_rmt',
    target_tissue: str = 'cns',
    annotations: Optional[Dict] = None,
    domains: Optional[List[Dict]] = None,
    target_id: Optional[str] = None,
    pdb_result: Optional[PDBResult] = None,
    af_result: Optional[Dict] = None,
    contact_map: Optional[Dict] = None,
    barrier_state: Optional[Dict] = None,
    motif_hits: Optional[List[Dict]] = None,
) -> Dict:
    """
    Phase 9 composite: S_total = weighted product of all 15-phase sub-scores.
    Mirrors scoreSplitDeliveryTotal() in splitter_engine.js v10.0.
    """
    annotations = annotations or {}
    domains     = domains or []
    motif_hits  = motif_hits or detect_functional_motifs(sequence)

    cut_pos = split_site.get('position', len(sequence) // 2)
    f1_seq  = sequence[:cut_pos]
    f2_seq  = sequence[cut_pos:]
    f1_mw   = estimate_fragment_mw(f1_seq)
    f2_mw   = estimate_fragment_mw(f2_seq)

    # --- Phase 2: contact disruption
    cd = score_contact_disruption(cut_pos, contact_map, annotations)

    # --- Phase 3: domain integrity
    parsed_domains = parse_domain_boundaries(domains, len(sequence))
    ds = score_domain_integrity(cut_pos, parsed_domains, len(sequence))

    # --- Phase 4: functional risk
    fr = score_functional_risk(cut_pos, sequence, annotations, motif_hits)

    # --- Phase 5: foldability
    f1_fold = score_fragment_foldability(f1_seq, 'F1')
    f2_fold = score_fragment_foldability(f2_seq, 'F2')

    # --- Phase 6: geometry
    geo = score_reassembly_geometry(cut_pos, pdb_result, split_system)

    # --- Phase 7: fusion burden
    fb = score_fusion_burden(f1_mw, f2_mw, split_system, bbb_strategy)

    # --- Phase 8: assembly context
    asm = score_assembly_context(cut_pos, target_id)

    # --- Phase 14: manufacturability
    mfg = score_manufacturability(f1_seq, f2_seq, split_system, delivery_mode)

    # --- Degron risk
    degron = _terminus_degron_risk(f1_seq, f2_seq)

    # --- Reassembly probability (simplified Michaelis-Menten)
    kd_map = {
        'split_intein_npu': 0.001,
        'split_gfp': 0.5,
        'fkbp_frb': 0.2,
        'leucine_zipper': 100.0,
        'nanobit': 190000.0,
    }
    kd = kd_map.get(split_system, 1.0)
    reassembly_prob = min(0.95, (conc_nm / (conc_nm + kd)) * 0.85)
    is_covalent = (split_system == 'split_intein_npu')

    # --- Sub-scores (0-1)
    S_site    = split_site.get('split_score', 0.5)
    S_contact = 1 - cd['penalty']
    S_domain  = ds['score']
    S_func    = 1 - fr['overall_risk']
    S_asm     = 1 - asm['assembly_risk']
    S_fold    = min(f1_fold['overall'], f2_fold['overall'])
    S_geo     = geo['score']
    S_fusion  = fb['overall_burden']
    S_reassem = reassembly_prob
    S_mfg     = mfg['feasibility'] / 100

    has_struct = bool(pdb_result and not pdb_result.error)
    struct_w   = 1.0 if has_struct else 0.70

    base_score = (
        S_site    * 0.18 +
        S_contact * 0.14 * struct_w +
        S_domain  * 0.14 +
        S_func    * 0.12 +
        S_asm     * 0.10 +
        S_fold    * 0.10 +
        S_geo     * 0.08 * struct_w +
        S_fusion  * 0.06 +
        S_reassem * 0.05 +
        S_mfg     * 0.03
    )

    has_annot = any(annotations.get(k) for k in ('active_site','binding_hotspot','interface','forbidden'))
    seq_qual  = min(1.0, len(sequence) / 100)
    U = 0.20 + (0 if has_annot else 0.10) + (0 if has_struct else 0.12) + (1 - seq_qual) * 0.10
    if split_system == 'fkbp_frb': U += 0.05
    if split_system == 'nanobit':  U += 0.15

    final_score = max(0, min(100, round(base_score * (1 - 0.30 * min(1.0, U)) * 100)))
    verdict = ('Excellent' if final_score >= 65 else 'Good' if final_score >= 45
               else 'Marginal' if final_score >= 25 else 'Poor')

    # --- Rejection check
    is_rejected, rejection_reason = _check_rejection(split_site, cd, ds, fr, fb, asm, f1_fold, f2_fold)

    # --- Waterfall steps
    steps = _compute_waterfall(split_site, cd, ds, fr, asm, f1_fold, f2_fold, geo, fb, mfg)

    return {
        'position'          : cut_pos,
        'aa'                : split_site.get('aa', ''),
        'f1_length'         : len(f1_seq),
        'f2_length'         : len(f2_seq),
        'f1_mw'             : f1_mw,
        'f2_mw'             : f2_mw,
        'split_score'       : split_site.get('split_score', 0.5),
        'split_site'        : split_site,
        'f1_seq'            : f1_seq[:50] + '…' if len(f1_seq) > 50 else f1_seq,
        'f2_seq'            : f2_seq[:50] + '…' if len(f2_seq) > 50 else f2_seq,
        'degron_risk'       : degron,
        'subscores': {
            'site_spell'        : round(S_site,    3),
            'contact_integrity' : round(S_contact, 3),
            'domain_integrity'  : round(S_domain,  3),
            'functional_safety' : round(S_func,    3),
            'assembly_context'  : round(S_asm,     3),
            'frag_foldability'  : round(S_fold,    3),
            'reassembly_geo'    : round(S_geo,     3),
            'fusion_burden'     : round(S_fusion,  3),
            'reassembly_prob'   : round(S_reassem, 3),
            'manufacturability' : round(S_mfg,     3),
        },
        'contact_disruption': cd,
        'domain_score'      : ds,
        'functional_risk'   : fr,
        'assembly_score'    : asm,
        'f1_foldability'    : f1_fold,
        'f2_foldability'    : f2_fold,
        'geometry'          : geo,
        'fusion_burden'     : fb,
        'manufacturability' : mfg,
        'reassembly_detail' : {
            'probability'    : round(reassembly_prob, 4),
            'probability_pct': round(reassembly_prob * 100),
            'kd_nm'          : kd,
            'covalent'       : is_covalent,
        },
        'motif_hits'        : motif_hits,
        'uncertainty'       : round(U, 3),
        'base_score'        : round(base_score, 3),
        'final_score'       : final_score,
        'verdict'           : verdict,
        'is_rejected'       : is_rejected,
        'rejection_reason'  : rejection_reason,
        'waterfall_steps'   : steps,
        'has_structure'     : has_struct,
        'has_annotations'   : has_annot,
        'score_range'       : {'low': max(0, final_score - 15), 'high': min(100, final_score + 15)},
    }


def _terminus_degron_risk(f1_seq: str, f2_seq: str) -> Dict:
    levels = ['low', 'medium', 'high']
    f1c = 'medium' if f1_seq and f1_seq[-1] in N_DEGRON_AA else 'low'
    f2n = 'high'   if f2_seq and f2_seq[0]  in N_DEGRON_AA else 'low'
    f2c = 'medium' if f2_seq and f2_seq[-1] in N_DEGRON_AA else 'low'
    worst = max(levels.index(f1c), levels.index(f2n), levels.index(f2c))
    return {'f1_cterminus': f1c, 'f2_nterminus': f2n, 'f2_cterminus': f2c, 'overall': levels[worst]}


def _check_rejection(site, cd, ds, fr, fb, asm, f1_fold, f2_fold) -> Tuple[bool, Optional[str]]:
    f1l = site.get('f1_length', 999)
    f2l = site.get('f2_length', 999)
    acc = (site.get('scores') or {}).get('accessibility', 1.0)
    reasons = []
    if acc < 0.20:
        reasons.append('Cuts within buried hydrophobic core')
    if ds and ds.get('classification') in ('catalytic_domain', 'intra_domain_scaffold'):
        reasons.append('Cuts catalytic scaffold or active site scaffold')
    if fr and fr.get('direct_risk', 0) > 0.75:
        reasons.append('Cuts protein-protein binding interface')
    if f1l > 10 and f2l > 10:
        ratio = max(f1l, f2l) / max(1, min(f1l, f2l))
        if ratio > 5:
            reasons.append('Fragment size ratio exceeds 5:1 (excessive asymmetry)')
    if asm and asm.get('assembly_class') in ('dimer_interface_break', 'oligomer_interface_break', 'catalytic_domain_break'):
        reasons.append(f'Cuts critical assembly interface ({asm["assembly_class"]})')
    if cd and cd.get('category') == 'critical':
        reasons.append('Contact disruption is critical — breaks core packing contacts')
    if f1l < 40 or f2l < 40:
        reasons.append('One or both fragments below minimum viable length (40aa)')
    if f1_fold and f2_fold and f1_fold.get('aggregation_risk',0) > 0.80 and f2_fold.get('aggregation_risk',0) > 0.80:
        reasons.append('Both fragments show catastrophically high aggregation risk')
    return (bool(reasons), reasons[0] if reasons else None)


def _compute_waterfall(site, cd, ds, fr, asm, f1_fold, f2_fold, geo, fb, mfg) -> List[Dict]:
    spell = round((site.get('split_score', 0.5)) * 100)
    steps = [{'label': 'Base SPELL site score', 'delta': spell - 100, 'running': spell, 'note': ''}]
    running = spell

    if cd and cd.get('category') != 'no_structure':
        d = -round(cd['penalty'] * 40)
        running = max(0, running + d)
        steps.append({'label': f'Contact disruption ({cd["category"]})', 'delta': d, 'running': running,
                       'note': f'{cd["broken_count"]} contacts broken'})

    if ds:
        d = round((ds['score'] - 0.5) * 30)
        running = max(0, min(100, running + d))
        steps.append({'label': f'Domain integrity ({ds["classification"]})', 'delta': d, 'running': running,
                       'note': ds.get('explanation', '')})

    if fr:
        d = -round(fr['overall_risk'] * 35)
        running = max(0, running + d)
        steps.append({'label': 'Functional residue risk', 'delta': d, 'running': running,
                       'note': '; '.join(fr.get('reasons', [])[:2])})

    if asm:
        d = -round(asm['assembly_risk'] * 30)
        running = max(0, running + d)
        steps.append({'label': f'Assembly context ({asm["assembly_class"]})', 'delta': d, 'running': running,
                       'note': asm['notes'][0] if asm['notes'] else ''})

    if f1_fold and f2_fold:
        weaker = min(f1_fold['overall'], f2_fold['overall'])
        d = round((weaker - 0.5) * 20)
        running = max(0, min(100, running + d))
        steps.append({'label': 'Fragment foldability (weaker)', 'delta': d, 'running': running,
                       'note': f'F1={round(f1_fold["overall"]*100)}% F2={round(f2_fold["overall"]*100)}%'})

    if geo:
        d = round((geo['score'] - 0.5) * 15)
        running = max(0, min(100, running + d))
        steps.append({'label': 'Reassembly geometry', 'delta': d, 'running': running,
                       'note': f'{geo.get("terminal_distance_a","?")}Å' if geo.get('terminal_distance_a') else 'no structure'})

    if fb:
        d = round((fb['overall_burden'] - 0.5) * 12)
        running = max(0, min(100, running + d))
        steps.append({'label': 'Fusion burden', 'delta': d, 'running': running,
                       'note': f'F1 {round(fb["f1_construct_mw"]/1000)}kDa F2 {round(fb["f2_construct_mw"]/1000)}kDa'})

    if mfg:
        d = round((mfg['feasibility'] - 50) / 10)
        running = max(0, min(100, running + d))
        steps.append({'label': f'Manufacturability (grade {mfg["grade"]})', 'delta': d, 'running': running,
                       'note': f'Limiting: {mfg["limiting_factor"]}'})

    return steps


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 13 — MONTE CARLO CUT RANKING
# ═══════════════════════════════════════════════════════════════════════════

def _rand_norm() -> float:
    """Box-Muller normal sample."""
    import math
    u = max(1e-10, random.random())
    v = random.random()
    return math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


def run_split_monte_carlo(scored_candidates: List[Dict], n: int = 300) -> Dict:
    """
    Stochastic rank stability test.
    CVs: SPELL ±15%, contact ±30%, functional ±25%, assembly ±25%, foldability ±20%.
    Mirrors runSplitMonteCarlo() in splitter_engine.js v10.0.
    """
    if not scored_candidates:
        return {'results': [], 'n': n, 'certainty': 'unknown'}

    K = len(scored_candidates)
    rank_counts   = [0] * K
    score_accum   = [[] for _ in range(K)]

    def sample_norm(center, cv):
        return max(0.0, min(100.0, center * (1 + cv * _rand_norm())))

    for _ in range(n):
        iter_scores = []
        for i, c in enumerate(scored_candidates):
            sc = c.get('subscores', {})
            spell  = (sc.get('site_spell', 0.5)) * 100
            cd_pen = (1 - sc.get('contact_integrity', 1.0)) * 40
            domain = (sc.get('domain_integrity', 0.5) - 0.5) * 30
            func   = sc.get('functional_safety', 1.0)
            asm    = sc.get('assembly_context', 0.9)
            fold   = sc.get('frag_foldability', 0.6)

            n_spell  = sample_norm(spell, 0.15)
            n_cd     = sample_norm(cd_pen, 0.30)
            n_func   = max(0, min(1, func * (1 + 0.25 * _rand_norm())))
            n_asm    = max(0, min(1, asm  * (1 + 0.25 * _rand_norm())))
            n_fold   = max(0, min(1, fold * (1 + 0.20 * _rand_norm())))
            n_fold_adj = (n_fold - 0.5) * 20

            noisy = max(0.0, min(100.0, n_spell - n_cd + domain - (1-n_func)*35 - (1-n_asm)*30 + n_fold_adj))
            iter_scores.append((i, noisy))

        iter_scores.sort(key=lambda x: -x[1])
        for rank, (i, s) in enumerate(iter_scores):
            if rank < 3:
                rank_counts[i] += 1
            score_accum[i].append(s)

    def percentile(arr, p):
        s = sorted(arr)
        idx = int(p / 100 * (len(s) - 1))
        return round(s[idx])

    results = []
    for i, c in enumerate(scored_candidates):
        arr = score_accum[i]
        results.append({
            'position'      : c.get('position'),
            'rank_pct3'     : round(rank_counts[i] / n * 100),
            'expected_score': round(sum(arr) / len(arr)) if arr else 0,
            'p10_score'     : percentile(arr, 10),
            'p90_score'     : percentile(arr, 90),
            'rank_stability': 'stable' if (percentile(arr, 90) - percentile(arr, 10)) < 25 else 'volatile',
        })

    results.sort(key=lambda x: -x['rank_pct3'])
    for i, r in enumerate(results):
        r['expected_rank'] = i + 1

    top = results[0] if results else {}
    certainty = ('dominant' if top.get('rank_pct3', 0) >= 50
                 else 'preferred' if top.get('rank_pct3', 0) >= 30
                 else 'marginal'  if top.get('rank_pct3', 0) >= 15
                 else 'toss-up')

    return {'results': results, 'n': n, 'top_candidate': top, 'certainty': certainty}


# ═══════════════════════════════════════════════════════════════════════════
# DELIVERY STRATEGY RANKER
# ═══════════════════════════════════════════════════════════════════════════

def rank_delivery_strategies(f1_mw: int, f2_mw: int,
                              target_tissue: str = 'cns') -> List[Dict]:
    results = []
    modes = [
        ('dual_aav',        'Dual AAV gene therapy',          'none'),
        ('mrna_lnp',        'mRNA / LNP co-delivery',         'none'),
        ('protein_biologic','Protein biologic + TfR-RMT',     'tfr_rmt'),
        ('protein_biologic','Protein biologic + RVG-29',      'rvg29'),
    ]
    for mode, label, strat in modes:
        d = _score_delivery_mode(f1_mw, f2_mw, mode, strat, target_tissue)
        results.append({
            'id'       : mode + (f'_{strat}' if strat != 'none' else ''),
            'label'    : label,
            'score'    : d['score'],
            'score_pct': d['score_pct'],
            'notes'    : d['notes'],
            'warnings' : d['warnings'],
            'color'    : d['color'],
        })
    results.sort(key=lambda x: -x['score'])
    for i, r in enumerate(results):
        r['rank'] = i + 1
    return results


def _score_delivery_mode(f1_mw, f2_mw, mode, bbb_strat, tissue):
    max_mw = max(f1_mw, f2_mw)
    notes, warnings = [], []
    score = 0.0
    if mode == 'dual_aav':
        payload_ok = max_mw <= 158000
        co_trans   = 0.55 if tissue == 'cns' else 0.65
        score      = co_trans if payload_ok else co_trans * (158000 / max_mw) * 0.5
        if not payload_ok:
            warnings.append(f'Fragment >{round(max_mw/1000)}kDa may exceed dual-AAV payload ceiling')
        notes.append('Co-transduction ~55% in CNS with AAV9/AAVrh10. Npu DnaE intein required.')
    elif mode == 'mrna_lnp':
        imbalance     = max(f1_mw, f2_mw) / max(1, min(f1_mw, f2_mw))
        stoich_pen    = 0.30 if imbalance > 2.5 else 0.10 if imbalance > 1.5 else 0
        score         = 0.70 * (1 - stoich_pen) * (0.75 if tissue == 'cns' else 1.0)
        if stoich_pen:
            warnings.append(f'MW imbalance {imbalance:.1f}x — stoichiometry risk')
        notes.append('LNP endosomal escape ~70%. CNS: 0.5-3% ID/g brain; IT administration improves.')
    else:
        # Protein biologic — simplified BBB score proxy
        lp1 = max(-5.0, min(2.0, -1.5 - f1_mw / 15000))
        lp2 = max(-5.0, min(2.0, -1.5 - f2_mw / 15000))
        mw_factor1 = math.exp(-0.0025 * max(0, f1_mw - 100))
        mw_factor2 = math.exp(-0.0025 * max(0, f2_mw - 100))
        lp_factor1 = math.exp(-0.5 * ((lp1 - 1.7) / 2.5) ** 2)
        lp_factor2 = math.exp(-0.5 * ((lp2 - 1.7) / 2.5) ** 2)
        f1_bbb = mw_factor1 * lp_factor1
        f2_bbb = mw_factor2 * lp_factor2
        if bbb_strat == 'tfr_rmt':
            f1_bbb = min(0.72, f1_bbb + 0.40)
            notes.append('TfR-RMT validated — pabinafusp alfa (Japan 2021 approval).')
        if bbb_strat in ('tfr_rmt', 'rvg29'):
            f2_bbb = min(0.45, f2_bbb + 0.25)
            notes.append('RVG-29 preferentially transduces RABV-infected neurons (nAChR targeting).')
        score = f1_bbb * f2_bbb
        notes.append(f'F1 BBB ~{round(f1_bbb*100)}% · F2 BBB ~{round(f2_bbb*100)}%')
        if max_mw > 50000 and bbb_strat == 'none':
            warnings.append('Large biologic with no BBB strategy: brain exposure <0.1% ID')

    score = max(0.0, min(1.0, score))
    return {
        'score'    : score,
        'score_pct': round(score * 100),
        'notes'    : notes,
        'warnings' : warnings,
        'color'    : '#3ecf8e' if score >= 0.55 else '#f59e0b' if score >= 0.30 else '#f87171',
    }


# ═══════════════════════════════════════════════════════════════════════════
# FULL ANALYSIS RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_full_analysis(
    sequence: Optional[str] = None,
    target_id: Optional[str] = None,
    pdb_text: Optional[str] = None,
    af_pdb_text: Optional[str] = None,
    split_system: str = 'split_intein_npu',
    conc_nm: float = 5.0,
    delivery_mode: str = 'protein_biologic',
    bbb_strategy: str = 'tfr_rmt',
    target_tissue: str = 'cns',
    top_n: int = 10,
    custom_annotations: Optional[Dict] = None,
    custom_domains: Optional[List[Dict]] = None,
    constraints: Optional[Dict] = None,
    run_mc: bool = True,
    mc_n: int = 300,
) -> Dict:
    """
    Master pipeline. Mirrors runFullAnalysis() in splitter_engine.js v10.0.
    """
    t0 = _time.perf_counter()

    # 1. Resolve sequence + annotations
    if target_id and target_id in RABV_TARGET_MAPS:
        preset      = RABV_TARGET_MAPS[target_id]
        seq         = preset.get('sequence') or sequence or ''
        annotations = {**preset['annotations'], **(custom_annotations or {})}
        domain_list = custom_domains or preset.get('domains', [])
        target_label = preset['name']
    else:
        fasta_r     = parse_fasta(sequence or '')
        seq         = fasta_r['sequence']
        annotations = custom_annotations or {}
        domain_list = custom_domains or []
        target_label = fasta_r.get('name', 'Custom')

    if not seq or len(seq) < 50:
        return {'error': 'Sequence too short or unavailable', 'sequence': seq, 'N': len(seq)}

    # 2. Parse structural data
    pdb_result  = parse_pdb(pdb_text)    if pdb_text    else None
    if pdb_result and pdb_result.error:  pdb_result = None
    af_result   = parse_alphafold(af_pdb_text) if af_pdb_text else None
    if af_result and af_result.get('error'): af_result = None
    contact_map = build_contact_map(pdb_result) if pdb_result else None

    # 3. Residue properties
    residue_props = estimate_residue_properties(seq)
    if pdb_result:
        residue_props = apply_structure_to_props(residue_props, pdb_result, af_result)

    # 4. Find candidates
    candidates = find_split_site_candidates(residue_props, annotations, constraints)

    # 5. Motif detection
    motif_hits = detect_functional_motifs(seq)

    # 6. Score all candidates
    scored = []
    for c in candidates:
        try:
            result = score_split_delivery_total(
                split_site   = c,
                sequence     = seq,
                split_system = split_system,
                conc_nm      = conc_nm,
                delivery_mode= delivery_mode,
                bbb_strategy = bbb_strategy,
                target_tissue= target_tissue,
                annotations  = annotations,
                domains      = domain_list,
                target_id    = target_id,
                pdb_result   = pdb_result,
                af_result    = af_result,
                contact_map  = contact_map,
                motif_hits   = motif_hits,
            )
            scored.append(result)
        except Exception as e:
            scored.append({
                'position': c.get('position'), 'final_score': 0,
                'is_rejected': True, 'rejection_reason': f'Scoring error: {e}', 'error': True,
            })

    # 7. Separate rejected
    passed   = [s for s in scored if not s.get('is_rejected') and not s.get('error')]
    rejected = [s for s in scored if s.get('is_rejected') or s.get('error')]
    passed.sort(key=lambda x: -x['final_score'])
    top_candidates = passed[:top_n]

    # 8. Monte Carlo
    mc_results = run_split_monte_carlo(top_candidates[:20], mc_n) if run_mc and top_candidates else {'results': [], 'n': mc_n, 'certainty': 'n/a'}

    # 9. Delivery ranking
    best_f1mw = top_candidates[0]['f1_mw'] if top_candidates else estimate_fragment_mw(seq[:len(seq)//2])
    best_f2mw = top_candidates[0]['f2_mw'] if top_candidates else estimate_fragment_mw(seq[len(seq)//2:])
    delivery_ranking = rank_delivery_strategies(best_f1mw, best_f2mw, target_tissue)

    # 10. Calibration check
    calibration = _calibrate(seq, annotations, residue_props)

    # 11. Summary
    best = top_candidates[0] if top_candidates else {}
    mc_top = mc_results.get('top_candidate') or {}
    summary = {
        'top_position'       : best.get('position'),
        'top_score'          : best.get('final_score'),
        'top_verdict'        : best.get('verdict'),
        'total_candidates'   : len(candidates),
        'passed_candidates'  : len(passed),
        'rejected_count'     : len(rejected),
        'has_structural_data': bool(pdb_result),
        'has_annotations'    : bool(annotations),
        'mc_top_position'    : mc_top.get('position'),
        'mc_certainty'       : mc_results.get('certainty'),
        'recommended_delivery': delivery_ranking[0]['label'] if delivery_ranking else None,
        'calibration_passed' : calibration.get('passed'),
        'motif_count'        : len(motif_hits),
        'rejected_examples'  : [f'pos {r.get("position","?")}: {r.get("rejection_reason","rejected")}' for r in rejected[:3]],
        'elapsed_ms'         : round((_time.perf_counter() - t0) * 1000, 1),
    }

    return {
        'sequence'           : seq,
        'N'                  : len(seq),
        'target'             : target_label,
        'residue_props'      : [{
            'position': p['position'], 'aa': p['aa'], 'ss': p['ss'],
            'accessibility': p['accessibility'], 'disorder': p['disorder'],
            'plddt': p.get('plddt'), 'has_structure': p.get('has_structure', False),
        } for p in residue_props],
        'candidates'         : passed,
        'rejected_candidates': rejected,
        'top_candidates'     : top_candidates,
        'mc_results'         : mc_results,
        'delivery_ranking'   : delivery_ranking,
        'calibration'        : calibration,
        'summary'            : summary,
        'motif_hits'         : motif_hits,
        'has_structure'      : bool(pdb_result),
        'has_annotations'    : bool(annotations),
        'pdb_parsed'         : {'chains': pdb_result.chains, 'n_residues': len(pdb_result.residues)} if pdb_result else None,
        'contact_map_stats'  : {'total_contacts': len(contact_map['contacts']), 'N': contact_map['N']} if contact_map else None,
    }


def _calibrate(sequence: str, annotations: Dict, residue_props: List[Dict]) -> Dict:
    """Quick self-test: known-good positions should rank above median."""
    good_pos = annotations.get('known_good_splits', [])
    if not good_pos:
        return {'calibration_score': None, 'details': [], 'passed': None,
                'note': 'No known_good_splits in annotations'}

    candidates = find_split_site_candidates(residue_props, annotations, {})
    if not candidates:
        return {'calibration_score': None, 'details': [], 'passed': None,
                'note': 'No candidates found for calibration'}

    scores = sorted(c['split_score'] for c in candidates)
    median = scores[len(scores) // 2]
    n_above = 0
    details = []
    for pos in good_pos:
        match = next((c for c in candidates if abs(c['position'] - pos) <= 5), None)
        score = match['split_score'] if match else None
        above = score is not None and score > median
        if above: n_above += 1
        details.append({'position': pos, 'score': score, 'above_median': above,
                         'verdict': 'pass' if above else 'fail'})

    cal_score = n_above / len(good_pos)
    return {
        'calibration_score': round(cal_score, 2),
        'details'          : details,
        'passed'           : cal_score >= 0.6,
        'note'             : f'{round(cal_score*100)}% of known-good splits ranked above median',
    }
