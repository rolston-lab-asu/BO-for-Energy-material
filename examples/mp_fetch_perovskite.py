"""
Fetch halide perovskite (ABX₃) candidates from the Materials Project.

Perovskite structure (Pm-3m, #221):
  A site : large monovalent cation  (corner,       ×1)
  B site : small divalent metal     (body-center,  ×1)
  X site : halide anion             (face-center,  ×3)

Features computed from composition only (no DFT needed upfront):
  a_eneg, b_eneg, x_eneg      : Pauling electronegativity per site
  a_radius, b_radius, x_radius : Shannon ionic radius (Å) by site
                                 A: CN=XII (+1), B: CN=VI (+2), X: CN=VI (−1)
                                 Falls back to atomic radius if Shannon data absent.
  a_max_oxid, b_max_oxid       : max common oxidation state
  tolerance_factor             : t = (r_A + r_X) / (√2 · (r_B + r_X))
                                 cubic stable range: 0.80 ≤ t ≤ 1.06
  octahedral_factor            : μ = r_B / r_X
                                 stable BX₆ octahedra: 0.41 ≤ μ ≤ 0.73

Structure-based features (from MP, distinguish polymorphs):
  spacegroup_number            : international spacegroup number (1–230)
  crystal_system               : integer encoding of crystal system
                                 (triclinic=1, monoclinic=2, orthorhombic=3,
                                  tetragonal=4, trigonal=5, hexagonal=6, cubic=7)
  volume_per_atom              : unit cell volume / nsites (Å³/atom)
  density                      : mass density (g/cm³)

Both objectives left blank for lazy evaluation by mp_nimo_pv.py:
  formation_energy_per_atom
  band_gap

Usage:
    export MP_API_KEY="your_key_here"
    python mp_fetch_perovskite.py [--config perovskite_config.yaml]

Then run the NIMO loop:
    python mp_nimo_pv.py --config perovskite_config.yaml
"""

import os
import csv
import sys
import math
import time
import argparse
from collections import Counter
from itertools import product

import yaml
from pymatgen.core import Element, Composition, Species
from mp_api.client import MPRester


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Shannon ionic radii ───────────────────────────────────────────────────────

# Manual fallbacks for species absent from pymatgen's Shannon table.
# Source: Shannon (1976) Acta Crystallogr. A32, 751–767.
_SHANNON_FALLBACK = {
    ("Sn", +2, "VI"): 1.18,   # Sn²⁺ not in pymatgen; pymatgen only has Sn⁴⁺
}

# Roman numeral strings for coordination numbers
_CN_ROMAN = {6: "VI", 12: "XII"}


def shannon_radius(symbol, oxidation_state, cn):
    """
    Return Shannon ionic radius (Å) for (symbol, oxidation_state, cn).

    cn must be an integer (6 or 12).  Falls back to hardcoded table, then
    to pymatgen atomic radius if Shannon data is unavailable.
    """
    key = (symbol, oxidation_state, _CN_ROMAN[cn])
    if key in _SHANNON_FALLBACK:
        return _SHANNON_FALLBACK[key]
    try:
        sp = Species(symbol, oxidation_state)
        return float(sp.get_shannon_radius(_CN_ROMAN[cn]))
    except Exception:
        el = Element(symbol)
        return float(el.atomic_radius) if el.atomic_radius is not None else float("nan")


# ── Element feature helpers ───────────────────────────────────────────────────

def element_features(symbol):
    """Return (eneg, atomic_radius_ang, max_common_oxidation_state)."""
    el     = Element(symbol)
    eneg   = float(el.X) if el.X is not None else float("nan")
    radius = float(el.atomic_radius) if el.atomic_radius is not None else float("nan")
    oxids  = el.common_oxidation_states
    max_ox = float(max(oxids)) if oxids else float("nan")
    return eneg, radius, max_ox


def goldschmidt_tolerance(r_a, r_b, r_x):
    """t = (r_A + r_X) / (√2 · (r_B + r_X))  — cubic stable: 0.80–1.06."""
    denom = math.sqrt(2) * (r_b + r_x)
    return (r_a + r_x) / denom if denom > 0 else float("nan")


def octahedral_factor(r_b, r_x):
    """μ = r_B / r_X  — stable BX₆ octahedra: 0.41–0.73."""
    return r_b / r_x if r_x > 0 else float("nan")


# ── Stoichiometry helpers ─────────────────────────────────────────────────────

def is_1_1_3(formula_str, a, b, x):
    """True if formula matches ABX₃ stoichiometry (or integer multiples)."""
    try:
        comp    = Composition(formula_str)
        amounts = {str(el): comp[el] for el in comp}
        if set(amounts.keys()) != {a, b, x}:
            return False
        vals = sorted(amounts.values())
        if len(vals) != 3:
            return False
        # Pattern: [k, k, 3k]  →  normed = [1, 1, 3]
        g      = math.gcd(int(vals[0]), math.gcd(int(vals[1]), int(vals[2])))
        normed = sorted([int(v) // g for v in vals])
        return normed == [1, 1, 3]
    except Exception:
        return False


def identify_sites(formula_str, a_pool, b_pool):
    """
    Return (a_elem, b_elem, x_elem) from a validated ABX₃ formula.
    X is the element with the highest count (×3).
    A and B are resolved by matching against the element pools.
    """
    comp    = Composition(formula_str)
    amounts = {str(el): comp[el] for el in comp}
    x_elem  = max(amounts, key=amounts.get)            # highest count = X site
    rest    = [k for k in amounts if k != x_elem]
    a_elem  = next((r for r in rest if r in a_pool), rest[0])
    b_elem  = next((r for r in rest if r in b_pool), rest[1] if len(rest) > 1 else rest[0])
    return a_elem, b_elem, x_elem


# ── MP query with retry ───────────────────────────────────────────────────────

# Integer encoding for crystal system — stable across pymatgen versions
_CRYSTAL_SYSTEM_INT = {
    "triclinic": 1, "monoclinic": 2, "orthorhombic": 3,
    "tetragonal": 4, "trigonal": 5, "hexagonal": 6, "cubic": 7,
}


def query_chemsys(mpr, chemsys, max_retries, base_delay):
    """Search a 3-element chemical system with exponential-backoff retry."""
    for attempt in range(max_retries):
        try:
            return mpr.materials.summary.search(
                chemsys=chemsys,
                num_elements=3,
                fields=["material_id", "formula_pretty",
                        "energy_above_hull", "is_stable",
                        "symmetry", "volume", "nsites", "density"]
            )
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Warning {chemsys}: failed after {max_retries} attempts: {e}")
                return []
            wait = base_delay * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{max_retries} for {chemsys} in {wait:.0f}s: {e}")
            time.sleep(wait)
    return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path):
    MP_API_KEY = os.environ.get("MP_API_KEY")
    if not MP_API_KEY:
        print("Error: set MP_API_KEY environment variable first.")
        sys.exit(1)

    cfg            = load_config(config_path)
    output_file    = cfg["files"]["candidates"]
    hull_threshold = cfg["hull_threshold"]
    A_ELEMENTS     = cfg["elements"]["A"]
    B_ELEMENTS     = cfg["elements"]["B"]
    X_ELEMENTS     = cfg["elements"]["X"]
    max_retries    = cfg["api"]["max_retries"]
    base_delay     = cfg["api"]["retry_base_delay"]
    inter_delay    = cfg["api"]["inter_call_delay"]

    combos = list(product(A_ELEMENTS, B_ELEMENTS, X_ELEMENTS))
    print(f"Fetching halide perovskite (ABX₃) candidates from Materials Project…")
    print(f"Hull threshold : ≤ {hull_threshold} eV/atom")
    print(f"Querying {len(combos)} A-B-X chemical systems…\n")

    rows     = []
    seen_ids = set()

    with MPRester(MP_API_KEY) as mpr:
        for a, b, x in combos:
            chemsys = "-".join(sorted([a, b, x]))
            docs    = query_chemsys(mpr, chemsys, max_retries, base_delay)
            time.sleep(inter_delay)

            hits = 0
            for doc in docs:
                mid     = doc.material_id
                if mid in seen_ids:
                    continue

                formula = doc.formula_pretty
                if not is_1_1_3(formula, a, b, x):
                    continue

                hull = doc.energy_above_hull
                if hull is None or hull > hull_threshold:
                    continue

                seen_ids.add(mid)
                a_elem, b_elem, x_elem = identify_sites(formula, A_ELEMENTS, B_ELEMENTS)

                a_eneg, _, a_ox = element_features(a_elem)
                b_eneg, _, b_ox = element_features(b_elem)
                x_eneg, _, _    = element_features(x_elem)

                # Shannon ionic radii — physically correct for Goldschmidt/octahedral factors
                # A: +1 cation, CN=XII (cuboctahedral corner site)
                # B: +2 cation, CN=VI  (octahedral body-centre site)
                # X: −1 halide, CN=VI  (face-centre site)
                a_rad = shannon_radius(a_elem, +1, 12)
                b_rad = shannon_radius(b_elem, +2,  6)
                x_rad = shannon_radius(x_elem, -1,  6)

                t_factor = goldschmidt_tolerance(a_rad, b_rad, x_rad)
                oct_fac  = octahedral_factor(b_rad, x_rad)

                # Structure-based features — distinguish polymorphs
                sym          = doc.symmetry
                sg_num       = int(sym.number) if sym and sym.number else 0
                cs_str       = sym.crystal_system.value.lower() if sym and sym.crystal_system else ""
                cs_int       = _CRYSTAL_SYSTEM_INT.get(cs_str, 0)
                nsites       = int(doc.nsites) if doc.nsites else 0
                vol          = float(doc.volume) if doc.volume else float("nan")
                vol_per_atom = round(vol / nsites, 4) if nsites > 0 else float("nan")
                density      = round(float(doc.density), 4) if doc.density else float("nan")

                rows.append({
                    "material_id":               mid,
                    "formula":                   formula,
                    "A":                         a_elem,
                    "B":                         b_elem,
                    "X":                         x_elem,
                    "a_eneg":                    round(a_eneg,   4),
                    "b_eneg":                    round(b_eneg,   4),
                    "x_eneg":                    round(x_eneg,   4),
                    "a_radius":                  round(a_rad,    4),
                    "b_radius":                  round(b_rad,    4),
                    "x_radius":                  round(x_rad,    4),
                    "a_max_oxid":                round(a_ox,     4),
                    "b_max_oxid":                round(b_ox,     4),
                    "tolerance_factor":          round(t_factor, 4),
                    "octahedral_factor":         round(oct_fac,  4),
                    "spacegroup_number":         sg_num,
                    "crystal_system":            cs_int,
                    "volume_per_atom":           vol_per_atom,
                    "density":                   density,
                    "energy_above_hull":         hull,
                    # Both objectives left blank — oracle fills lazily.
                    # band_gap_raw : raw PBE value from MP (for traceability)
                    # band_gap     : scissor-corrected value (written by NIMO loop)
                    "formation_energy_per_atom": "",
                    "band_gap_raw":              "",
                    "band_gap":                  "",
                })
                hits += 1

            if hits:
                print(f"  {chemsys}: {hits} ABX₃ perovskite(s) found")

    # ── Deduplicate: keep ground-state polymorph per composition ─────────────
    # Multiple MP entries for the same formula (e.g. cubic/orthorhombic CsSnI₃)
    # represent polymorphs of the same compound.  A real experiment produces one
    # material; keeping all entries conflates "which composition?" with "which
    # polymorph?".  We retain only the entry with the lowest energy_above_hull
    # (ground state); ties broken by material_id for determinism.
    best: dict[str, dict] = {}
    for row in rows:
        formula = row["formula"]
        hull    = float(row["energy_above_hull"])
        if formula not in best or hull < float(best[formula]["energy_above_hull"]):
            best[formula] = row
    before = len(rows)
    rows   = sorted(best.values(), key=lambda r: r["formula"])
    print(f"Deduplicated {before} entries → {len(rows)} unique compositions "
          f"(kept lowest-hull polymorph per formula)")

    print(f"\nTotal candidates (hull ≤ {hull_threshold} eV/atom): {len(rows)}")
    if not rows:
        print(f"No candidates found. Try raising hull_threshold in {config_path}.")
        sys.exit(1)

    fieldnames = [
        "material_id", "formula", "A", "B", "X",
        "a_eneg", "b_eneg", "x_eneg",
        "a_radius", "b_radius", "x_radius",
        "a_max_oxid", "b_max_oxid",
        "tolerance_factor", "octahedral_factor",
        "spacegroup_number", "crystal_system", "volume_per_atom", "density",
        "energy_above_hull",
        "formation_energy_per_atom", "band_gap_raw", "band_gap",
    ]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written to {output_file}")

    print("\nPreview (first 5 rows):")
    with open(output_file) as f:
        for i, line in enumerate(f):
            print(" ", line.rstrip())
            if i >= 5:
                break

    # Summary stats
    x_counts   = Counter(r["X"] for r in rows)
    b_counts   = Counter(r["B"] for r in rows)
    on_hull    = sum(1 for r in rows if float(r["energy_above_hull"]) == 0.0)
    t_vals     = [r["tolerance_factor"] for r in rows if r["tolerance_factor"] != "nan"]
    cubic_ok   = sum(1 for r in rows
                     if r["tolerance_factor"] not in ("nan", "")
                     and 0.80 <= float(r["tolerance_factor"]) <= 1.06)
    oct_ok     = sum(1 for r in rows
                     if r["octahedral_factor"] not in ("nan", "")
                     and 0.41 <= float(r["octahedral_factor"]) <= 0.73)

    print(f"\nBy X-site              : {dict(x_counts)}")
    print(f"By B-site              : {dict(b_counts)}")
    print(f"On hull (0 eV)         : {on_hull}")
    print(f"Near hull (>0)         : {len(rows) - on_hull}")
    print(f"Cubic t in [0.80,1.06] : {cubic_ok}/{len(rows)}")
    print(f"Stable oct μ in [0.41,0.73]: {oct_ok}/{len(rows)}")
    print(f"\nNext step:")
    print(f"  python mp_nimo_pv.py --config {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="perovskite_config.yaml")
    args = parser.parse_args()
    main(args.config)
