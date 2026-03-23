"""
Fetch antiperovskite (M₃AX) candidates from the Materials Project.

Antiperovskite structure (Pm-3m, #221):
  - M site : transition metal  (3 atoms, face-center positions)
  - A site : main-group element (1 atom, corner)
  - X site : light anion        (1 atom, body-center)  N or C

Features computed from composition only (no DFT needed upfront):
  m_eneg        : Pauling electronegativity of M element
  a_eneg        : Pauling electronegativity of A element
  x_eneg        : Pauling electronegativity of X element
  m_radius      : atomic radius of M element (Å)
  a_radius      : atomic radius of A element (Å)
  x_radius      : atomic radius of X element (Å)
  m_max_oxid    : max common oxidation state of M
  a_max_oxid    : max common oxidation state of A
  tolerance_factor : t = (r_A + r_X) / (√2 · (r_M + r_X))

Objective (filled lazily by the NIMO loop):
  formation_energy_per_atom

Usage:
    export MP_API_KEY="your_key_here"
    python mp_fetch_antiperovskite.py
"""

import os
import csv
import sys
import math
import numpy as np
from itertools import product
from pymatgen.core import Element, Composition
from mp_api.client import MPRester


MP_API_KEY = os.environ.get("MP_API_KEY")
if not MP_API_KEY:
    print("Error: set MP_API_KEY environment variable first.")
    sys.exit(1)

OUTPUT_FILE = "antiperovskite_candidates.csv"

# ── Element pools ─────────────────────────────────────────────────────────────
M_ELEMENTS = ["Mn", "Fe", "Co", "Ni", "Cr"]           # transition metals
A_ELEMENTS = ["Ga", "Ge", "Sn", "Al", "Zn", "In", "Si", "Cu", "Pb"]
X_ELEMENTS = ["N", "C"]                                # nitrides and carbides


def element_features(symbol):
    """Return (eneg, atomic_radius_ang, max_common_oxidation_state)."""
    el = Element(symbol)
    eneg   = float(el.X) if el.X is not None else float("nan")
    radius = float(el.atomic_radius) if el.atomic_radius is not None else float("nan")
    oxids  = el.common_oxidation_states
    max_ox = float(max(oxids)) if oxids else float("nan")
    return eneg, radius, max_ox


def tolerance_factor(r_m, r_a, r_x):
    """
    Goldschmidt-style tolerance factor for M₃AX antiperovskite.
    t = (r_A + r_X) / (√2 · (r_M + r_X))
    Stable cubic phase: 0.71 ≤ t ≤ 1.05
    """
    denom = math.sqrt(2) * (r_m + r_x)
    return (r_a + r_x) / denom if denom > 0 else float("nan")


def is_3_1_1(formula_str, m, a, x):
    """
    Return True if formula has exactly stoichiometry M₃AX:
    3 atoms of M, 1 of A, 1 of X (or integer multiples thereof).
    """
    try:
        comp = Composition(formula_str)
        amounts = {str(el): comp[el] for el in comp}
        if set(amounts.keys()) != {m, a, x}:
            return False
        vals = sorted(amounts.values())   # [1, 1, 3] or multiples
        if len(vals) != 3:
            return False
        # normalise to smallest
        g = math.gcd(int(vals[0]), math.gcd(int(vals[1]), int(vals[2])))
        normed = sorted([int(v) // g for v in vals])
        return normed == [1, 1, 3]
    except Exception:
        return False


def identify_sites(formula_str, m, a, x):
    """
    Given a valid M₃AX formula, return which element is the M (×3) site.
    Returns (m_elem, a_elem, x_elem).
    """
    comp = Composition(formula_str)
    amounts = {str(el): comp[el] for el in comp}
    m_elem = max(amounts, key=lambda k: amounts[k])   # element with count 3
    rest   = [k for k in amounts if k != m_elem]
    # A is the main-group element from our A pool; X is N or C
    a_elem = next((r for r in rest if r == a), rest[0])
    x_elem = next((r for r in rest if r == x), rest[1])
    return m_elem, a_elem, x_elem


# ── Fetch ─────────────────────────────────────────────────────────────────────

print("Fetching antiperovskite (M₃AX) candidates from Materials Project…\n")

rows = []
seen_ids = set()

combos = list(product(M_ELEMENTS, A_ELEMENTS, X_ELEMENTS))
print(f"Querying {len(combos)} M-A-X chemical systems…")

with MPRester(MP_API_KEY) as mpr:
    for m, a, x in combos:
        chemsys = "-".join(sorted([m, a, x]))
        try:
            docs = mpr.materials.summary.search(
                chemsys=chemsys,
                num_elements=3,
                fields=["material_id", "formula_pretty",
                        "formation_energy_per_atom", "energy_above_hull",
                        "is_stable"]
            )
        except Exception as e:
            print(f"  Warning {chemsys}: {e}")
            continue

        hits = 0
        for doc in docs:
            mid = doc.material_id
            if mid in seen_ids:
                continue
            formula = doc.formula_pretty
            if not is_3_1_1(formula, m, a, x):
                continue

            seen_ids.add(mid)
            m_elem, a_elem, x_elem = identify_sites(formula, m, a, x)

            m_eneg, m_rad, m_ox = element_features(m_elem)
            a_eneg, a_rad, a_ox = element_features(a_elem)
            x_eneg, x_rad, _    = element_features(x_elem)
            t_factor = tolerance_factor(m_rad, a_rad, x_rad)

            rows.append({
                "material_id":             mid,
                "formula":                 formula,
                "M":                       m_elem,
                "A":                       a_elem,
                "X":                       x_elem,
                "m_eneg":                  round(m_eneg,   4),
                "a_eneg":                  round(a_eneg,   4),
                "x_eneg":                  round(x_eneg,   4),
                "m_radius":                round(m_rad,    4),
                "a_radius":                round(a_rad,    4),
                "x_radius":                round(x_rad,    4),
                "m_max_oxid":              round(m_ox,     4),
                "a_max_oxid":              round(a_ox,     4),
                "tolerance_factor":        round(t_factor, 4),
                "energy_above_hull":       doc.energy_above_hull
                                           if doc.energy_above_hull is not None else "",
                "formation_energy_per_atom": ""   # filled lazily by NIMO loop
            })
            hits += 1

        if hits:
            print(f"  {chemsys}: {hits} antiperovskite(s) found")


print(f"\nTotal antiperovskite candidates: {len(rows)}")
if not rows:
    print("No candidates found. Try broadening M/A/X element pools.")
    sys.exit(1)

# ── Write CSV ─────────────────────────────────────────────────────────────────

fieldnames = [
    "material_id", "formula", "M", "A", "X",
    "m_eneg", "a_eneg", "x_eneg",
    "m_radius", "a_radius", "x_radius",
    "m_max_oxid", "a_max_oxid", "tolerance_factor",
    "energy_above_hull", "formation_energy_per_atom"
]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Written to {OUTPUT_FILE}")

# ── Preview ───────────────────────────────────────────────────────────────────
print("\nPreview (first 5 rows):")
with open(OUTPUT_FILE) as f:
    for i, line in enumerate(f):
        print(" ", line.rstrip())
        if i >= 5:
            break

# ── Distribution summary ──────────────────────────────────────────────────────
from collections import Counter
x_counts = Counter(r["X"] for r in rows)
m_counts = Counter(r["M"] for r in rows)
print(f"\nBy X-site : {dict(x_counts)}")
print(f"By M-site : {dict(m_counts)}")
