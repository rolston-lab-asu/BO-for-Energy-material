"""

Fetch binary oxide candidates from the Materials Project and write candidates.csv.

Features used (composition-based, no DFT needed upfront):
  - x             : fraction of element A in A_x O_y  (e.g. x in Fe_x O)
  - mean_eneg     : mean Pauling electronegativity of non-oxygen elements
  - mean_radius   : mean atomic radius of non-oxygen elements (angstrom)
  - max_oxid      : maximum common oxidation state of non-oxygen elements

Objective (queried lazily during the NIMO loop, not fetched here):
  - formation_energy_per_atom  (left as NaN → NIMO will fill it in)

Usage:
    export MP_API_KEY="your_key_here"
    python mp_fetch_candidates.py

Get a free API key at: https://next.materialsproject.org/api
"""

import os
import csv
import sys
import numpy as np
from mp_api.client import MPRester
from pymatgen.core import Element


MP_API_KEY = os.environ.get("MP_API_KEY")
if not MP_API_KEY:
    print("Error: set MP_API_KEY environment variable first.")
    sys.exit(1)

OUTPUT_FILE = "mp_candidates.csv"

# Elements to pair with oxygen (common binary oxide-forming metals)
TARGET_ELEMENTS = [
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "Zr", "Nb", "Mo", "Sn", "W", "Al", "Mg", "Ca"
]


def get_element_features(symbol):
    """Return (electronegativity, atomic_radius_angstrom, max_common_oxidation_state)."""
    el = Element(symbol)
    eneg = el.X if el.X is not None else float("nan")
    radius = el.atomic_radius if el.atomic_radius is not None else float("nan")
    # atomic_radius is in angstrom already in pymatgen
    oxid_states = el.common_oxidation_states
    max_oxid = max(oxid_states) if oxid_states else float("nan")
    return eneg, float(radius), float(max_oxid)


def formula_to_x_fraction(formula, metal_symbol):
    """
    Given a formula like Fe2O3, return x/(x+y) where Fe_x O_y.
    Returns the fraction of the metal in the formula.
    """
    from pymatgen.core import Composition
    try:
        comp = Composition(formula)
        metal_amt = comp[Element(metal_symbol)]
        o_amt = comp[Element("O")]
        total = metal_amt + o_amt
        return metal_amt / total if total > 0 else float("nan")
    except Exception:
        return float("nan")


print("Fetching binary oxide data from Materials Project...")

rows = []

with MPRester(MP_API_KEY) as mpr:
    for metal in TARGET_ELEMENTS:
        try:
            docs = mpr.materials.summary.search(
                chemsys=f"{metal}-O",
                fields=["material_id", "formula_pretty", "formation_energy_per_atom",
                        "energy_above_hull", "is_stable"]
            )
        except Exception as e:
            print(f"  Warning: could not fetch {metal}-O: {e}")
            continue

        eneg, radius, max_oxid = get_element_features(metal)

        for doc in docs:
            formula = doc.formula_pretty
            x_frac = formula_to_x_fraction(formula, metal)

            if np.isnan(x_frac):
                continue

            rows.append({
                "material_id": doc.material_id,
                "formula": formula,
                "metal": metal,
                "x_fraction": round(x_frac, 4),
                "mean_eneg": round(eneg, 4),
                "mean_radius": round(radius, 4),
                "max_oxid": round(max_oxid, 4),
                # objective — left blank so NIMO treats them as unmeasured
                "formation_energy_per_atom": ""
            })

        print(f"  {metal}-O: {len(docs)} entries")


print(f"\nTotal candidates: {len(rows)}")

if len(rows) == 0:
    print("No data fetched. Check your API key and network connection.")
    sys.exit(1)

# Deduplicate by material_id
seen = set()
unique_rows = []
for r in rows:
    if r["material_id"] not in seen:
        seen.add(r["material_id"])
        unique_rows.append(r)

print(f"Unique candidates: {len(unique_rows)}")

# Write candidates.csv (NIMO format: numeric features + objective columns last)
fieldnames = ["material_id", "formula", "metal",
              "x_fraction", "mean_eneg", "mean_radius", "max_oxid",
              "formation_energy_per_atom"]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(unique_rows)

print(f"Written to {OUTPUT_FILE}")
print("\nPreview (first 5 rows):")
with open(OUTPUT_FILE) as f:
    for i, line in enumerate(f):
        print(" ", line.rstrip())
        if i >= 5:
            break
