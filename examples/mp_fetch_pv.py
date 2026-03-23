"""
Fetch binary chalcogenide (M-X, X = S/Se/Te) photovoltaic candidates
from the Materials Project.

Features computed from composition only (no DFT needed upfront):
  cation_eneg    : Pauling electronegativity of the cation
  anion_eneg     : Pauling electronegativity of the anion
  eneg_diff      : |cation_eneg - anion_eneg|  (ionicity proxy)
  cation_radius  : atomic radius of the cation (Å)
  anion_radius   : atomic radius of the anion (Å)
  cation_fraction: cation / (cation + anion) atomic fraction
  cation_max_oxid: max common oxidation state of the cation

Both objectives are left blank for lazy evaluation by the NIMO loop:
  formation_energy_per_atom   — thermodynamic stability
  band_gap                    — electronic structure (eV)

The NIMO loop converts band_gap → |band_gap - bg_target| at runtime.

Usage:
    export MP_API_KEY="your_key_here"
    python mp_fetch_pv.py [--config pv_config.yaml]
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
from pymatgen.core import Element, Composition
from mp_api.client import MPRester


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Element and formula helpers ───────────────────────────────────────────────

def element_features(symbol):
    """Return (eneg, atomic_radius_ang, max_common_oxidation_state)."""
    el     = Element(symbol)
    eneg   = float(el.X) if el.X is not None else float("nan")
    radius = float(el.atomic_radius) if el.atomic_radius is not None else float("nan")
    oxids  = el.common_oxidation_states
    max_ox = float(max(oxids)) if oxids else float("nan")
    return eneg, radius, max_ox


def is_binary(formula_str, cation, anion):
    """True if formula contains exactly {cation, anion} elements."""
    try:
        comp = Composition(formula_str)
        return set(str(el) for el in comp) == {cation, anion}
    except Exception:
        return False


def cation_fraction(formula_str, cation):
    """Cation / (cation + anion) atomic fraction."""
    try:
        comp   = Composition(formula_str)
        c_amt  = comp[Element(cation)]
        total  = sum(comp[el] for el in comp)
        return c_amt / total if total > 0 else float("nan")
    except Exception:
        return float("nan")


# ── MP query with retry ───────────────────────────────────────────────────────

def query_chemsys(mpr, chemsys, max_retries, base_delay):
    """Search a chemical system with exponential-backoff retry."""
    for attempt in range(max_retries):
        try:
            return mpr.materials.summary.search(
                chemsys=chemsys,
                num_elements=2,
                fields=["material_id", "formula_pretty",
                        "energy_above_hull", "is_stable"]
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
    CATIONS        = cfg["elements"]["cations"]
    ANIONS         = cfg["elements"]["anions"]
    max_retries    = cfg["api"]["max_retries"]
    base_delay     = cfg["api"]["retry_base_delay"]
    inter_delay    = cfg["api"]["inter_call_delay"]

    combos = list(product(CATIONS, ANIONS))
    print(f"Fetching binary chalcogenide PV candidates from Materials Project…")
    print(f"Hull threshold : ≤ {hull_threshold} eV/atom")
    print(f"Querying {len(combos)} cation-anion chemical systems…\n")

    rows     = []
    seen_ids = set()

    with MPRester(MP_API_KEY) as mpr:
        for cation, anion in combos:
            chemsys = "-".join(sorted([cation, anion]))
            docs    = query_chemsys(mpr, chemsys, max_retries, base_delay)
            time.sleep(inter_delay)

            c_eneg, c_radius, c_max_ox = element_features(cation)
            a_eneg, a_radius, _        = element_features(anion)
            eneg_diff = abs(c_eneg - a_eneg)

            hits = 0
            for doc in docs:
                mid     = doc.material_id
                if mid in seen_ids:
                    continue

                hull = doc.energy_above_hull
                if hull is None or hull > hull_threshold:
                    continue

                formula = doc.formula_pretty
                if not is_binary(formula, cation, anion):
                    continue

                xf = cation_fraction(formula, cation)
                if math.isnan(xf):
                    continue

                seen_ids.add(mid)
                rows.append({
                    "material_id":               mid,
                    "formula":                   formula,
                    "cation":                    cation,
                    "anion":                     anion,
                    "cation_eneg":               round(c_eneg,    4),
                    "anion_eneg":                round(a_eneg,    4),
                    "eneg_diff":                 round(eneg_diff, 4),
                    "cation_radius":             round(c_radius,  4),
                    "anion_radius":              round(a_radius,  4),
                    "cation_fraction":           round(xf,        4),
                    "cation_max_oxid":           round(c_max_ox,  4),
                    "energy_above_hull":         hull,
                    # Both objectives left blank — oracle fills lazily
                    "formation_energy_per_atom": "",
                    "band_gap":                  "",
                })
                hits += 1

            if hits:
                print(f"  {chemsys}: {hits} candidate(s)")

    print(f"\nTotal candidates (hull ≤ {hull_threshold} eV/atom): {len(rows)}")
    if not rows:
        print(f"No candidates found. Try raising hull_threshold in {config_path}.")
        sys.exit(1)

    fieldnames = [
        "material_id", "formula", "cation", "anion",
        "cation_eneg", "anion_eneg", "eneg_diff",
        "cation_radius", "anion_radius", "cation_fraction", "cation_max_oxid",
        "energy_above_hull",
        "formation_energy_per_atom", "band_gap",
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

    anion_counts  = Counter(r["anion"]  for r in rows)
    cation_counts = Counter(r["cation"] for r in rows)
    on_hull       = sum(1 for r in rows if float(r["energy_above_hull"]) == 0.0)
    print(f"\nBy anion       : {dict(anion_counts)}")
    print(f"By cation      : {dict(cation_counts)}")
    print(f"On hull (0 eV) : {on_hull}")
    print(f"Near hull (>0) : {len(rows) - on_hull}")
    print(f"\nNext step: run mp_nimo_pv.py to optimise over these {len(rows)} candidates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pv_config.yaml")
    args = parser.parse_args()
    main(args.config)
