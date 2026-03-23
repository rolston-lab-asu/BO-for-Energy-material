"""
Fetch Li-intercalation cathode candidates from the Materials Project.

Queries the insertion_electrodes endpoint (not summary) with a single
batch request, then filters and computes features.

Features (all available before measuring objectives):
  tm_eneg                : composition-weighted Pauling electronegativity of TM(s)
  tm_ionic_radius        : composition-weighted atomic radius of TM(s) (Å)
  tm_max_oxidation_state : composition-weighted max common oxidation state
  anion_eneg             : Pauling eneg of highest-eneg non-TM element (O/S/F)
  li_per_fu              : Li atoms per formula unit in discharge structure
  spacegroup_number      : SG number of Li-free framework structure (1–230)
  crystal_system         : integer encoding (cubic=7, hex=6, …, triclinic=1)
  volume_per_atom        : framework volume / nsites  (Å³/atom)
  density                : framework mass density  (g/cm³)

  NEW — physics-motivated features:
  polyanion_eneg         : Pauling eneg of polyanion central atom (P/S/Si/As/B/Mo/W/V/…)
                           encodes Goodenough inductive effect on TM redox potential
                           0.0 for simple oxides/fluorides (no polyanion)
  structure_prototype    : integer encoding of framework topology
                           olivine(SG 62)=1, spinel(SG 227)=2, layered(SG 166)=3,
                           monoclinic-layered(SG 12)=4, NASICON(SG 167)=5, other=0
                           directly predicts ΔV and voltage plateau shape
  tm_oxidation_discharge : average oxidation state of primary TM in discharged structure
                           pins the actual redox couple (Fe²⁺↔Fe³⁺ vs Fe³⁺↔Fe⁴⁺)
                           fetched from MP oxidation_states field

Objectives left blank for lazy oracle evaluation by mp_nimo_battery.py:
  average_voltage   — V vs Li/Li⁺ (DFT-computed, no correction needed)
  max_delta_volume  — fractional volume change on lithiation

Deduplication: one entry per framework_formula, keeping highest average_voltage.
This removes thermo_type duplicates and multiple voltage-step sub-ranges.

Usage:
    export MP_API_KEY="your_key_here"
    python mp_fetch_battery.py [--config battery_config.yaml]
"""

import os
import csv
import sys
import math
import time
import argparse
from collections import Counter

import yaml
from pymatgen.core import Element, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester


# ── Polyanion central-atom electronegativity ──────────────────────────────────
# Maps polyanion-forming elements to their Pauling electronegativity.
# These atoms appear in XO₄ units (phosphate, sulfate, silicate, etc.) and
# withdraw electron density from TM via X–O–TM bridge (Goodenough inductive effect).
# Higher eneg → stronger withdrawal → higher TM d-band energy → higher voltage.
_POLYANION_ELEMENTS = {
    "P":  2.19,   # phosphate  PO₄³⁻  — LiFePO₄ ~3.5 V
    "S":  2.58,   # sulfate    SO₄²⁻  — Li₂Fe(SO₄)₂ ~3.9 V
    "Si": 1.90,   # silicate   SiO₄⁴⁻ — Li₂MnSiO₄ ~4.1 V
    "As": 2.18,   # arsenate   AsO₄³⁻
    "B":  2.04,   # borate     BO₃/BO₄
    "Mo": 2.16,   # molybdate  MoO₄²⁻
    "W":  2.36,   # tungstate  WO₄²⁻
    "Ge": 2.01,   # germanate  GeO₄⁴⁻
    "Nb": 1.60,   # niobate    NbO₄³⁻
    "Ta": 1.50,   # tantalate  TaO₄³⁻
    "Sb": 2.05,   # antimonate SbO₄³⁻
}

# Spacegroup → structural prototype integer encoding
# Encodes framework topology which controls Li mobility and volume change
_SG_PROTOTYPE = {
    62:  1,   # Pnma  — olivine (LiFePO₄ family)      low ΔV, stable
    227: 2,   # Fd-3m — spinel  (LiMn₂O₄ family)      low ΔV, 3D framework
    166: 3,   # R-3m  — layered (LiCoO₂/NMC family)   moderate ΔV, shear between layers
    12:  4,   # C2/m  — monoclinic layered (Li₂MnO₃)  moderate ΔV
    167: 5,   # R-3c  — NASICON-related framework      low ΔV, 3D channels
    15:  4,   # C2/c  — monoclinic layered variant
    148: 5,   # R-3   — rhombohedral framework
}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Crystal system encoding ───────────────────────────────────────────────────

_CRYSTAL_SYSTEM_INT = {
    "triclinic": 1, "monoclinic": 2, "orthorhombic": 3,
    "tetragonal": 4, "trigonal": 5, "hexagonal": 6, "cubic": 7,
}


# ── Element feature helpers ───────────────────────────────────────────────────

def element_features(symbol):
    """Return (eneg, atomic_radius_ang, max_common_oxidation_state)."""
    el     = Element(symbol)
    eneg   = float(el.X) if el.X is not None else float("nan")
    radius = float(el.atomic_radius) if el.atomic_radius is not None else float("nan")
    oxids  = el.common_oxidation_states
    max_ox = float(max(oxids)) if oxids else float("nan")
    return eneg, radius, max_ox


def tm_features(framework_comp, tm_set):
    """
    Return (tm_eneg, tm_ionic_radius, tm_max_oxidation_state, primary_tm)
    as composition-weighted averages over all TM atoms in the framework.
    primary_tm is the TM symbol with the highest stoichiometry.
    """
    tm_amounts = {str(el): framework_comp[el]
                  for el in framework_comp if str(el) in tm_set}
    if not tm_amounts:
        return float("nan"), float("nan"), float("nan"), ""

    total = sum(tm_amounts.values())
    eneg = radius = max_ox = 0.0
    for sym, amt in tm_amounts.items():
        w = amt / total
        e, r, ox = element_features(sym)
        eneg   += w * e
        radius += w * r
        max_ox += w * ox

    primary_tm = max(tm_amounts, key=tm_amounts.get)
    return eneg, radius, max_ox, primary_tm


def anion_eneg(framework_comp, tm_set):
    """
    Pauling electronegativity of the highest-eneg non-TM element
    (picks O for oxides/phosphates, F for fluorides, S for sulfides).
    """
    best_eneg = float("nan")
    for el in framework_comp:
        sym = str(el)
        if sym in tm_set:
            continue
        e, _, _ = element_features(sym)
        if math.isnan(e):
            continue
        if math.isnan(best_eneg) or e > best_eneg:
            best_eneg = e
    return best_eneg


def li_per_fu(formula_discharge):
    """Li atoms per formula unit in the discharge (lithiated) structure."""
    try:
        comp = Composition(formula_discharge)
        return float(comp[Element("Li")])
    except Exception:
        return float("nan")


def structure_features(host_structure):
    """
    Extract (spacegroup_number, crystal_system_int, volume_per_atom, density)
    from the Li-free framework structure.  Returns (0, 0, nan, nan) on failure.
    """
    if host_structure is None:
        return 0, 0, float("nan"), float("nan")
    try:
        sga    = SpacegroupAnalyzer(host_structure, symprec=0.1)
        sg_num = sga.get_space_group_number()
        cs_str = sga.get_crystal_system().lower()
        cs_int = _CRYSTAL_SYSTEM_INT.get(cs_str, 0)
    except Exception:
        sg_num, cs_int = 0, 0
    nsites       = host_structure.num_sites
    vol_per_atom = round(host_structure.volume / nsites, 4) if nsites > 0 else float("nan")
    density      = round(float(host_structure.density), 4)
    return sg_num, cs_int, vol_per_atom, density


# ── New physics features ──────────────────────────────────────────────────────

def polyanion_eneg(framework_comp, tm_set):
    """
    Return the Pauling electronegativity of the polyanion central atom.

    Scans the framework composition for elements in _POLYANION_ELEMENTS that
    are not TMs.  Returns the eneg of the one with highest stoichiometry
    (the dominant polyanion), or 0.0 for simple oxides/fluorides.

    Physical meaning: higher value → stronger inductive effect → higher voltage.
    """
    best_eneg = 0.0
    best_amt  = 0.0
    for el in framework_comp:
        sym = str(el)
        if sym in tm_set:
            continue
        eneg = _POLYANION_ELEMENTS.get(sym)
        if eneg is None:
            continue
        amt = framework_comp[el]
        if amt > best_amt:
            best_eneg = eneg
            best_amt  = amt
    return round(best_eneg, 4)


def structure_prototype(sg_num):
    """
    Integer encoding of the framework structural prototype from spacegroup number.
    0 = other/unknown.  See _SG_PROTOTYPE for the mapping.
    """
    return _SG_PROTOTYPE.get(sg_num, 0)


# ── MP query ──────────────────────────────────────────────────────────────────

def query_tm_oxidation_states(mpr, material_ids, tm_set, max_retries, base_delay):
    """
    Batch-fetch oxidation states from MP summary endpoint.
    Returns dict: material_id → average oxidation state of primary TM.
    Falls back to nan on failure.
    """
    # Strip the _Li suffix to get material_ids
    mpids = list({bid.split("_")[0] for bid in material_ids})
    for attempt in range(max_retries):
        try:
            docs = mpr.materials.summary.search(
                material_ids=mpids,
                all_fields=False,
                fields=["material_id", "possible_species"],
            )
            break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Warning: oxidation state fetch failed: {e}")
                return {}
            wait = base_delay * (2 ** attempt)
            time.sleep(wait)
    else:
        return {}

    result = {}
    for doc in docs:
        if not doc.possible_species:
            continue
        # possible_species = list of strings like ["Fe2+", "O2-", "P5+"]
        tm_ox_vals = []
        for species_str in doc.possible_species:
            # Parse element and charge: "Fe2+" → ("Fe", 2), "Fe3+" → ("Fe", 3)
            try:
                import re
                m = re.match(r"([A-Z][a-z]?)(\d+)([+-])", species_str)
                if m:
                    elem, val, sign = m.group(1), int(m.group(2)), m.group(3)
                    ox = val if sign == "+" else -val
                    if elem in tm_set:
                        tm_ox_vals.append(ox)
            except Exception:
                continue
        if tm_ox_vals:
            result[doc.material_id] = round(sum(tm_ox_vals) / len(tm_ox_vals), 4)
    return result


def query_insertion_electrodes(mpr, cfg, max_retries, base_delay):
    """Single batch query for all Li insertion electrodes within voltage/volume filters."""
    bat = cfg["battery"]
    for attempt in range(max_retries):
        try:
            return mpr.materials.insertion_electrodes.search(
                working_ion=bat["working_ion"],
                average_voltage=(bat["voltage_min"], bat["voltage_max"]),
                max_delta_volume=(0.0, bat["max_delta_volume"]),
                all_fields=False,
                fields=[
                    "battery_id", "battery_formula", "framework_formula",
                    "average_voltage", "max_delta_volume",
                    "formula_discharge", "elements",
                    "framework", "host_structure",
                    "thermo_type", "stability_discharge",
                    "capacity_grav",
                ]
            )
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  Error querying insertion electrodes after {max_retries} attempts: {e}")
                return []
            wait = base_delay * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{max_retries} in {wait:.0f}s: {e}")
            time.sleep(wait)
    return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path):
    MP_API_KEY = os.environ.get("MP_API_KEY")
    if not MP_API_KEY:
        print("Error: set MP_API_KEY environment variable first.")
        sys.exit(1)

    cfg             = load_config(config_path)
    output_file     = cfg["files"]["candidates"]
    TM_SET          = set(cfg["battery"]["tm_elements"])
    HULL_THRESHOLD  = cfg["battery"]["hull_threshold"]
    max_retries     = cfg["api"]["max_retries"]
    base_delay      = cfg["api"]["retry_base_delay"]

    print(f"Fetching Li-intercalation cathode candidates from Materials Project…")
    print(f"Voltage  : {cfg['battery']['voltage_min']}–{cfg['battery']['voltage_max']} V vs Li/Li⁺")
    print(f"ΔV_max   : ≤ {cfg['battery']['max_delta_volume'] * 100:.0f}% volume change")
    print(f"Hull     : discharge stability ≤ {HULL_THRESHOLD} eV/atom")
    print(f"TM pool  : {sorted(TM_SET)}\n")

    with MPRester(MP_API_KEY) as mpr:
        docs = query_insertion_electrodes(mpr=mpr, cfg=cfg,
                                          max_retries=max_retries,
                                          base_delay=base_delay)
    print(f"MP returned {len(docs)} insertion electrode entries before filtering.\n")

    rows = []
    skipped_hull = skipped_tm = skipped_struct = 0

    for doc in docs:
        # ── Hull stability filter on discharge structure ───────────────────
        stab = doc.stability_discharge
        if stab is None or float(stab) > HULL_THRESHOLD:
            skipped_hull += 1
            continue

        # ── TM presence filter ─────────────────────────────────────────────
        elem_set = {str(e) for e in (doc.elements or [])}
        if not elem_set.intersection(TM_SET):
            skipped_tm += 1
            continue

        # ── Framework composition and TM features ─────────────────────────
        if doc.framework is None:
            skipped_struct += 1
            continue
        framework_comp = doc.framework

        tm_e, tm_r, tm_ox, primary_tm = tm_features(framework_comp, TM_SET)
        if not primary_tm:
            skipped_tm += 1
            continue

        a_eneg  = anion_eneg(framework_comp, TM_SET)
        li_cnt  = li_per_fu(doc.formula_discharge or "")
        sg_num, cs_int, vol_per_atom, density = structure_features(doc.host_structure)
        pol_eneg = polyanion_eneg(framework_comp, TM_SET)
        sg_proto = structure_prototype(sg_num)

        rows.append({
            "battery_id":             doc.battery_id,
            "framework_formula":      doc.framework_formula or "",
            "primary_tm":             primary_tm,
            "formula_discharge":      doc.formula_discharge or "",
            "tm_eneg":                round(tm_e,   4),
            "tm_ionic_radius":        round(tm_r,   4),
            "tm_max_oxidation_state": round(tm_ox,  4),
            "anion_eneg":             round(a_eneg, 4),
            "li_per_fu":              round(li_cnt, 4),
            "spacegroup_number":      sg_num,
            "crystal_system":         cs_int,
            "volume_per_atom":        vol_per_atom,
            "density":                density,
            "polyanion_eneg":         pol_eneg,
            "structure_prototype":    sg_proto,
            "tm_oxidation_discharge": "",     # filled by batch query below
            # Objectives left blank — oracle fills lazily
            "average_voltage":        "",
            "max_delta_volume":       "",
            # Keep for preview / reporting (not a feature)
            "_voltage_for_dedup":     float(doc.average_voltage or 0),
        })

    print(f"After filtering:")
    print(f"  Skipped (hull)          : {skipped_hull}")
    print(f"  Skipped (no TM / frame) : {skipped_tm + skipped_struct}")
    print(f"  Kept                    : {len(rows)}")

    # ── Deduplicate: one entry per framework_formula, keep highest voltage ─
    best: dict[str, dict] = {}
    for row in rows:
        key  = row["framework_formula"]
        volt = row["_voltage_for_dedup"]
        if key not in best or volt > best[key]["_voltage_for_dedup"]:
            best[key] = row

    before = len(rows)
    rows   = sorted(best.values(), key=lambda r: r["framework_formula"])
    print(f"\nDeduplicated {before} → {len(rows)} unique framework compositions "
          f"(kept highest-voltage entry per formula)")

    # Remove the internal dedup helper column before writing
    for row in rows:
        del row["_voltage_for_dedup"]

    # ── Batch-fetch TM oxidation states from MP summary ────────────────────
    print(f"\nFetching TM oxidation states for {len(rows)} candidates…")
    battery_ids = [r["battery_id"] for r in rows]
    with MPRester(MP_API_KEY) as mpr:
        ox_lookup = query_tm_oxidation_states(
            mpr, battery_ids, TM_SET, max_retries, base_delay
        )
    filled_ox = 0
    for row in rows:
        mpid = row["battery_id"].split("_")[0]
        ox   = ox_lookup.get(mpid)
        if ox is not None:
            row["tm_oxidation_discharge"] = ox
            filled_ox += 1
        else:
            row["tm_oxidation_discharge"] = float("nan")
    print(f"  Oxidation states filled: {filled_ox}/{len(rows)}")

    if not rows:
        print(f"\nNo candidates found. Try relaxing filters in {config_path}.")
        sys.exit(1)

    print(f"\nTotal candidates: {len(rows)}")

    fieldnames = [
        "battery_id", "framework_formula", "primary_tm", "formula_discharge",
        "tm_eneg", "tm_ionic_radius", "tm_max_oxidation_state", "anion_eneg",
        "li_per_fu", "spacegroup_number", "crystal_system",
        "volume_per_atom", "density",
        "polyanion_eneg", "structure_prototype", "tm_oxidation_discharge",
        "average_voltage", "max_delta_volume",
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
    tm_counts  = Counter(r["primary_tm"] for r in rows)
    cs_names   = {v: k for k, v in _CRYSTAL_SYSTEM_INT.items()}
    cs_counts  = Counter(cs_names.get(r["crystal_system"], "?") for r in rows)
    print(f"\nBy primary TM  : {dict(tm_counts)}")
    print(f"By crystal sys : {dict(cs_counts)}")
    print(f"\nNext step:")
    print(f"  python mp_nimo_battery.py --config {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="battery_config.yaml")
    args = parser.parse_args()
    main(args.config)
