"""
NIMO Bayesian optimization loop for antiperovskite (M₃AX) candidates.

The Materials Project API acts as the oracle — no robot needed.
Goal: find the most stable (lowest formation_energy_per_atom) antiperovskite
using the fewest possible queries.

Features:
  - Checkpoint/resume: crashes mid-run restart from last completed cycle
  - Retry with exponential backoff for MP API failures
  - All parameters driven by antiperovskite_config.yaml

Usage:
    export MP_API_KEY="your_key_here"
    python mp_nimo_antiperovskite.py                       # PHYSBO (default from config)
    python mp_nimo_antiperovskite.py --method BLOX
    python mp_nimo_antiperovskite.py --method RE           # random baseline
    python mp_nimo_antiperovskite.py --config my.yaml      # custom config

Prerequisites:
    Run mp_fetch_antiperovskite.py first to generate the candidates CSV.
"""

import os
import sys
import csv
import json
import time
import argparse

import yaml
import numpy as np
import matplotlib.pyplot as plt

import nimo
from mp_api.client import MPRester


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(path, method, next_cycle, best_curve, all_queried):
    data = {
        "method":      method,
        "next_cycle":  next_cycle,
        "best_curve":  best_curve,
        "all_queried": [list(r) for r in all_queried],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_candidates(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save_candidates(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_nimo_csv(candidates, path, feature_cols):
    """Write numeric-only NIMO CSV: feature cols + objective last."""
    header = feature_cols + ["formation_energy_per_atom"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for c in candidates:
            row = [c[col] for col in feature_cols]
            row.append(c["formation_energy_per_atom"])
            writer.writerow(row)


def read_proposals(proposals_path, candidates):
    """Return list of (candidate_index, candidate_row) from NIMO proposals CSV."""
    proposed = []
    with open(proposals_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["actions"])
            proposed.append((idx, candidates[idx]))
    return proposed


def measured_values(candidates):
    return [
        float(c["formation_energy_per_atom"])
        for c in candidates
        if c["formation_energy_per_atom"] != ""
    ]


# ── MP API with retry ─────────────────────────────────────────────────────────

def query_mp(material_id, mp_api_key, max_retries, base_delay, inter_delay):
    """Return formation_energy_per_atom for a material_id, with retry."""
    for attempt in range(max_retries):
        try:
            with MPRester(mp_api_key) as mpr:
                docs = mpr.materials.summary.search(
                    material_ids=[material_id],
                    fields=["formation_energy_per_atom"]
                )
            time.sleep(inter_delay)
            if not docs or docs[0].formation_energy_per_atom is None:
                return None
            return float(docs[0].formation_energy_per_atom)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  API error for {material_id} after {max_retries} attempts: {e}")
                return None
            wait = base_delay * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{max_retries} for {material_id} in {wait:.0f}s")
            time.sleep(wait)
    return None


# ── Main loop ─────────────────────────────────────────────────────────────────

def main(method, config_path):
    MP_API_KEY = os.environ.get("MP_API_KEY")
    if not MP_API_KEY:
        print("Error: set MP_API_KEY environment variable first.")
        sys.exit(1)

    cfg = load_config(config_path)

    CANDIDATES_FILE = cfg["files"]["candidates"]
    PROPOSALS_FILE  = cfg["files"]["proposals"]
    NIMO_CSV        = cfg["files"]["nimo_csv"]
    CHECKPOINT_FILE = cfg["files"]["checkpoint"]
    PLOT_PREFIX     = cfg["files"]["plot_prefix"]
    NUM_CYCLES      = cfg["nimo"]["num_cycles"]
    SEED_CYCLES     = cfg["nimo"]["seed_cycles"]
    NUM_OBJECTIVES  = cfg["nimo"]["num_objectives"]
    NUM_PROPOSALS   = cfg["nimo"]["num_proposals"]
    FEATURE_COLS    = cfg["feature_cols"]
    MAX_RETRIES     = cfg["api"]["max_retries"]
    BASE_DELAY      = cfg["api"]["retry_base_delay"]
    INTER_DELAY     = cfg["api"]["inter_call_delay"]

    candidates = load_candidates(CANDIDATES_FILE)
    fieldnames = list(candidates[0].keys())

    # ── Resume from checkpoint if available ───────────────────────────────────
    ckpt = load_checkpoint(CHECKPOINT_FILE)
    if ckpt:
        start_cycle = ckpt["next_cycle"]
        best_curve  = ckpt["best_curve"]
        all_queried = [tuple(r) for r in ckpt["all_queried"]]
        already     = sum(1 for c in candidates if c["formation_energy_per_atom"] != "")
        print(f"Resuming from checkpoint: cycle {start_cycle + 1}/{NUM_CYCLES} "
              f"({already} already measured)")
    else:
        start_cycle = 0
        best_curve  = []
        all_queried = []

    n_total = len(candidates)
    print(f"Candidates: {n_total} | Method: {method} | "
          f"Seed cycles: {SEED_CYCLES} | Total cycles: {NUM_CYCLES}\n")

    history = None

    for cycle in range(start_cycle, NUM_CYCLES):

        remaining = sum(1 for c in candidates if c["formation_energy_per_atom"] == "")
        if remaining == 0:
            print(f"\nAll candidates measured after cycle {cycle}. Stopping early.")
            break

        current_method = "RE" if cycle < SEED_CYCLES else method
        print(f"── Cycle {cycle + 1:2d}/{NUM_CYCLES} [{current_method}] ({remaining} unmeasured) ──")

        build_nimo_csv(candidates, NIMO_CSV, FEATURE_COLS)

        nimo.selection(
            method=current_method,
            input_file=NIMO_CSV,
            output_file=PROPOSALS_FILE,
            num_objectives=NUM_OBJECTIVES,
            num_proposals=NUM_PROPOSALS,
            minimization=True,
        )

        proposed = read_proposals(PROPOSALS_FILE, candidates)

        for idx, cand in proposed:
            mid     = cand["material_id"]
            formula = cand["formula"]
            m, a, x = cand["M"], cand["A"], cand["X"]

            if cand["formation_energy_per_atom"] != "":
                print(f"  Already measured {formula}, skipping.")
                continue

            print(f"  Querying: {formula} ({mid})  [{m}₃{a}{x}]")
            value = query_mp(mid, MP_API_KEY, MAX_RETRIES, BASE_DELAY, INTER_DELAY)

            if value is None:
                print(f"  No data returned for {mid}.")
                continue

            print(f"  formation_energy_per_atom = {value:.4f} eV/atom")
            candidates[idx]["formation_energy_per_atom"] = str(round(value, 6))
            all_queried.append((cycle + 1, formula, mid, value))

        # Persist candidates and checkpoint after every cycle
        save_candidates(CANDIDATES_FILE, candidates, fieldnames)

        vals = measured_values(candidates)
        if vals:
            best = min(vals)
            best_curve.append(best)
            best_cand = next(
                c for c in candidates
                if c["formation_energy_per_atom"] != ""
                and abs(float(c["formation_energy_per_atom"]) - best) < 1e-6
            )
            print(f"  Best so far: {best:.4f} eV/atom  "
                  f"({best_cand['formula']}, {best_cand['material_id']})\n")

        build_nimo_csv(candidates, NIMO_CSV, FEATURE_COLS)
        history = nimo.history(
            input_file=NIMO_CSV,
            num_objectives=NUM_OBJECTIVES,
            itt=cycle,
            history_file=history,
        )

        save_checkpoint(CHECKPOINT_FILE, method, cycle + 1, best_curve, all_queried)

    # ── Clean up checkpoint on successful completion ───────────────────────────
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint removed (run complete).")

    # ── Final report ──────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"Optimization complete — {method}, {len(all_queried)} queries")

    if best_curve:
        best_val = best_curve[-1]
        best_c = next(
            c for c in candidates
            if c["formation_energy_per_atom"] != ""
            and abs(float(c["formation_energy_per_atom"]) - best_val) < 1e-6
        )
        print(f"Best material     : {best_c['formula']}  ({best_c['material_id']})")
        print(f"Best Eform        : {best_val:.4f} eV/atom")
        print(f"Sites             : M={best_c['M']}, A={best_c['A']}, X={best_c['X']}")
        print(f"Tolerance factor  : {best_c['tolerance_factor']}")

    print(f"\n{'Cycle':>6}  {'Formula':<16}  {'ID':<14}  {'Eform (eV/at)':>13}")
    print("  " + "-" * 55)
    for cyc, frm, mid, val in sorted(all_queried, key=lambda r: r[3]):
        print(f"  {cyc:>5}  {frm:<16}  {mid:<14}  {val:>13.4f}")

    # ── Convergence plot ──────────────────────────────────────────────────────
    if len(best_curve) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(range(1, len(best_curve) + 1), best_curve,
                     marker="o", color="steelblue", linewidth=2)
        axes[0].axvline(SEED_CYCLES, color="gray", linestyle="--",
                        alpha=0.6, label="RE → AI switch")
        axes[0].set_xlabel("Cycle")
        axes[0].set_ylabel("Best formation energy (eV/atom)")
        axes[0].set_title(f"NIMO [{method}] — Antiperovskite M₃AX")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        cycs = [r[0] for r in all_queried]
        vals = [r[3] for r in all_queried]
        axes[1].scatter(cycs, vals, c=vals, cmap="coolwarm_r", zorder=3)
        axes[1].set_xlabel("Cycle")
        axes[1].set_ylabel("Measured Eform (eV/atom)")
        axes[1].set_title("All measured candidates")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = f"{PLOT_PREFIX}_{method}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"\nConvergence plot → {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None,
                        choices=["PHYSBO", "BLOX", "RE", "NTS", "AX"],
                        help="NIMO AI method (overrides config default)")
    parser.add_argument("--config", default="antiperovskite_config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    method = args.method or cfg["nimo"]["default_method"]
    main(method, args.config)
