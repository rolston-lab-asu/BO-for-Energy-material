"""
NIMO multi-objective Bayesian optimization for photovoltaic candidates.

Two objectives (both minimized simultaneously via Pareto front search):
  1. formation_energy_per_atom   — thermodynamic stability
  2. |band_gap - bg_target|      — proximity to Shockley-Queisser optimum (1.34 eV)

The Materials Project API serves as the oracle: each proposed candidate is
queried for both formation_energy_per_atom and band_gap in a single call.

Features:
  - Checkpoint/resume
  - Retry with exponential backoff for MP API failures
  - All parameters driven by pv_config.yaml
  - Pareto-front plot at the end

Usage:
    export MP_API_KEY="your_key_here"
    python mp_nimo_pv.py                    # PHYSBO (default)
    python mp_nimo_pv.py --method BLOX
    python mp_nimo_pv.py --method RE        # random baseline
    python mp_nimo_pv.py --config my.yaml

Prerequisites:
    Run mp_fetch_pv.py first to generate the candidates CSV.
"""

import os
import sys
import csv
import json
import time
import argparse

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nimo
from mp_api.client import MPRester


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(path, method, next_cycle, curves, all_queried):
    data = {
        "method":      method,
        "next_cycle":  next_cycle,
        "curves":      curves,               # {"best_fe": [...], "best_bg_dev": [...]}
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


def build_nimo_csv(candidates, path, feature_cols, bg_target):
    """
    Write NIMO CSV with 2 objectives last:
      col[-2] = formation_energy_per_atom
      col[-1] = |band_gap - bg_target|

    Both are blank (→ NaN) until the oracle fills them in together.
    NIMO treats a row as training data only when both objectives are non-NaN.
    """
    header = feature_cols + ["formation_energy_per_atom", "band_gap_dev"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for c in candidates:
            row = [c[col] for col in feature_cols]
            fe  = c["formation_energy_per_atom"]
            bg  = c["band_gap"]
            if fe != "" and bg != "":
                bg_dev = abs(float(bg) - bg_target)
                row.extend([fe, str(round(bg_dev, 6))])
            else:
                row.extend(["", ""])
            writer.writerow(row)


def read_proposals(proposals_path, candidates):
    proposed = []
    with open(proposals_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["actions"])
            proposed.append((idx, candidates[idx]))
    return proposed


def measured_candidates(candidates):
    """Return candidates where BOTH objectives are known."""
    return [c for c in candidates
            if c["formation_energy_per_atom"] != "" and c["band_gap"] != ""]


# ── Pareto front ──────────────────────────────────────────────────────────────

def pareto_front(points):
    """
    Return indices of Pareto-optimal points minimizing both objectives.
    points: list of (obj1, obj2)
    """
    dominated = [False] * len(points)
    for i, pi in enumerate(points):
        for j, pj in enumerate(points):
            if i == j:
                continue
            if (pj[0] <= pi[0] and pj[1] <= pi[1] and
                    (pj[0] < pi[0] or pj[1] < pi[1])):
                dominated[i] = True
                break
    return [i for i in range(len(points)) if not dominated[i]]


# ── MP API with retry ─────────────────────────────────────────────────────────

def query_mp(material_id, mp_api_key, max_retries, base_delay, inter_delay):
    """
    Return (formation_energy_per_atom, band_gap) for a material_id.
    Either value may be None if MP has no data.
    """
    for attempt in range(max_retries):
        try:
            with MPRester(mp_api_key) as mpr:
                docs = mpr.materials.summary.search(
                    material_ids=[material_id],
                    fields=["formation_energy_per_atom", "band_gap"]
                )
            time.sleep(inter_delay)
            if not docs:
                return None, None
            doc = docs[0]
            fe  = float(doc.formation_energy_per_atom) if doc.formation_energy_per_atom is not None else None
            bg  = float(doc.band_gap)                  if doc.band_gap is not None else None
            return fe, bg
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  API error for {material_id} after {max_retries} attempts: {e}")
                return None, None
            wait = base_delay * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{max_retries} for {material_id} in {wait:.0f}s")
            time.sleep(wait)
    return None, None


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
    NUM_OBJECTIVES  = cfg["nimo"]["num_objectives"]   # 2
    NUM_PROPOSALS   = cfg["nimo"]["num_proposals"]
    FEATURE_COLS    = cfg["feature_cols"]
    MAX_RETRIES     = cfg["api"]["max_retries"]
    BASE_DELAY      = cfg["api"]["retry_base_delay"]
    INTER_DELAY     = cfg["api"]["inter_call_delay"]
    BG_TARGET       = cfg["pv"]["bg_target"]

    candidates = load_candidates(CANDIDATES_FILE)
    fieldnames = list(candidates[0].keys())

    # ── Resume from checkpoint ─────────────────────────────────────────────────
    ckpt = load_checkpoint(CHECKPOINT_FILE)
    if ckpt:
        start_cycle = ckpt["next_cycle"]
        curves      = ckpt["curves"]
        all_queried = [tuple(r) for r in ckpt["all_queried"]]
        already     = len(measured_candidates(candidates))
        print(f"Resuming from checkpoint: cycle {start_cycle + 1}/{NUM_CYCLES} "
              f"({already} already measured)")
    else:
        start_cycle = 0
        curves      = {"best_fe": [], "best_bg_dev": []}
        all_queried = []

    n_total = len(candidates)
    print(f"Candidates: {n_total} | Method: {method} | "
          f"Seed cycles: {SEED_CYCLES} | Total cycles: {NUM_CYCLES}")
    print(f"Objectives: minimize formation_energy  +  |band_gap − {BG_TARGET} eV|\n")

    history = None

    for cycle in range(start_cycle, NUM_CYCLES):

        remaining = sum(1 for c in candidates
                        if c["formation_energy_per_atom"] == "" or c["band_gap"] == "")
        if remaining == 0:
            print(f"\nAll candidates measured after cycle {cycle}. Stopping early.")
            break

        current_method = "RE" if cycle < SEED_CYCLES else method
        print(f"── Cycle {cycle + 1:2d}/{NUM_CYCLES} [{current_method}] "
              f"({remaining} unmeasured) ──")

        build_nimo_csv(candidates, NIMO_CSV, FEATURE_COLS, BG_TARGET)

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

            if cand["formation_energy_per_atom"] != "" and cand["band_gap"] != "":
                print(f"  Already measured {formula}, skipping.")
                continue

            print(f"  Querying: {formula} ({mid})")
            fe, bg = query_mp(mid, MP_API_KEY, MAX_RETRIES, BASE_DELAY, INTER_DELAY)

            if fe is None or bg is None:
                print(f"  Incomplete data for {mid} (fe={fe}, bg={bg}), skipping.")
                continue

            bg_dev = abs(bg - BG_TARGET)
            print(f"  Eform = {fe:+.4f} eV/atom   band_gap = {bg:.4f} eV   "
                  f"|bg − {BG_TARGET}| = {bg_dev:.4f} eV")

            candidates[idx]["formation_energy_per_atom"] = str(round(fe, 6))
            candidates[idx]["band_gap"]                  = str(round(bg, 6))
            all_queried.append((cycle + 1, formula, mid, fe, bg, bg_dev))

        save_candidates(CANDIDATES_FILE, candidates, fieldnames)

        # ── Track best-so-far for each objective independently ─────────────────
        meas = measured_candidates(candidates)
        if meas:
            fe_vals    = [float(c["formation_energy_per_atom"]) for c in meas]
            bg_devs    = [abs(float(c["band_gap"]) - BG_TARGET) for c in meas]

            best_fe     = min(fe_vals)
            best_bg_dev = min(bg_devs)
            curves["best_fe"].append(best_fe)
            curves["best_bg_dev"].append(best_bg_dev)

            # Most stable semiconductor found
            best_fe_c = next(c for c in meas
                             if abs(float(c["formation_energy_per_atom"]) - best_fe) < 1e-5)
            # Closest to SQ optimum
            best_bg_c = next(c for c in meas
                             if abs(abs(float(c["band_gap"]) - BG_TARGET) - best_bg_dev) < 1e-5)

            print(f"  Best Eform : {best_fe:+.4f} eV/atom  "
                  f"({best_fe_c['formula']}, bg={float(best_fe_c['band_gap']):.3f} eV)")
            print(f"  Best bg    : {float(best_bg_c['band_gap']):.4f} eV  "
                  f"({best_bg_c['formula']}, Δ={best_bg_dev:.4f} eV)\n")

        build_nimo_csv(candidates, NIMO_CSV, FEATURE_COLS, BG_TARGET)
        history = nimo.history(
            input_file=NIMO_CSV,
            num_objectives=NUM_OBJECTIVES,
            itt=cycle,
            history_file=history,
        )

        save_checkpoint(CHECKPOINT_FILE, method, cycle + 1, curves, all_queried)

    # ── Clean up checkpoint ────────────────────────────────────────────────────
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint removed (run complete).")

    # ── Final report ──────────────────────────────────────────────────────────
    meas = measured_candidates(candidates)
    print("=" * 60)
    print(f"Optimization complete — {method}, {len(all_queried)} queries\n")

    if meas:
        fe_vals = [float(c["formation_energy_per_atom"]) for c in meas]
        bg_vals = [float(c["band_gap"]) for c in meas]
        bg_devs = [abs(bg - BG_TARGET) for bg in bg_vals]

        # Pareto front
        points   = list(zip(fe_vals, bg_devs))
        pf_idx   = pareto_front(points)
        pf_cands = [meas[i] for i in pf_idx]
        pf_cands.sort(key=lambda c: float(c["band_gap"]))

        print(f"Pareto front ({len(pf_cands)} materials):")
        print(f"  {'Formula':<16}  {'ID':<14}  {'Eform':>9}  {'Band gap':>9}  "
              f"{'Δbg':>8}")
        print("  " + "-" * 60)
        for c in pf_cands:
            fe  = float(c["formation_energy_per_atom"])
            bg  = float(c["band_gap"])
            dev = abs(bg - BG_TARGET)
            print(f"  {c['formula']:<16}  {c['material_id']:<14}  "
                  f"{fe:>+9.4f}  {bg:>9.4f}  {dev:>8.4f}")

    print(f"\n{'Cyc':>4}  {'Formula':<16}  {'ID':<14}  "
          f"{'Eform':>9}  {'BG':>7}  {'Δbg':>7}")
    print("  " + "-" * 60)
    for row in sorted(all_queried, key=lambda r: r[5]):   # sort by Δbg
        cyc, frm, mid, fe, bg, dev = row
        print(f"  {cyc:>3}  {frm:<16}  {mid:<14}  "
              f"{fe:>+9.4f}  {bg:>7.4f}  {dev:>7.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if len(curves["best_fe"]) > 1 and meas:
        fig = plt.figure(figsize=(16, 5))
        gs  = fig.add_gridspec(1, 3, wspace=0.35)

        cycles_ax = range(1, len(curves["best_fe"]) + 1)

        # Panel 1: best formation energy per cycle
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(cycles_ax, curves["best_fe"], marker="o", color="steelblue", linewidth=2)
        ax1.axvline(SEED_CYCLES, color="gray", linestyle="--", alpha=0.6,
                    label="RE → AI switch")
        ax1.set_xlabel("Cycle")
        ax1.set_ylabel("Best formation energy (eV/atom)")
        ax1.set_title(f"[{method}] Stability convergence")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2: best |band_gap - target| per cycle
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(cycles_ax, curves["best_bg_dev"], marker="s", color="darkorange",
                 linewidth=2)
        ax2.axvline(SEED_CYCLES, color="gray", linestyle="--", alpha=0.6,
                    label="RE → AI switch")
        ax2.set_xlabel("Cycle")
        ax2.set_ylabel(f"|band_gap − {BG_TARGET}| (eV)")
        ax2.set_title(f"[{method}] Band gap convergence")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Pareto front scatter (Eform vs band_gap)
        ax3 = fig.add_subplot(gs[2])
        fe_m  = [float(c["formation_energy_per_atom"]) for c in meas]
        bg_m  = [float(c["band_gap"]) for c in meas]
        dev_m = [abs(bg - BG_TARGET) for bg in bg_m]

        sc = ax3.scatter(fe_m, bg_m, c=dev_m, cmap="RdYlGn_r",
                         s=60, zorder=3, label="Measured")

        # Highlight Pareto front
        pf_fe  = [fe_m[i] for i in pf_idx]
        pf_bg  = [bg_m[i] for i in pf_idx]
        ax3.scatter(pf_fe, pf_bg, s=120, facecolors="none",
                    edgecolors="black", linewidths=1.5,
                    zorder=4, label="Pareto front")

        ax3.axhline(BG_TARGET, color="green", linestyle="--", alpha=0.7,
                    label=f"SQ optimum ({BG_TARGET} eV)")
        ax3.axhspan(1.1, 1.7, color="green", alpha=0.06, label="PV window (1.1–1.7 eV)")

        plt.colorbar(sc, ax=ax3, label=f"|bg − {BG_TARGET}| (eV)")
        ax3.set_xlabel("Formation energy (eV/atom)")
        ax3.set_ylabel("Band gap (eV)")
        ax3.set_title("Pareto front: stability vs band gap")
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)

        plot_path = f"{PLOT_PREFIX}_{method}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot → {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None,
                        choices=["PHYSBO", "BLOX", "RE", "NTS", "AX"],
                        help="NIMO AI method (overrides config default)")
    parser.add_argument("--config", default="pv_config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    method = args.method or cfg["nimo"]["default_method"]
    main(method, args.config)
