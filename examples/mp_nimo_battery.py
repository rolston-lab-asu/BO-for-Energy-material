"""
NIMO multi-objective Bayesian optimization for Li-intercalation cathodes.

Two objectives (both minimized simultaneously):
  1. neg_avg_voltage  = -average_voltage   → maximise voltage
  2. max_delta_volume = fractional volume change on lithiation → minimise

No scissor correction needed — MP voltages are DFT-computed and reasonable
for relative screening without systematic correction.

Oracle queries the MP insertion_electrodes endpoint by battery_id per proposal.

Usage:
    export MP_API_KEY="your_key_here"
    python mp_nimo_battery.py                  # PHYSBO (default)
    python mp_nimo_battery.py --method BLOX
    python mp_nimo_battery.py --config battery_config.yaml

Prerequisites:
    Run mp_fetch_battery.py first.
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
        "curves":      curves,
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


def measured_candidates(candidates):
    return [c for c in candidates
            if c["average_voltage"] != "" and c["max_delta_volume"] != ""]


def build_nimo_csv(candidates, path, feature_cols, practical_max):
    """
    Write NIMO CSV with 2 objectives as the last two columns:
      col[-2] = neg_practical_voltage  = -min(voltage, practical_max)
      col[-1] = max_delta_volume
    Capping at practical_max keeps NIMO optimising within the electrolyte
    stability window rather than chasing DFT-only high-voltage outliers.
    Both blank until oracle fills them.
    """
    header = feature_cols + ["neg_avg_voltage", "max_delta_volume"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for c in candidates:
            row = [c[col] for col in feature_cols]
            volt = c["average_voltage"]
            mdv  = c["max_delta_volume"]
            if volt != "" and mdv != "":
                practical_volt = min(float(volt), practical_max)
                row += [round(-practical_volt, 6), mdv]
            else:
                row += ["", ""]
            writer.writerow(row)


def read_proposals(proposals_file, candidates):
    """Match NIMO proposals back to candidates by row index."""
    with open(proposals_file) as f:
        reader = csv.DictReader(f)
        proposed = []
        for prow in reader:
            # Skip the 'actions' index column; take only numeric feature columns
            feat_vals = [float(v) for k, v in prow.items()
                         if k != "actions" and v.strip() != ""]
            for idx, cand in enumerate(candidates):
                try:
                    cand_vals = [float(cand[col]) for col in
                                 [k for k in cand if k not in
                                  ("battery_id", "framework_formula", "primary_tm",
                                   "formula_discharge", "average_voltage", "max_delta_volume")]]
                except ValueError:
                    continue
                if len(feat_vals) == len(cand_vals) and all(
                        abs(a - b) < 1e-6 for a, b in zip(feat_vals, cand_vals)):
                    proposed.append((idx, cand))
                    break
    return proposed


# ── Pareto front ──────────────────────────────────────────────────────────────

def pareto_front(points):
    """Return indices of non-dominated points (both objectives minimised)."""
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


# ── MP oracle ─────────────────────────────────────────────────────────────────

def query_mp_battery(battery_id, mp_api_key, max_retries, base_delay, inter_delay):
    """
    Return (average_voltage, max_delta_volume) for a given battery_id.
    Either value may be None on failure.
    """
    for attempt in range(max_retries):
        try:
            with MPRester(mp_api_key) as mpr:
                docs = mpr.materials.insertion_electrodes.search(
                    battery_ids=[battery_id],
                    all_fields=False,
                    fields=["average_voltage", "max_delta_volume"]
                )
            time.sleep(inter_delay)
            if not docs:
                return None, None
            doc = docs[0]
            volt = float(doc.average_voltage) if doc.average_voltage is not None else None
            mdv  = float(doc.max_delta_volume) if doc.max_delta_volume is not None else None
            return volt, mdv
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  API error for {battery_id} after {max_retries} attempts: {e}")
                return None, None
            wait = base_delay * (2 ** attempt)
            print(f"  Retry {attempt + 1}/{max_retries} for {battery_id} in {wait:.0f}s")
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
    NUM_OBJECTIVES  = cfg["nimo"]["num_objectives"]
    NUM_PROPOSALS   = cfg["nimo"]["num_proposals"]
    FEATURE_COLS    = cfg["feature_cols"]
    PRACTICAL_MAX   = cfg["battery"].get("voltage_practical_max", 5.5)
    MAX_RETRIES     = cfg["api"]["max_retries"]
    BASE_DELAY      = cfg["api"]["retry_base_delay"]
    INTER_DELAY     = cfg["api"]["inter_call_delay"]

    candidates = load_candidates(CANDIDATES_FILE)
    fieldnames = list(candidates[0].keys())

    # ── Resume from checkpoint ─────────────────────────────────────────────
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
        curves      = {"best_voltage": [], "best_delta_volume": []}
        all_queried = []

    print(f"Candidates : {len(candidates)} | Method: {method} | "
          f"Seed cycles: {SEED_CYCLES} | Total cycles: {NUM_CYCLES}")
    print(f"Objectives : maximise voltage  +  minimise volume change\n")

    history = None

    for cycle in range(start_cycle, NUM_CYCLES):

        remaining = sum(1 for c in candidates
                        if c["average_voltage"] == "" or c["max_delta_volume"] == "")
        if remaining == 0:
            print(f"\nAll candidates measured after cycle {cycle}. Stopping early.")
            break

        current_method = "RE" if cycle < SEED_CYCLES else method
        print(f"── Cycle {cycle + 1:2d}/{NUM_CYCLES} [{current_method}] "
              f"({remaining} unmeasured) ──")

        build_nimo_csv(candidates, NIMO_CSV, FEATURE_COLS, PRACTICAL_MAX)

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
            bid     = cand["battery_id"]
            formula = cand["framework_formula"]
            tm      = cand["primary_tm"]

            if cand["average_voltage"] != "" and cand["max_delta_volume"] != "":
                print(f"  Already measured {formula}, skipping.")
                continue

            print(f"  Querying: {formula} ({bid})  [TM={tm}]")
            volt, mdv = query_mp_battery(
                bid, MP_API_KEY, MAX_RETRIES, BASE_DELAY, INTER_DELAY
            )

            if volt is None or mdv is None:
                print(f"  Incomplete data for {bid}, skipping.")
                continue

            print(f"  avg_voltage = {volt:.4f} V    max_ΔV = {mdv:.4f}")

            candidates[idx]["average_voltage"]  = str(round(volt, 6))
            candidates[idx]["max_delta_volume"] = str(round(mdv,  6))
            all_queried.append((cycle + 1, formula, bid, tm, volt, mdv))

        save_candidates(CANDIDATES_FILE, candidates, fieldnames)

        meas = measured_candidates(candidates)
        if meas:
            volts = [float(c["average_voltage"])  for c in meas]
            mdvs  = [float(c["max_delta_volume"]) for c in meas]
            best_volt = max(volts)
            best_mdv  = min(mdvs)
            curves["best_voltage"].append(best_volt)
            curves["best_delta_volume"].append(best_mdv)

            best_v_c = next(c for c in meas
                            if abs(float(c["average_voltage"]) - best_volt) < 1e-5)
            best_m_c = next(c for c in meas
                            if abs(float(c["max_delta_volume"]) - best_mdv) < 1e-5)
            print(f"  Best voltage : {best_volt:.4f} V  "
                  f"({best_v_c['framework_formula']})")
            print(f"  Best ΔV      : {best_mdv:.4f}    "
                  f"({best_m_c['framework_formula']})\n")

        build_nimo_csv(candidates, NIMO_CSV, FEATURE_COLS, PRACTICAL_MAX)
        history = nimo.history(
            input_file=NIMO_CSV,
            num_objectives=NUM_OBJECTIVES,
            itt=cycle,
            history_file=history,
        )
        save_checkpoint(CHECKPOINT_FILE, method, cycle + 1, curves, all_queried)

    # ── Cleanup checkpoint ─────────────────────────────────────────────────
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint removed (run complete).")

    # ── Final report ───────────────────────────────────────────────────────
    meas = measured_candidates(candidates)
    total_q = len(meas)
    print(f"\n{'=' * 68}")
    print(f"Optimization complete — {method}, {total_q} queries\n")

    # Pareto front
    points = [(-float(c["average_voltage"]), float(c["max_delta_volume"]))
              for c in meas]
    pf_idx = pareto_front(points)
    pf = [meas[i] for i in pf_idx]
    pf.sort(key=lambda c: -float(c["average_voltage"]))

    print(f"Pareto front ({len(pf)} materials):")
    print(f"  {'Formula':<22} {'battery_id':<20} {'Voltage':>7}  {'ΔV':>7}  TM")
    print("  " + "-" * 68)
    for c in pf:
        print(f"  {c['framework_formula']:<22} {c['battery_id']:<20} "
              f"{float(c['average_voltage']):>7.4f}  "
              f"{float(c['max_delta_volume']):>7.4f}  {c['primary_tm']}")

    print(f"\n{'Cyc':>4}  {'Formula':<22}  {'TM':>2}  {'Voltage':>8}  {'ΔV':>8}")
    print("  " + "-" * 52)
    for row in sorted(all_queried, key=lambda r: -r[4]):
        cyc, frm, bid, tm, volt, mdv = row
        print(f"  {cyc:>3}  {frm:<22}  {tm:>2}  {volt:>8.4f}  {mdv:>8.4f}")

    # ── Plot ───────────────────────────────────────────────────────────────
    if not curves["best_voltage"]:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: voltage convergence
    ax = axes[0]
    ax.plot(range(1, len(curves["best_voltage"]) + 1),
            curves["best_voltage"], "b-o", ms=4)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Best avg voltage (V vs Li/Li⁺)")
    ax.set_title(f"[{method}] Voltage convergence")
    ax.grid(True, alpha=0.3)

    # Panel 2: volume change convergence
    ax = axes[1]
    ax.plot(range(1, len(curves["best_delta_volume"]) + 1),
            curves["best_delta_volume"], "r-o", ms=4)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Best max ΔV (fraction)")
    ax.set_title(f"[{method}] Volume change convergence")
    ax.grid(True, alpha=0.3)

    # Panel 3: Pareto scatter coloured by TM
    ax = axes[2]
    tm_colors = {
        "Co": "royalblue", "Mn": "tomato", "Ni": "seagreen",
        "Fe": "goldenrod",  "V":  "mediumpurple", "Cr": "darkorange",
        "Ti": "teal",       "Mo": "brown",
    }
    for c in meas:
        v   = float(c["average_voltage"])
        mdv = float(c["max_delta_volume"])
        tm  = c["primary_tm"]
        col = tm_colors.get(tm, "grey")
        ax.scatter(v, mdv, color=col, alpha=0.6, s=40)

    # Highlight Pareto front
    for c in pf:
        v   = float(c["average_voltage"])
        mdv = float(c["max_delta_volume"])
        ax.scatter(v, mdv, color="black", zorder=5, s=80, marker="*")

    # Legend for TMs seen
    seen_tms = sorted({c["primary_tm"] for c in meas})
    handles  = [plt.Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=tm_colors.get(t, "grey"),
                            markersize=8, label=t) for t in seen_tms]
    handles.append(plt.Line2D([0], [0], marker="*", color="black",
                               linestyle="None", markersize=10, label="Pareto"))
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.set_xlabel("Average voltage (V vs Li/Li⁺)")
    ax.set_ylabel("Max ΔV (fraction)")
    ax.set_title(f"[{method}] Voltage vs Volume change")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{PLOT_PREFIX}_{method}_battery.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot → {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None)
    parser.add_argument("--config", default="battery_config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    method = args.method or cfg["nimo"]["default_method"]
    main(method, args.config)
