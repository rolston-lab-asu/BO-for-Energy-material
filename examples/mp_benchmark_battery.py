"""
Benchmark NIMO methods (RE, PHYSBO, BLOX, NTS, AX) on the Li-intercalation
battery candidate pool with TWO simultaneous objectives.

No MP API calls — both objectives are read directly from battery_candidates.csv,
which must be fully populated before running.

Objectives (both minimized by NIMO):
  1. neg_avg_voltage  = -average_voltage   → maximise voltage
  2. max_delta_volume                       → minimise volume change

Oracle is a table lookup, so runs complete quickly.

Metrics reported per method per seed:
  - best_voltage      : maximum average_voltage seen so far at each cycle
  - best_delta_volume : minimum max_delta_volume seen so far at each cycle
  - hypervolume       : 2D hypervolume of the current Pareto front
  - discovery         : cycle when global_best_voltage first reached
                        (within 1e-4 V tolerance), or None

Summary table shows mean/std discovery cycle, found%, and mean final
hypervolume per method.

Usage:
    python mp_benchmark_battery.py [--config battery_config.yaml]
    python mp_benchmark_battery.py --seeds 5 --cycles 80
    python mp_benchmark_battery.py --methods PHYSBO BLOX RE --seeds 3

Prerequisites:
    battery_candidates.csv must have average_voltage AND max_delta_volume
    filled for ALL rows.  Run:
        python mp_bulk_fetch_battery.py --config battery_config.yaml
    to populate all values in a single batch query before benchmarking.
"""

import os
import sys
import csv
import copy
import argparse
import tempfile

import yaml
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import nimo


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_candidates(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def build_nimo_csv(candidates, path, feature_cols, practical_max):
    """
    Write NIMO CSV with 2 objective columns last:
      col[-2] = neg_practical_voltage = -min(voltage, practical_max)
      col[-1] = max_delta_volume
    Capped at practical_max so BO optimises within the electrolyte window.
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


def read_proposals(proposals_path, candidates):
    proposed = []
    with open(proposals_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["actions"])
            proposed.append((idx, candidates[idx]))
    return proposed


# ── Hypervolume ───────────────────────────────────────────────────────────────

def hypervolume_2d(points, ref):
    """
    2-D hypervolume dominated by `points` w.r.t. `ref`.

    Both objectives are minimized (neg_volt, delta_vol).
    ref must be strictly worse (larger) than all Pareto-optimal points.
    """
    ref_v, ref_m = ref
    dominated = [(v, m) for v, m in points if v < ref_v and m < ref_m]
    if not dominated:
        return 0.0
    pts = sorted(dominated, key=lambda p: p[0])
    hv, prev_y = 0.0, ref_m
    for x, y in pts:
        if y < prev_y:
            hv += (ref_v - x) * (prev_y - y)
            prev_y = y
    return hv


# ── Pareto helpers ────────────────────────────────────────────────────────────

def pareto_front_indices(points):
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


# ── Single-run simulation ─────────────────────────────────────────────────────

def run_one(method, candidates_full, feature_cols, cfg,
            ref_point, global_best_voltage, practical_max, seed_offset, workdir):
    """
    Simulate one optimization run for a single seed.

    Returns
    -------
    volt_curve   : list[float]  best_voltage after each cycle
    mdv_curve    : list[float]  best_delta_volume after each cycle
    hv_curve     : list[float]  hypervolume of Pareto front after each cycle
    discovery    : int | None   1-indexed cycle when global_best_voltage first
                                reached (within 1e-4 V), or None
    """
    NUM_CYCLES     = cfg["nimo"]["num_cycles"]
    SEED_CYCLES    = cfg["nimo"]["seed_cycles"]
    NUM_OBJECTIVES = cfg["nimo"]["num_objectives"]
    NUM_PROPOSALS  = cfg["nimo"]["num_proposals"]

    nimo_csv  = os.path.join(workdir, "nimo_working.csv")
    prop_file = os.path.join(workdir, "proposals.csv")

    # Oracle: keyed by battery_id → (average_voltage, max_delta_volume)
    oracle = {}
    for c in candidates_full:
        volt = c["average_voltage"]
        mdv  = c["max_delta_volume"]
        if volt != "" and mdv != "":
            oracle[c["battery_id"]] = (float(volt), float(mdv))

    # Working copy — start with all objectives blank
    candidates = copy.deepcopy(candidates_full)
    for c in candidates:
        c["average_voltage"]  = ""
        c["max_delta_volume"] = ""

    volt_curve = []
    mdv_curve  = []
    hv_curve   = []
    discovery  = None

    for cycle in range(NUM_CYCLES):
        remaining = [c for c in candidates
                     if c["average_voltage"] == "" or c["max_delta_volume"] == ""]
        if not remaining:
            break

        current_method = "RE" if cycle < SEED_CYCLES else method

        build_nimo_csv(candidates, nimo_csv, feature_cols, practical_max)

        re_seed = (seed_offset * 1000 + cycle) if current_method == "RE" else None

        nimo.selection(
            method=current_method,
            input_file=nimo_csv,
            output_file=prop_file,
            num_objectives=NUM_OBJECTIVES,
            num_proposals=NUM_PROPOSALS,
            minimization=True,
            re_seed=re_seed,
            sample_mode="moderate",
        )

        proposed = read_proposals(prop_file, candidates)

        for idx, cand in proposed:
            if cand["average_voltage"] != "" and cand["max_delta_volume"] != "":
                continue
            val = oracle.get(cand["battery_id"])
            if val is None:
                continue
            volt_val, mdv_val = val
            candidates[idx]["average_voltage"]  = str(round(volt_val, 6))
            candidates[idx]["max_delta_volume"] = str(round(mdv_val,  6))

        # Collect measured points
        meas = [(float(c["average_voltage"]), float(c["max_delta_volume"]))
                for c in candidates
                if c["average_voltage"] != "" and c["max_delta_volume"] != ""]

        if meas:
            volts = [v for v, _ in meas]
            mdvs  = [m for _, m in meas]
            # Use capped voltages consistently — matches what NIMO optimises
            best_v   = min(max(volts), practical_max)
            best_mdv = min(mdvs)

            pts = [(-min(v, practical_max), m) for v, m in meas]
            hv  = hypervolume_2d(pts, ref_point)

            volt_curve.append(best_v)
            mdv_curve.append(best_mdv)
            hv_curve.append(hv)

            if discovery is None and abs(best_v - global_best_voltage) < 1e-4:
                discovery = cycle + 1
        else:
            volt_curve.append(float("nan"))
            mdv_curve.append(float("nan"))
            hv_curve.append(0.0)

    return volt_curve, mdv_curve, hv_curve, discovery


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def benchmark_method(method, candidates_full, feature_cols, cfg,
                     ref_point, global_best_voltage, practical_max, num_seeds):
    results = []
    for seed in range(num_seeds):
        with tempfile.TemporaryDirectory() as workdir:
            vc, mc, hvc, disc = run_one(
                method, candidates_full, feature_cols, cfg,
                ref_point, global_best_voltage, practical_max,
                seed_offset=seed, workdir=workdir,
            )
        results.append((vc, mc, hvc, disc))
        disc_str = str(disc) if disc is not None else "never"
        if vc:
            print(f"  [{method}] seed {seed + 1}/{num_seeds}  "
                  f"discovery={disc_str}  "
                  f"final_volt={vc[-1]:.4f} V  "
                  f"final_hv={hvc[-1]:.4f}")
        else:
            print(f"  [{method}] seed {seed + 1}/{num_seeds}  (no measurements)")
    return results


# ── Padding helper ────────────────────────────────────────────────────────────

def pad_curves(curves, num_cycles):
    out = []
    for c in curves:
        if not c:
            out.append([float("nan")] * num_cycles)
            continue
        padded = list(c)
        while len(padded) < num_cycles:
            padded.append(padded[-1])
        out.append(padded[:num_cycles])
    return np.array(out)


# ── Statistical significance ──────────────────────────────────────────────────

def significance_vs_re(all_results, methods):
    """
    Two-sided Mann-Whitney U test comparing final-HV of each method vs RE.
    Unpaired test: each seed is an independent run.
    """
    re_hvs = [r[2][-1] for r in all_results.get("RE", []) if r[2]]
    pvals  = {}
    for method in methods:
        if method == "RE":
            pvals[method] = (float("nan"), float("nan"))
            continue
        hvs = [r[2][-1] for r in all_results[method] if r[2]]
        if len(hvs) < 2 or len(re_hvs) < 2:
            pvals[method] = (float("nan"), float("nan"))
            continue
        stat, p = stats.mannwhitneyu(hvs, re_hvs, alternative="two-sided")
        pvals[method] = (stat, p)
    return pvals


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results, methods, cfg):
    NUM_CYCLES = cfg["nimo"]["num_cycles"]
    pvals = significance_vs_re(all_results, methods)

    print("\n" + "=" * 88)
    print(f"{'Method':<10}  {'Mean disc':>10}  {'Std disc':>9}  "
          f"{'Found%':>7}  {'Mean HV':>10}  {'p vs RE':>9}")
    print("  " + "-" * 72)
    for method in methods:
        results   = all_results[method]
        discs     = [r[3] for r in results if r[3] is not None]
        hv_finals = [r[2][-1] for r in results if r[2]]
        found_pct = 100.0 * len(discs) / max(len(results), 1)
        mean_disc = f"{np.mean(discs):.1f}" if discs else "—"
        std_disc  = f"{np.std(discs):.1f}"  if discs else "—"
        mean_hv   = f"{np.mean(hv_finals):.4f}" if hv_finals else "—"
        _, p      = pvals.get(method, (float("nan"), float("nan")))
        p_str     = f"{p:.3f}" if not np.isnan(p) else "—"
        print(f"  {method:<10}  {mean_disc:>10}  {std_disc:>9}  "
              f"{found_pct:>6.0f}%  {mean_hv:>10}  {p_str:>9}")
    print("=" * 88)
    print("  p vs RE: two-sided Mann-Whitney U test on final-cycle HV across seeds.")
    print("  p < 0.05 → statistically significant difference from random search.")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(all_results, methods, cfg, output_prefix, num_seeds):
    NUM_CYCLES  = cfg["nimo"]["num_cycles"]
    SEED_CYCLES = cfg["nimo"]["seed_cycles"]

    colors = cm.tab10(np.linspace(0, 0.9, max(len(methods), 1)))
    cycles = np.arange(1, NUM_CYCLES + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: best voltage convergence
    ax1 = axes[0]
    for i, method in enumerate(methods):
        curves = [r[0] for r in all_results[method]]
        mat    = pad_curves(curves, NUM_CYCLES)
        mean_  = np.nanmean(mat, axis=0)
        std_   = np.nanstd(mat, axis=0)
        ax1.plot(cycles, mean_, label=method, color=colors[i], linewidth=2)
        ax1.fill_between(cycles, mean_ - std_, mean_ + std_,
                         color=colors[i], alpha=0.15)
    ax1.axvline(SEED_CYCLES, color="gray", linestyle="--", alpha=0.5,
                label="RE → AI switch")
    ax1.set_xlabel("Cycle")
    ax1.set_ylabel("Best avg voltage (V vs Li/Li⁺)")
    ax1.set_title(f"Battery  —  Voltage convergence  (mean ± 1σ, {num_seeds} seeds)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: hypervolume convergence
    ax2 = axes[1]
    for i, method in enumerate(methods):
        curves = [r[2] for r in all_results[method]]
        mat    = pad_curves(curves, NUM_CYCLES)
        mean_  = np.nanmean(mat, axis=0)
        std_   = np.nanstd(mat, axis=0)
        ax2.plot(cycles, mean_, label=method, color=colors[i], linewidth=2)
        ax2.fill_between(cycles, mean_ - std_, mean_ + std_,
                         color=colors[i], alpha=0.15)
    ax2.axvline(SEED_CYCLES, color="gray", linestyle="--", alpha=0.5,
                label="RE → AI switch")
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Hypervolume (V · fraction)")
    ax2.set_title(f"Battery  —  2D hypervolume  (mean ± 1σ, {num_seeds} seeds)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: discovery cycle bar chart
    ax3 = axes[2]
    disc_means, disc_errs = [], []
    for method in methods:
        discs = [r[3] for r in all_results[method] if r[3] is not None]
        if discs:
            disc_means.append(np.mean(discs))
            disc_errs.append(np.std(discs))
        else:
            disc_means.append(NUM_CYCLES + 1)
            disc_errs.append(0)

    x_pos = np.arange(len(methods))
    ax3.bar(x_pos, disc_means, yerr=disc_errs, color=colors[:len(methods)],
            capsize=5, alpha=0.85)
    ax3.axhline(SEED_CYCLES, color="gray", linestyle="--", alpha=0.5,
                label="End of seed phase")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods)
    ax3.set_ylabel("Cycle of first best-voltage discovery")
    ax3.set_title("Discovery cycle (lower = better)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = f"{output_prefix}_benchmark_battery.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(methods, config_path, num_seeds, num_cycles_override, practical_max_override=None):
    cfg = load_config(config_path)

    if num_cycles_override is not None:
        cfg["nimo"]["num_cycles"] = num_cycles_override

    CANDIDATES_FILE = cfg["files"]["candidates"]
    FEATURE_COLS    = cfg["feature_cols"]
    PLOT_PREFIX     = cfg["files"]["plot_prefix"]
    if practical_max_override is not None:
        cfg["battery"]["voltage_practical_max"] = practical_max_override

    if not os.path.exists(CANDIDATES_FILE):
        print(f"Error: {CANDIDATES_FILE} not found. Run mp_fetch_battery.py first.")
        sys.exit(1)

    candidates_full = load_candidates(CANDIDATES_FILE)

    blank_v = sum(1 for c in candidates_full if c.get("average_voltage", "") == "")
    blank_m = sum(1 for c in candidates_full if c.get("max_delta_volume", "") == "")
    if blank_v > 0:
        print(f"Warning: {blank_v}/{len(candidates_full)} candidates have no "
              "average_voltage. They will be skipped.")
    if blank_m > 0:
        print(f"Warning: {blank_m}/{len(candidates_full)} candidates have no "
              "max_delta_volume. They will be skipped.")

    known = [c for c in candidates_full
             if c.get("average_voltage", "") != ""
             and c.get("max_delta_volume", "") != ""]
    if not known:
        print("Error: no measured candidates. Run the NIMO loop or "
              "mp_bulk_fetch_battery.py first.")
        sys.exit(1)

    PRACTICAL_MAX = cfg["battery"].get("voltage_practical_max", 5.5)

    oracle_volts = [float(c["average_voltage"])  for c in known]
    oracle_mdvs  = [float(c["max_delta_volume"]) for c in known]

    # Cap voltages at practical_max for discovery and HV — same as what NIMO sees
    practical_volts     = [min(v, PRACTICAL_MAX) for v in oracle_volts]
    global_best_voltage = max(practical_volts)
    global_best_mdv     = min(oracle_mdvs)

    best_v_cand = next(c for c in known
                       if min(float(c["average_voltage"]), PRACTICAL_MAX) >= global_best_voltage - 1e-5)
    best_m_cand = next(c for c in known
                       if abs(float(c["max_delta_volume"]) - global_best_mdv) < 1e-5)

    # Reference point for hypervolume (in capped neg_volt, mdv space)
    max_neg_volt = max(-min(float(c["average_voltage"]), PRACTICAL_MAX) for c in known)
    max_mdv      = max(float(c["max_delta_volume"]) for c in known)
    ref_point    = (max_neg_volt + 0.1, max_mdv + 0.01)

    print(f"Candidates        : {len(candidates_full)} total, {len(known)} measured")
    print(f"Practical voltage cap : {PRACTICAL_MAX} V (electrolyte stability limit)")
    print(f"Global best voltage: {global_best_voltage:.4f} V  "
          f"({best_v_cand['framework_formula']}, {best_v_cand['battery_id']})")
    print(f"Global best ΔV     : {global_best_mdv:.4f}    "
          f"({best_m_cand['framework_formula']}, {best_m_cand['battery_id']})")
    print(f"HV reference      : neg_volt={ref_point[0]:.4f}, mdv={ref_point[1]:.4f}")
    print(f"Methods           : {methods}")
    print(f"Seeds per method  : {num_seeds}")
    print(f"Cycles            : {cfg['nimo']['num_cycles']}\n")

    all_results = {}
    for method in methods:
        print(f"── Benchmarking {method} ──")
        all_results[method] = benchmark_method(
            method, candidates_full, FEATURE_COLS, cfg,
            ref_point, global_best_voltage, PRACTICAL_MAX, num_seeds,
        )

    print_summary(all_results, methods, cfg)
    plot_results(all_results, methods, cfg, PLOT_PREFIX, num_seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark NIMO methods on battery electrode candidates — "
            "2-objective (voltage + volume change), no API calls."
        )
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["RE", "PHYSBO", "BLOX", "NTS", "AX"],
        choices=["RE", "PHYSBO", "BLOX", "NTS", "AX"],
    )
    parser.add_argument("--seeds",  type=int, default=5)
    parser.add_argument("--cycles", type=int, default=None)
    parser.add_argument("--config", default="battery_config.yaml")
    parser.add_argument(
        "--practical-max", type=float, default=None,
        help="Override voltage_practical_max from config (e.g. 5.5 to remove cap)",
    )
    args = parser.parse_args()
    main(args.methods, args.config, args.seeds, args.cycles, args.practical_max)
