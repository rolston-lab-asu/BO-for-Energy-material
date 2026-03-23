"""
Benchmark NIMO methods (RE, PHYSBO, BLOX, NTS, AX) on the halide perovskite
candidate pool with TWO simultaneous objectives.

No MP API calls are needed — both formation_energy_per_atom and band_gap
(scissor-corrected) are read directly from perovskite_candidates.csv, which
must be fully populated before running this script.

Objectives (both minimized):
  1. formation_energy_per_atom   — thermodynamic stability (eV/atom)
  2. band_gap_dev = |band_gap_corrected - bg_target|  — proximity to SQ optimum

The oracle is a table lookup, so each run completes in seconds.

Metrics reported per method per seed:
  - best_bg_dev : minimum |band_gap - bg_target| seen so far at each cycle
  - best_fe     : minimum formation_energy_per_atom seen so far at each cycle
  - hypervolume : 2D hypervolume of the current Pareto front at each cycle
  - discovery   : cycle (1-indexed) when global-best band_gap_dev is first
                  reached (within 1e-4 eV tolerance), or None

Summary table shows mean/std discovery cycle, found%, and mean final
hypervolume per method.

Output plot (3-panel PNG) is saved to {plot_prefix}_benchmark_2obj.png.

Usage:
    python mp_benchmark_perovskite.py [--config perovskite_config.yaml]
    python mp_benchmark_perovskite.py --seeds 5 --cycles 60
    python mp_benchmark_perovskite.py --methods PHYSBO BLOX RE --seeds 3

Prerequisites:
    perovskite_candidates.csv must have formation_energy_per_atom AND band_gap
    filled for ALL rows (run mp_fetch_perovskite.py + mp_nimo_perovskite.py to
    completion, or fetch all rows via RE sweep).
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
matplotlib.use("Agg")           # headless — no display required
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


def build_nimo_csv(candidates, path, feature_cols, bg_target):
    """
    Write a numeric-only NIMO CSV with 2 objective columns last:
      col[-2] = formation_energy_per_atom
      col[-1] = band_gap_dev = |band_gap_corrected - bg_target|

    Both objective columns are blank until the oracle fills them together.
    The band_gap column in the candidates dict holds the scissor-corrected value.
    """
    header = feature_cols + ["formation_energy_per_atom", "band_gap_dev"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for c in candidates:
            row = [c[col] for col in feature_cols]
            fe  = c["formation_energy_per_atom"]
            bg  = c["band_gap"]                   # scissor-corrected
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


# ── Hypervolume ───────────────────────────────────────────────────────────────

def hypervolume_2d(points, ref):
    """
    Compute the 2-D hypervolume dominated by `points` relative to `ref`.

    Parameters
    ----------
    points : list of (fe, bg_dev)
        Objective values of measured candidates; both objectives are minimized.
    ref : (ref_fe, ref_bg_dev)
        Reference point strictly worse (larger) than all Pareto-optimal points.

    Returns
    -------
    float
        2-D hypervolume.  Returns 0.0 if no point dominates the reference.
    """
    ref_fe, ref_bg = ref
    # Keep only points that dominate the reference in both objectives
    dominated_pts = [(fe, bg) for fe, bg in points
                     if fe < ref_fe and bg < ref_bg]
    if not dominated_pts:
        return 0.0

    # Sort ascending by fe (x-axis); sweep vertically
    pts = sorted(dominated_pts, key=lambda p: p[0])
    hv, prev_y = 0.0, ref_bg
    for x, y in pts:
        if y < prev_y:
            hv += (ref_fe - x) * (prev_y - y)
            prev_y = y
    return hv


# ── Pareto helpers ────────────────────────────────────────────────────────────

def pareto_front_indices(points):
    """Return indices of non-dominated points (minimization of all objectives)."""
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

def run_one(method, candidates_full, feature_cols, cfg, bg_target,
            ref_point, global_best_bg_dev, seed_offset, workdir):
    """
    Simulate one full optimization run for a single seed.

    Parameters
    ----------
    method             : str   NIMO selection method name
    candidates_full    : list  All candidate dicts with oracle values populated
    feature_cols       : list  Column names to include as features in NIMO CSV
    cfg                : dict  Parsed YAML config
    bg_target          : float Shockley-Queisser band-gap target (eV)
    ref_point          : (float, float)  Hypervolume reference (ref_fe, ref_bg_dev)
    global_best_bg_dev : float  Best |band_gap - bg_target| in the full oracle
    seed_offset        : int   Multiplied into the RE seed for variation across runs
    workdir            : str   Temporary directory for intermediate files

    Returns
    -------
    bg_dev_curve : list[float]  best_bg_dev after each cycle
    fe_curve     : list[float]  best_fe after each cycle
    hv_curve     : list[float]  hypervolume of Pareto front after each cycle
    discovery    : int | None   1-indexed cycle when global_best_bg_dev first
                                reached (within 1e-4 eV), or None
    """
    NUM_CYCLES     = cfg["nimo"]["num_cycles"]
    SEED_CYCLES    = cfg["nimo"]["seed_cycles"]
    NUM_OBJECTIVES = cfg["nimo"]["num_objectives"]
    NUM_PROPOSALS  = cfg["nimo"]["num_proposals"]

    nimo_csv  = os.path.join(workdir, "nimo_working.csv")
    prop_file = os.path.join(workdir, "proposals.csv")

    # Oracle: keyed by material_id → (fe, band_gap_corrected)
    oracle = {}
    for c in candidates_full:
        fe = c["formation_energy_per_atom"]
        bg = c["band_gap"]
        if fe != "" and bg != "":
            oracle[c["material_id"]] = (float(fe), float(bg))

    # Working copy — start with all objectives blank
    candidates = copy.deepcopy(candidates_full)
    for c in candidates:
        c["formation_energy_per_atom"] = ""
        c["band_gap"] = ""

    bg_dev_curve = []
    fe_curve     = []
    hv_curve     = []
    discovery    = None

    for cycle in range(NUM_CYCLES):
        remaining = [c for c in candidates
                     if c["formation_energy_per_atom"] == "" or c["band_gap"] == ""]
        if not remaining:
            break

        current_method = "RE" if cycle < SEED_CYCLES else method

        build_nimo_csv(candidates, nimo_csv, feature_cols, bg_target)

        re_seed = (seed_offset * 1000 + cycle) if current_method == "RE" else None

        nimo.selection(
            method=current_method,
            input_file=nimo_csv,
            output_file=prop_file,
            num_objectives=NUM_OBJECTIVES,
            num_proposals=NUM_PROPOSALS,
            minimization=True,
            re_seed=re_seed,
            sample_mode="moderate",   # required by NTS; harmless for other methods
        )

        proposed = read_proposals(prop_file, candidates)

        for idx, cand in proposed:
            if (cand["formation_energy_per_atom"] != "" and cand["band_gap"] != ""):
                continue
            mid = cand["material_id"]
            val = oracle.get(mid)
            if val is None:
                continue
            fe_val, bg_val = val
            candidates[idx]["formation_energy_per_atom"] = str(round(fe_val, 6))
            candidates[idx]["band_gap"]                  = str(round(bg_val, 6))

        # Collect all measured points
        meas_fe  = []
        meas_bg  = []
        for c in candidates:
            if c["formation_energy_per_atom"] != "" and c["band_gap"] != "":
                meas_fe.append(float(c["formation_energy_per_atom"]))
                meas_bg.append(float(c["band_gap"]))

        if meas_fe:
            meas_bg_devs = [abs(bg - bg_target) for bg in meas_bg]
            best_fe      = min(meas_fe)
            best_bg_dev  = min(meas_bg_devs)

            points = list(zip(meas_fe, meas_bg_devs))
            hv     = hypervolume_2d(points, ref_point)

            fe_curve.append(best_fe)
            bg_dev_curve.append(best_bg_dev)
            hv_curve.append(hv)

            if discovery is None and abs(best_bg_dev - global_best_bg_dev) < 1e-4:
                discovery = cycle + 1   # 1-indexed
        else:
            fe_curve.append(float("nan"))
            bg_dev_curve.append(float("nan"))
            hv_curve.append(0.0)

    return bg_dev_curve, fe_curve, hv_curve, discovery


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def benchmark_method(method, candidates_full, feature_cols, cfg, bg_target,
                     ref_point, global_best_bg_dev, num_seeds):
    """
    Run `method` num_seeds times with different seed offsets.

    Returns
    -------
    list of (bg_dev_curve, fe_curve, hv_curve, discovery) tuples
    """
    results = []
    for seed in range(num_seeds):
        with tempfile.TemporaryDirectory() as workdir:
            bg_dev_curve, fe_curve, hv_curve, disc = run_one(
                method, candidates_full, feature_cols, cfg, bg_target,
                ref_point, global_best_bg_dev,
                seed_offset=seed, workdir=workdir,
            )
        results.append((bg_dev_curve, fe_curve, hv_curve, disc))
        disc_str = str(disc) if disc is not None else "never"
        if bg_dev_curve:
            print(f"  [{method}] seed {seed + 1}/{num_seeds}  "
                  f"discovery={disc_str}  "
                  f"final_bg_dev={bg_dev_curve[-1]:.4f}  "
                  f"final_hv={hv_curve[-1]:.4f}")
        else:
            print(f"  [{method}] seed {seed + 1}/{num_seeds}  (no measurements)")
    return results


# ── Padding helper ────────────────────────────────────────────────────────────

def pad_curves(curves, num_cycles):
    """Pad / truncate curves to the same length, forward-filling the last value."""
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


# ── Summary table ─────────────────────────────────────────────────────────────

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

def plot_results(all_results, methods, cfg, output_prefix, num_seeds, bg_target):
    NUM_CYCLES  = cfg["nimo"]["num_cycles"]
    SEED_CYCLES = cfg["nimo"]["seed_cycles"]

    colors = cm.tab10(np.linspace(0, 0.9, max(len(methods), 1)))
    cycles = np.arange(1, NUM_CYCLES + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: best_bg_dev convergence (lower = better) ─────────────────────
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
    ax1.set_ylabel(f"|band_gap − {bg_target}| (eV)")
    ax1.set_title(f"Perovskite ABX₃  —  band-gap dev  (mean ± 1σ, {num_seeds} seeds)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: hypervolume convergence (higher = better) ────────────────────
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
    ax2.set_ylabel("Hypervolume (eV²/atom)")
    ax2.set_title(f"Perovskite ABX₃  —  2D hypervolume  (mean ± 1σ, {num_seeds} seeds)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: discovery cycle bar chart ────────────────────────────────────
    ax3 = axes[2]
    disc_means, disc_errs = [], []
    for method in methods:
        discs = [r[3] for r in all_results[method] if r[3] is not None]
        if discs:
            disc_means.append(np.mean(discs))
            disc_errs.append(np.std(discs))
        else:
            disc_means.append(NUM_CYCLES + 1)   # "never found"
            disc_errs.append(0)

    x_pos = np.arange(len(methods))
    ax3.bar(x_pos, disc_means, yerr=disc_errs, color=colors[:len(methods)],
            capsize=5, alpha=0.85)
    ax3.axhline(SEED_CYCLES, color="gray", linestyle="--", alpha=0.5,
                label="End of seed phase")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods)
    ax3.set_ylabel("Cycle of first best-band-gap-dev discovery")
    ax3.set_title("Discovery cycle (lower = better)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = f"{output_prefix}_benchmark_2obj.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(methods, config_path, num_seeds, num_cycles_override):
    cfg = load_config(config_path)

    if num_cycles_override is not None:
        cfg["nimo"]["num_cycles"] = num_cycles_override

    CANDIDATES_FILE = cfg["files"]["candidates"]
    FEATURE_COLS    = cfg["feature_cols"]
    PLOT_PREFIX     = cfg["files"]["plot_prefix"]
    BG_TARGET       = float(cfg["pv"]["bg_target"])

    if not os.path.exists(CANDIDATES_FILE):
        print(f"Error: {CANDIDATES_FILE} not found.\n"
              "Run mp_fetch_perovskite.py first, then run the NIMO loop to\n"
              "completion so all formation energies and band gaps are populated.")
        sys.exit(1)

    candidates_full = load_candidates(CANDIDATES_FILE)

    # Check that both objectives are populated
    blank_fe = sum(1 for c in candidates_full
                   if c.get("formation_energy_per_atom", "") == "")
    blank_bg = sum(1 for c in candidates_full
                   if c.get("band_gap", "") == "")
    if blank_fe > 0:
        print(f"Warning: {blank_fe}/{len(candidates_full)} candidates have no "
              "formation_energy_per_atom. They will be skipped by the oracle.")
    if blank_bg > 0:
        print(f"Warning: {blank_bg}/{len(candidates_full)} candidates have no "
              "band_gap. They will be skipped by the oracle.")

    known = [c for c in candidates_full
             if c.get("formation_energy_per_atom", "") != ""
             and c.get("band_gap", "") != ""]
    if not known:
        print("Error: no measured candidates in CSV. "
              "Run the NIMO loop to completion first.")
        sys.exit(1)

    # Compute global oracle statistics
    oracle_fe      = [float(c["formation_energy_per_atom"]) for c in known]
    oracle_bg      = [float(c["band_gap"]) for c in known]
    oracle_bg_devs = [abs(bg - BG_TARGET) for bg in oracle_bg]

    global_best_fe      = min(oracle_fe)
    global_best_bg_dev  = min(oracle_bg_devs)

    # Reference point: slightly worse than the worst feasible values
    max_fe     = max(oracle_fe)
    max_bg_dev = max(oracle_bg_devs)
    ref_point  = (max_fe + 0.1, max_bg_dev + 0.1)

    best_fe_cand = next(c for c in known
                        if abs(float(c["formation_energy_per_atom"]) - global_best_fe) < 1e-5)
    best_bg_cand = next(c for c in known
                        if abs(abs(float(c["band_gap"]) - BG_TARGET) - global_best_bg_dev) < 1e-5)

    print(f"Candidates      : {len(candidates_full)} total, {len(known)} with both objectives")
    print(f"Global best Eform : {global_best_fe:.4f} eV/atom  "
          f"({best_fe_cand.get('formula', '?')}, {best_fe_cand.get('material_id', '?')})")
    print(f"Global best bg_dev: {global_best_bg_dev:.4f} eV  "
          f"(bg={float(best_bg_cand['band_gap']):.4f} eV, "
          f"target={BG_TARGET} eV, "
          f"{best_bg_cand.get('formula', '?')}, {best_bg_cand.get('material_id', '?')})")
    print(f"HV reference    : fe={ref_point[0]:.4f}, bg_dev={ref_point[1]:.4f}")
    print(f"Methods         : {methods}")
    print(f"Seeds per method: {num_seeds}")
    print(f"Cycles          : {cfg['nimo']['num_cycles']}\n")

    all_results = {}

    for method in methods:
        print(f"── Benchmarking {method} ──")
        all_results[method] = benchmark_method(
            method, candidates_full, FEATURE_COLS, cfg, BG_TARGET,
            ref_point, global_best_bg_dev, num_seeds,
        )

    print_summary(all_results, methods, cfg)
    plot_results(all_results, methods, cfg, PLOT_PREFIX, num_seeds, BG_TARGET)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark NIMO methods on perovskite candidates — 2-objective "
            "(formation energy + band gap deviation), no API calls."
        )
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["RE", "PHYSBO", "BLOX", "NTS", "AX"],
        choices=["RE", "PHYSBO", "BLOX", "NTS", "AX"],
        help="Methods to compare (default: all five)",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of independent seeds per method (default: 5)",
    )
    parser.add_argument(
        "--cycles", type=int, default=None,
        help="Override num_cycles from config",
    )
    parser.add_argument(
        "--config", default="perovskite_config.yaml",
        help="Path to YAML config (default: perovskite_config.yaml)",
    )
    args = parser.parse_args()
    main(args.methods, args.config, args.seeds, args.cycles)
