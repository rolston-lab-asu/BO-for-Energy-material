"""
Benchmark PHYSBO vs BLOX vs RE vs NTS vs AX on the antiperovskite candidate pool.

No MP API calls needed — formation energies are read directly from the candidates
CSV (pre-populated by mp_fetch_antiperovskite.py + mp_nimo_antiperovskite.py).
The oracle is a simple table look-up, so each run is fast (~seconds).

Each method is run NUM_SEEDS times with different RE seed offsets to account for
stochasticity in the seed phase and in methods like RE/NTS.

Metrics reported per method:
  - Mean / std of cycles to first find the global best (discovery cycle)
  - Mean best-so-far curve ± 1 std across seeds
  - Final best energy reached (should match global optimum for all methods eventually)

Usage:
    python mp_benchmark_antiperovskite.py [--config antiperovskite_config.yaml]
    python mp_benchmark_antiperovskite.py --seeds 10 --cycles 30
    python mp_benchmark_antiperovskite.py --methods PHYSBO BLOX RE --seeds 5

Prerequisites:
    antiperovskite_candidates.csv must have formation_energy_per_atom filled for
    ALL rows (run the NIMO loop to completion first, or run fetch + full RE sweep).
"""

import os
import sys
import csv
import copy
import argparse
import tempfile
import shutil

import yaml
import numpy as np
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


def save_candidates(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_nimo_csv(candidates, path, feature_cols):
    """Write numeric-only NIMO CSV: features + objective last."""
    header = feature_cols + ["formation_energy_per_atom"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for c in candidates:
            row = [c[col] for col in feature_cols]
            row.append(c["formation_energy_per_atom"])
            writer.writerow(row)


def read_proposals(proposals_path, candidates):
    proposed = []
    with open(proposals_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["actions"])
            proposed.append((idx, candidates[idx]))
    return proposed


# ── Single-run simulation ─────────────────────────────────────────────────────

def run_one(method, candidates_full, fieldnames, feature_cols, cfg, seed_offset,
            workdir):
    """
    Simulate one optimization run.

    candidates_full : list of dicts with ALL formation_energy_per_atom known
    workdir         : temp directory for NIMO CSV / proposals files
    seed_offset     : added to cycle index as RE seed for reproducible variation

    Returns
    -------
    best_curve : list[float]  best-so-far after each cycle
    discovery  : int | None   cycle (1-indexed) when global best first reached,
                               or None if never found within num_cycles
    """
    NUM_CYCLES     = cfg["nimo"]["num_cycles"]
    SEED_CYCLES    = cfg["nimo"]["seed_cycles"]
    NUM_OBJECTIVES = cfg["nimo"]["num_objectives"]
    NUM_PROPOSALS  = cfg["nimo"]["num_proposals"]

    nimo_csv   = os.path.join(workdir, "nimo_working.csv")
    prop_file  = os.path.join(workdir, "proposals.csv")

    # Oracle: ground-truth values keyed by material_id
    oracle = {c["material_id"]: float(c["formation_energy_per_atom"])
              for c in candidates_full
              if c["formation_energy_per_atom"] != ""}

    global_best = min(oracle.values())

    # Working copy — start with all objectives blank
    candidates = copy.deepcopy(candidates_full)
    for c in candidates:
        c["formation_energy_per_atom"] = ""

    best_curve = []
    discovery  = None

    for cycle in range(NUM_CYCLES):
        remaining = [c for c in candidates if c["formation_energy_per_atom"] == ""]
        if not remaining:
            break

        current_method = "RE" if cycle < SEED_CYCLES else method

        build_nimo_csv(candidates, nimo_csv, feature_cols)

        # RE seed varies per run so different seeds explore different starts
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
            if cand["formation_energy_per_atom"] != "":
                continue
            mid   = cand["material_id"]
            value = oracle.get(mid)
            if value is None:
                continue
            candidates[idx]["formation_energy_per_atom"] = str(round(value, 6))

        vals = [float(c["formation_energy_per_atom"])
                for c in candidates
                if c["formation_energy_per_atom"] != ""]

        if vals:
            best = min(vals)
            best_curve.append(best)
            if discovery is None and abs(best - global_best) < 1e-5:
                discovery = cycle + 1   # 1-indexed
        else:
            # No measurement yet (shouldn't happen after cycle 0)
            best_curve.append(float("nan"))

    return best_curve, discovery


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def benchmark_method(method, candidates_full, fieldnames, feature_cols, cfg,
                     num_seeds):
    """Run `method` num_seeds times; return list of (best_curve, discovery) tuples."""
    results = []
    for seed in range(num_seeds):
        with tempfile.TemporaryDirectory() as workdir:
            curve, disc = run_one(
                method, candidates_full, fieldnames, feature_cols, cfg,
                seed_offset=seed, workdir=workdir,
            )
        results.append((curve, disc))
        disc_str = str(disc) if disc is not None else "never"
        print(f"  [{method}] seed {seed + 1}/{num_seeds}  "
              f"discovery={disc_str}  "
              f"final_best={curve[-1]:.4f}" if curve else "  (no measurements)")
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def pad_curves(curves, num_cycles):
    """Pad/truncate all curves to the same length, forward-filling the last value."""
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


def plot_results(all_results, methods, cfg, output_prefix, num_seeds):
    NUM_CYCLES  = cfg["nimo"]["num_cycles"]
    SEED_CYCLES = cfg["nimo"]["seed_cycles"]

    colors = cm.tab10(np.linspace(0, 0.9, len(methods)))
    cycles = np.arange(1, NUM_CYCLES + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: best-so-far convergence curves ──────────────────────────────────
    ax = axes[0]
    for i, method in enumerate(methods):
        results   = all_results[method]
        curves    = [r[0] for r in results]
        mat       = pad_curves(curves, NUM_CYCLES)
        mean_     = np.nanmean(mat, axis=0)
        std_      = np.nanstd(mat, axis=0)

        ax.plot(cycles, mean_, label=method, color=colors[i], linewidth=2)
        ax.fill_between(cycles, mean_ - std_, mean_ + std_,
                        color=colors[i], alpha=0.15)

    ax.axvline(SEED_CYCLES, color="gray", linestyle="--", alpha=0.5,
               label="RE → AI switch")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Best formation energy (eV/atom)")
    ax.set_title(f"Antiperovskite M₃AX  —  mean ± 1σ over {num_seeds} seeds")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Right: discovery cycle bar chart ──────────────────────────────────────
    ax2 = axes[1]
    disc_means, disc_errs, bar_labels = [], [], []

    for i, method in enumerate(methods):
        discs = [r[1] for r in all_results[method] if r[1] is not None]
        if discs:
            disc_means.append(np.mean(discs))
            disc_errs.append(np.std(discs))
        else:
            disc_means.append(NUM_CYCLES + 1)   # "never found"
            disc_errs.append(0)
        bar_labels.append(method)

    x_pos = np.arange(len(methods))
    ax2.bar(x_pos, disc_means, yerr=disc_errs, color=colors[:len(methods)],
            capsize=5, alpha=0.85)
    ax2.axhline(SEED_CYCLES, color="gray", linestyle="--", alpha=0.5,
                label="End of seed phase")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bar_labels)
    ax2.set_ylabel("Cycle of first global-best discovery")
    ax2.set_title("Discovery cycle (lower = better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = f"{output_prefix}_benchmark.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved → {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results, methods, cfg):
    NUM_CYCLES = cfg["nimo"]["num_cycles"]
    print("\n" + "=" * 60)
    print(f"{'Method':<10}  {'Mean disc':>10}  {'Std disc':>9}  "
          f"{'Found%':>7}  {'Mean final best':>16}")
    print("  " + "-" * 57)
    for method in methods:
        results = all_results[method]
        discs   = [r[1] for r in results if r[1] is not None]
        finals  = [r[0][-1] for r in results if r[0]]
        found_pct = 100 * len(discs) / len(results)
        mean_disc = f"{np.mean(discs):.1f}" if discs else "—"
        std_disc  = f"{np.std(discs):.1f}"  if discs else "—"
        mean_fin  = f"{np.mean(finals):.4f}" if finals else "—"
        print(f"  {method:<10}  {mean_disc:>10}  {std_disc:>9}  "
              f"{found_pct:>6.0f}%  {mean_fin:>16}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(methods, config_path, num_seeds, num_cycles_override):
    cfg = load_config(config_path)

    if num_cycles_override is not None:
        cfg["nimo"]["num_cycles"] = num_cycles_override

    CANDIDATES_FILE = cfg["files"]["candidates"]
    FEATURE_COLS    = cfg["feature_cols"]
    PLOT_PREFIX     = cfg["files"]["plot_prefix"]

    if not os.path.exists(CANDIDATES_FILE):
        print(f"Error: {CANDIDATES_FILE} not found.\n"
              "Run mp_fetch_antiperovskite.py first, then run the NIMO loop\n"
              "to completion so all formation energies are populated.")
        sys.exit(1)

    candidates_full = load_candidates(CANDIDATES_FILE)
    fieldnames      = list(candidates_full[0].keys())

    # Check that all formation energies are known
    blank = sum(1 for c in candidates_full if c["formation_energy_per_atom"] == "")
    if blank > 0:
        print(f"Warning: {blank}/{len(candidates_full)} candidates have no formation "
              "energy. They will be skipped by the oracle (treated as unmeasurable).")

    known = [c for c in candidates_full if c["formation_energy_per_atom"] != ""]
    if not known:
        print("Error: no formation energies in candidates CSV. "
              "Run the NIMO loop to completion first.")
        sys.exit(1)

    global_best_val = min(float(c["formation_energy_per_atom"]) for c in known)
    global_best_cand = next(c for c in known
                            if abs(float(c["formation_energy_per_atom"]) - global_best_val) < 1e-5)
    print(f"Candidates      : {len(candidates_full)} total, {len(known)} with known Eform")
    print(f"Global best     : {global_best_val:.4f} eV/atom  "
          f"({global_best_cand['formula']}, {global_best_cand['material_id']})")
    print(f"Methods         : {methods}")
    print(f"Seeds per method: {num_seeds}")
    print(f"Cycles          : {cfg['nimo']['num_cycles']}\n")

    all_results = {}

    for method in methods:
        print(f"── Benchmarking {method} ──")
        all_results[method] = benchmark_method(
            method, candidates_full, fieldnames, FEATURE_COLS, cfg, num_seeds
        )

    print_summary(all_results, methods, cfg)
    plot_results(all_results, methods, cfg, PLOT_PREFIX, num_seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark NIMO methods on antiperovskite candidates (no API calls)."
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
        "--config", default="antiperovskite_config.yaml",
        help="Path to YAML config (default: antiperovskite_config.yaml)",
    )
    args = parser.parse_args()
    main(args.methods, args.config, args.seeds, args.cycles)
