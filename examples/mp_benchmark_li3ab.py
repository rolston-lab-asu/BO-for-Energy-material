"""
mp_benchmark_li3ab.py
======================
Offline BO benchmark for Li3(A2-)(B-) antiperovskite candidate ranking.

Requires li3ab_candidates.csv to be present (run mp_generate_li3ab_candidates.py first).

Oracle: two analytical proxy objectives, no DFT or API calls needed.
  obj1 = t_deviation   = |tolerance_factor - 0.90|  (minimise)
  obj2 = b_radius_norm = b_radius / 2.64             (minimise, paddlewheel proxy)

Usage
-----
  cd examples/
  python mp_generate_li3ab_candidates.py   # generate candidates first
  python mp_benchmark_li3ab.py [--seeds N] [--cycles N] [--methods RE PHYSBO BLOX]

Output
------
  li3ab_convergence_benchmark.png   -- 3-panel figure (HV, obj1, discovery)
  console summary table
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--seeds",   type=int, default=5)
parser.add_argument("--cycles",  type=int, default=20)
parser.add_argument("--methods", nargs="+", default=["RE", "PHYSBO", "BLOX"])
args = parser.parse_args()

CANDIDATES_FILE = "li3ab_candidates.csv"
if not os.path.exists(CANDIDATES_FILE):
    sys.exit(
        f"ERROR: {CANDIDATES_FILE} not found.\n"
        "Run mp_generate_li3ab_candidates.py first."
    )

# ── Load candidates ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "a_radius", "b_radius", "a_eneg", "b_eneg",
    "tolerance_factor", "octahedral_factor", "na_analogue",
]
OBJ_COLS = ["t_deviation", "b_radius_norm"]

candidates = []
with open(CANDIDATES_FILE, newline="") as f:
    for row in csv.DictReader(f):
        candidates.append(row)

N = len(candidates)
print(f"Loaded {N} candidates from {CANDIDATES_FILE}")

# Build oracle arrays
oracle_obj = np.array(
    [[float(c[obj]) for obj in OBJ_COLS] for c in candidates]
)  # shape (N, 2)

feature_arr = np.array(
    [[float(c[feat]) for feat in FEATURE_COLS] for c in candidates]
)  # shape (N, 7)

# Global Pareto front (all candidates, complete information)
def pareto_front(obj_matrix):
    """Return boolean mask of non-dominated rows (minimise both objectives)."""
    n = len(obj_matrix)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.all(obj_matrix[j] <= obj_matrix[i]) and np.any(obj_matrix[j] < obj_matrix[i]):
                dominated[i] = True
                break
    return ~dominated

full_pareto_mask = pareto_front(oracle_obj)
full_pareto_obj  = oracle_obj[full_pareto_mask]
global_best_idx  = int(np.argmin(oracle_obj[:, 0]))   # best t_deviation

# Hypervolume (2D, minimisation): dominated area relative to reference point
def hypervolume_2d(points, ref):
    """Compute 2-D hypervolume for minimisation problems."""
    pts = np.array(points)
    # filter dominated by ref
    pts = pts[np.all(pts < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for p in pts:
        hv += (ref[0] - p[0]) * (prev_y - p[1])
        prev_y = p[1]
    return hv

ref_point = oracle_obj.max(axis=0) + 0.05
full_hv   = hypervolume_2d(oracle_obj[full_pareto_mask], ref_point)

print(f"Global best (min t_deviation): {candidates[global_best_idx]['formula']}  "
      f"t={candidates[global_best_idx]['tolerance_factor']}, "
      f"t_dev={oracle_obj[global_best_idx, 0]:.4f}")
print(f"Pareto front size: {full_pareto_mask.sum()}")
print(f"Full hypervolume (reference): {full_hv:.4f}\n")

# ── NIMO wrapper ───────────────────────────────────────────────────────────────
try:
    import nimo
    NIMO_AVAILABLE = True
except ImportError:
    NIMO_AVAILABLE = False
    print("WARNING: nimo not installed. Only RE will run; GP methods skipped.\n")

def write_nimo_csv(queried_mask, obj_arr, path):
    """Write NIMO working CSV: features + objectives (blank if not yet queried)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FEATURE_COLS + OBJ_COLS)
        for i in range(N):
            row_feat = list(feature_arr[i])
            if queried_mask[i]:
                row_obj = list(obj_arr[i])
            else:
                row_obj = [""] * len(OBJ_COLS)
            w.writerow(row_feat + row_obj)

def read_proposal(proposals_path):
    """Return the row index proposed by NIMO (reads actions column)."""
    with open(proposals_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return int(row["actions"])
    raise ValueError(f"No proposal found in {proposals_path}")

def run_seed(method, seed_offset, num_cycles, seed_cycles):
    """Run one seed of the BO loop. Returns (hv_curve, best_t_dev_curve, disc_cycle)."""
    queried  = np.zeros(N, dtype=bool)
    observed = np.full((N, 2), np.nan)
    hv_curve       = []
    best_t_dev     = float("inf")
    best_curve     = []
    disc_cycle     = None
    rng = np.random.default_rng(seed_offset * 1000)

    for cycle in range(num_cycles):
        unqueried = np.where(~queried)[0]
        if len(unqueried) == 0:
            # Pool exhausted — pad curves with last values
            hv_curve.append(hv_curve[-1] if hv_curve else 0.0)
            best_curve.append(best_t_dev)
            continue

        use_re = (not NIMO_AVAILABLE) or (method == "RE") or (cycle < seed_cycles)

        if use_re:
            idx = int(rng.choice(unqueried))
        else:
            # NIMO proposal
            nimo_csv = "li3ab_nimo_working.csv"
            prop_csv = "li3ab_proposals.csv"
            write_nimo_csv(queried, observed, nimo_csv)
            try:
                nimo.selection(
                    csv_file=nimo_csv,
                    output_file=prop_csv,
                    method=method,
                    num_objectives=len(OBJ_COLS),
                    num_proposals=1,
                    random_seed=seed_offset * 1000 + cycle,
                )
                idx = read_proposal(prop_csv)
                if queried[idx]:
                    unqueried = np.where(~queried)[0]
                    idx = int(rng.choice(unqueried))
            except Exception:
                unqueried = np.where(~queried)[0]
                idx = int(rng.choice(unqueried))

        # Oracle query
        queried[idx]     = True
        observed[idx, :] = oracle_obj[idx, :]

        # Update metrics
        obs_pts = observed[queried]
        pf_mask = pareto_front(obs_pts)
        hv = hypervolume_2d(obs_pts[pf_mask], ref_point)
        hv_curve.append(hv)

        curr_t_dev = oracle_obj[idx, 0]
        if curr_t_dev < best_t_dev:
            best_t_dev = curr_t_dev
        best_curve.append(best_t_dev)

        if disc_cycle is None and idx == global_best_idx:
            disc_cycle = cycle + 1   # 1-indexed

    return hv_curve, best_curve, disc_cycle

# ── Run benchmark ──────────────────────────────────────────────────────────────
SEED_CYCLES = 4
results = {}

for method in args.methods:
    hv_all, best_all, disc_all = [], [], []
    for s in range(args.seeds):
        hv_curve, best_curve, disc = run_seed(method, s, args.cycles, SEED_CYCLES)
        hv_all.append(hv_curve)
        best_all.append(best_curve)
        disc_all.append(disc if disc is not None else args.cycles + 1)
    results[method] = {
        "hv":   np.array(hv_all),
        "best": np.array(best_all),
        "disc": np.array(disc_all),
    }

# ── Print summary ──────────────────────────────────────────────────────────────
print("=" * 60)
print(f"{'Method':10} {'Mean disc':10} {'Found%':8} {'Mean final HV':14}")
print("-" * 60)
for method, res in results.items():
    found  = np.mean(res["disc"] <= args.cycles) * 100
    m_disc = np.mean(res["disc"][res["disc"] <= args.cycles]) if found > 0 else float("nan")
    m_hv   = np.mean(res["hv"][:, -1])
    disc_str = f"{m_disc:.1f}" if not math.isnan(m_disc) else "never"
    print(f"  {method:8} {disc_str:>10} {found:>7.0f}%  {m_hv:>12.4f}")
print("=" * 60)
print(f"\nPool: {N} candidates | Cycles: {args.cycles} | Seeds: {args.seeds}")
print(f"Coverage: {args.cycles}/{N} = {100*args.cycles/N:.0f}%")
print(f"HV reference: {full_hv:.4f} (full Pareto front)")

# ── Plot ───────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"RE": "gray", "PHYSBO": "royalblue", "BLOX": "darkorange",
              "NTS": "green", "AX": "purple"}
    cycles_arr = np.arange(1, args.cycles + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: best t_deviation vs cycle
    ax = axes[0]
    for m, res in results.items():
        mu  = res["best"].mean(axis=0)
        std = res["best"].std(axis=0)
        c = colors.get(m, "black")
        ax.plot(cycles_arr, mu, label=m, color=c)
        ax.fill_between(cycles_arr, mu - std, mu + std, alpha=0.15, color=c)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Best $|t - 0.90|$ (lower = better)")
    ax.set_title("Geometric stability (proxy obj 1)")
    ax.legend()

    # Panel 2: hypervolume vs cycle
    ax = axes[1]
    for m, res in results.items():
        mu  = res["hv"].mean(axis=0)
        std = res["hv"].std(axis=0)
        c = colors.get(m, "black")
        ax.plot(cycles_arr, mu, label=m, color=c)
        ax.fill_between(cycles_arr, mu - std, mu + std, alpha=0.15, color=c)
    ax.axhline(full_hv, linestyle="--", color="black", linewidth=0.8, label="Full HV")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("2D Hypervolume")
    ax.set_title("Multi-objective progress")
    ax.legend()

    # Panel 3: discovery cycle bar chart
    ax = axes[2]
    methods = list(results.keys())
    mean_disc = []
    for m in methods:
        d = results[m]["disc"]
        found = d[d <= args.cycles]
        mean_disc.append(found.mean() if len(found) > 0 else args.cycles + 1)
    bars = ax.bar(methods, mean_disc, color=[colors.get(m, "black") for m in methods])
    ax.set_ylabel("Mean discovery cycle (lower = better)")
    ax.set_title("Discovery of globally best candidate")
    ax.set_ylim(0, args.cycles + 2)
    for bar, val in zip(bars, mean_disc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        f"Li$_3$(A$^{{2-}}$)(B$^-$) BO Benchmark  "
        f"({N} candidates, {args.cycles} cycles, {args.seeds} seeds)",
        fontsize=12,
    )
    plt.tight_layout()
    out = "li3ab_convergence_benchmark.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
except ImportError:
    print("\nmatplotlib not available — skipping plot.")
