"""
mp_generate_li3ab_candidates.py
================================
Generate the full Goldschmidt tolerance-factor map for Li3(A2-)(B-) antiperovskite
solid electrolytes and filter to structurally viable candidates.

No API calls required. All radii are from literature (see li3ab_config.yaml).

Outputs
-------
li3ab_map.csv          -- all 48 (A, B) pairs with tolerance factor and pass/fail
li3ab_candidates.csv   -- the 16 viable pairs (t in [0.80, 1.00]) with proxy objectives
li3ab_tolerance_map.png -- heatmap of t values coloured by stability window

Proxy objectives (both minimised, no DFT required):
  t_deviation  = |t - t_ideal|          geometric stability (smaller = better)
  b_radius_norm = b_radius / r_B_max    paddlewheel proxy (smaller B- = faster rotation)

Usage
-----
  cd examples/
  python mp_generate_li3ab_candidates.py
"""

import csv
import math
import yaml

# ── Load config ────────────────────────────────────────────────────────────────
with open("li3ab_config.yaml") as f:
    cfg = yaml.safe_load(f)

r_Li    = cfg["tolerance"]["li_radius"]
T_MIN   = cfg["tolerance"]["t_min"]
T_MAX   = cfg["tolerance"]["t_max"]
T_IDEAL = cfg["tolerance"]["t_ideal"]

anions_A = cfg["anions_A"]
anions_B = cfg["anions_B"]

B_MAX_RADIUS = max(v["radius"] for v in anions_B.values())

files = cfg["files"]

# ── Compute map ────────────────────────────────────────────────────────────────
all_rows  = []   # all 48 pairs
cand_rows = []   # viable pairs only

for a_name, a_data in anions_A.items():
    r_A  = a_data["radius"]
    e_A  = a_data["eneg"]
    denom = math.sqrt(2) * (r_A + r_Li)

    for b_name, b_data in anions_B.items():
        r_B  = b_data["radius"]
        e_B  = b_data["eneg"]
        na   = 1 if b_data["na_analogue"] else 0

        t    = (r_Li + r_B) / denom
        oct_ = r_A / r_B
        viable = T_MIN <= t <= T_MAX

        # proxy objectives (only meaningful for viable candidates)
        t_dev     = abs(t - T_IDEAL)
        b_rad_norm = r_B / B_MAX_RADIUS   # normalise so values are in (0, 1)

        formula = f"Li3({a_data['element']})({b_name})"
        comp_id  = f"Li3-{a_data['element']}-{b_name}"

        row = {
            "composition_id":   comp_id,
            "formula":          formula,
            "a_ion":            a_name,
            "b_ion":            b_name,
            "a_radius":         round(r_A,  3),
            "b_radius":         round(r_B,  3),
            "a_eneg":           round(e_A,  2),
            "b_eneg":           round(e_B,  2),
            "tolerance_factor": round(t,    4),
            "octahedral_factor":round(oct_, 4),
            "na_analogue":      na,
            "viable":           int(viable),
        }

        if viable:
            row["t_deviation"]   = round(t_dev,      4)
            row["b_radius_norm"] = round(b_rad_norm, 4)
        else:
            row["t_deviation"]   = ""
            row["b_radius_norm"] = ""

        all_rows.append(row)

        if viable:
            cand_rows.append(row)

# ── Write full map CSV ─────────────────────────────────────────────────────────
map_fields = [
    "composition_id", "formula", "a_ion", "b_ion",
    "a_radius", "b_radius", "a_eneg", "b_eneg",
    "tolerance_factor", "octahedral_factor", "na_analogue", "viable",
    "t_deviation", "b_radius_norm",
]
with open(files["full_map"], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=map_fields)
    w.writeheader()
    w.writerows(all_rows)

print(f"Wrote {len(all_rows)} rows to {files['full_map']}")

# ── Write candidates CSV (NIMO-ready) ─────────────────────────────────────────
cand_fields = [
    "composition_id", "formula", "a_ion", "b_ion",
    "a_radius", "b_radius", "a_eneg", "b_eneg",
    "tolerance_factor", "octahedral_factor", "na_analogue",
    "t_deviation", "b_radius_norm",
]
with open(files["candidates"], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cand_fields)
    w.writeheader()
    # write only the relevant columns
    for row in cand_rows:
        w.writerow({k: row[k] for k in cand_fields})

print(f"Wrote {len(cand_rows)} viable candidates to {files['candidates']}")

# ── Print summary table ────────────────────────────────────────────────────────
print()
print("=" * 72)
print("Li3(A2-)(B-) Tolerance Factor Map — Viable Candidates (t in [0.80, 1.00])")
print("=" * 72)
print(f"{'Formula':24} {'t':7} {'t_dev':7} {'oct':7} {'Na?':5} {'Proxy score'}")
print("-" * 72)
for r in sorted(cand_rows, key=lambda x: x["t_deviation"]):
    na_str = "YES" if r["na_analogue"] else "no"
    score  = 1.0 - r["t_deviation"] - 0.5 * r["b_radius_norm"]   # illustrative
    print(f"{r['formula']:24} {r['tolerance_factor']:7.4f} "
          f"{r['t_deviation']:7.4f} {r['octahedral_factor']:7.4f} "
          f"{na_str:5} {score:6.3f}")
print("=" * 72)
print(f"\nTotal: {len(all_rows)} pairs screened, {len(cand_rows)} viable")
print("\nNote: proxy objectives are geometric. Replace t_deviation and")
print("b_radius_norm with DFT formation_energy_per_atom once computed.")

# ── Heatmap ────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    a_names = list(anions_A.keys())
    b_names = list(anions_B.keys())
    t_matrix = np.zeros((len(a_names), len(b_names)))

    for i, a_name in enumerate(a_names):
        r_A   = anions_A[a_name]["radius"]
        denom = math.sqrt(2) * (r_A + r_Li)
        for j, b_name in enumerate(b_names):
            r_B = anions_B[b_name]["radius"]
            t_matrix[i, j] = (r_Li + r_B) / denom

    fig, ax = plt.subplots(figsize=(11, 4))
    cmap = plt.cm.RdYlGn
    im = ax.imshow(t_matrix, cmap=cmap, vmin=0.5, vmax=1.2, aspect="auto")

    # overlay hatch for viable window
    for i in range(len(a_names)):
        for j in range(len(b_names)):
            t_val = t_matrix[i, j]
            color  = "black" if T_MIN <= t_val <= T_MAX else "white"
            weight = "bold"  if T_MIN <= t_val <= T_MAX else "normal"
            ax.text(j, i, f"{t_val:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)

    ax.set_xticks(range(len(b_names)))
    ax.set_xticklabels(b_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(a_names)))
    ax.set_yticklabels([n.replace("-", "") for n in a_names], fontsize=10)
    ax.set_xlabel("B$^-$ anion", fontsize=11)
    ax.set_ylabel("A$^{2-}$ anion", fontsize=11)
    ax.set_title(
        "Li$_3$(A$^{2-}$)(B$^-$) Tolerance Factor Map\n"
        "Bold values: stable window $t \\in [0.80, 1.00]$",
        fontsize=11,
    )
    plt.colorbar(im, ax=ax, label="Tolerance factor $t$", shrink=0.8)
    plt.tight_layout()
    plt.savefig(files["heatmap"], dpi=150)
    print(f"\nHeatmap saved to {files['heatmap']}")

except ImportError:
    print("\nmatplotlib not available — skipping heatmap.")
