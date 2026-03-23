"""
Step 1: Tolerance factor screening for Li₃(A²⁻)(B⁻) antiperovskites.

Enumerates all (A, B) pairs where:
  - A²⁻ is a divalent anion (chalcogenide): O, S, Se, Te
  - B⁻  is a monovalent non-halide anion (polyatomic or hydride)

Computes Goldschmidt-style tolerance factor:
  t = (r_A + r_B) / [√2 · (r_Li + r_B)]

Cubic antiperovskite stable window: 0.80 ≤ t ≤ 0.98
  - t > 0.98 → face-sharing octahedra (unstable)
  - t < 0.80 → octahedral tilting / orthorhombic distortion

Radii sources:
  - Shannon 1976 effective ionic radii (CN=VI) for monatomic ions
  - Jenkins et al. 1999 (Inorg. Chem. 38, 3609) thermochemical radii for polyatomic ions
  - Fang & Jena 2017 (PNAS 114, 11046) geometric radii noted where they differ

Usage:
    source /home/hithesh/project/nimo/.venv/bin/activate
    python li3ab_tolerance_screening.py
"""

import math
import csv

# ── Radii (Å) ────────────────────────────────────────────────────────────────

# Li⁺ Shannon effective ionic radius, CN=VI
R_LI = 0.76

# A-site: divalent anions (Shannon effective ionic radii, CN=VI)
A_SITE = {
    "O²⁻":  1.40,
    "S²⁻":  1.84,
    "Se²⁻": 1.98,
    "Te²⁻": 2.21,
}

# B-site: monovalent non-halide anions
# Jenkins 1999 thermochemical radii unless noted otherwise
B_SITE = {
    "H⁻":      {"r": 1.40, "source": "Shannon 1976",  "formula": "H"},
    "OH⁻":     {"r": 1.33, "source": "Shannon 1976",   "formula": "OH"},
    "NH₂⁻":    {"r": 1.28, "source": "Jenkins 1999",   "formula": "NH2"},
    "HCOO⁻":   {"r": 1.36, "source": "Jenkins 1999",   "formula": "HCOO"},
    "NO₂⁻":    {"r": 1.55, "source": "Jenkins 1999",   "formula": "NO2"},
    "OCN⁻":    {"r": 1.59, "source": "Jenkins 1999",   "formula": "OCN"},
    "CH₃COO⁻": {"r": 1.62, "source": "Jenkins 1999",   "formula": "CH3COO"},
    "CN⁻":     {"r": 1.77, "source": "Jenkins 1999",   "formula": "CN"},
    "NO₃⁻":    {"r": 1.79, "source": "Jenkins 1999",   "formula": "NO3"},
    "N₃⁻":     {"r": 1.80, "source": "Jenkins 1999",   "formula": "N3"},
    "BH₄⁻":    {"r": 2.05, "source": "Jenkins 1999",   "formula": "BH4",
                 "r_fj": 2.03, "note": "Fang & Jena 2017: 2.03 Å (geometric)"},
    "SCN⁻":    {"r": 2.13, "source": "Jenkins 1999",   "formula": "SCN"},
    "AlH₄⁻":   {"r": 2.26, "source": "Jenkins 1999",   "formula": "AlH4",
                 "r_fj": 2.66, "note": "Fang & Jena 2017: 2.66 Å (geometric)"},
    "BF₄⁻":    {"r": 2.28, "source": "Jenkins 1999",   "formula": "BF4",
                 "r_fj": 2.43, "note": "Fang & Jena 2017: 2.43 Å (geometric)"},
    "ClO₄⁻":   {"r": 2.36, "source": "Jenkins 1999",   "formula": "ClO4"},
}

# Cubic stability window (from literature, see report)
T_MIN = 0.80
T_MAX = 0.98


def tolerance_factor(r_a, r_b, r_li=R_LI):
    """
    Goldschmidt tolerance factor for Li₃AB antiperovskite (Pm-3m).

    Site assignments:
      B⁻  at corner    (1a, Wyckoff) → perovskite A-site (CN=12)
      A²⁻ at body ctr  (1b, Wyckoff) → perovskite B-site (CN=6)
      Li⁺  at face ctr  (3d, Wyckoff) → perovskite X-site (×3)

    By direct analogy with perovskite t = (r_A + r_X) / [√2(r_B + r_X)]:

        t = (r_B + r_Li) / [√2 · (r_A + r_Li)]

    where r_B = monovalent anion (corner), r_A = divalent anion (body center).
    Stable cubic phase: 0.80 ≤ t ≤ 0.98
    """
    denom = math.sqrt(2) * (r_a + r_li)
    return (r_b + r_li) / denom if denom > 0 else float("nan")


def octahedral_factor(r_li, r_a):
    """
    Octahedral factor μ = r_Li / r_A.
    Li⁺ sits inside the A²⁻–Li₆ octahedron (body-center site).
    Stable octahedral coordination: 0.414 ≤ μ ≤ 0.732
    """
    return r_li / r_a if r_a > 0 else float("nan")


# ── Known experimental references for validation ─────────────────────────────

HALIDE_REFS = {
    "Li₃OCl":  {"r_a": 1.40, "r_b": 1.81, "status": "synthesized (disputed σ)"},
    "Li₃OBr":  {"r_a": 1.40, "r_b": 1.96, "status": "synthesized (disputed σ)"},
    "Na₃ONO₂": {"r_a": 1.40, "r_b": 1.55, "r_cation": 1.02,
                 "status": "synthesized, paddlewheel confirmed"},
    "Na₃OCN":  {"r_a": 1.40, "r_b": 1.77, "r_cation": 1.02,
                 "status": "synthesized, conductivity jumps"},
}


# ── Compute ──────────────────────────────────────────────────────────────────

print("=" * 90)
print("Li₃(A²⁻)(B⁻) Antiperovskite Tolerance Factor Screening")
print(f"Cubic stability window: {T_MIN} ≤ t ≤ {T_MAX}")
print(f"Li⁺ radius (CN=VI): {R_LI} Å")
print("=" * 90)

# ── Validation against known halide systems ──────────────────────────────────

print("\n── Validation: known systems ──\n")
print(f"{'System':<14} {'r_A (Å)':>8} {'r_B (Å)':>8} {'t':>7} {'μ':>7} {'Window?':>8}  Status")
print("-" * 85)

for name, ref in HALIDE_REFS.items():
    r_cation = ref.get("r_cation", R_LI)
    t = tolerance_factor(ref["r_a"], ref["r_b"], r_cation)
    mu = octahedral_factor(r_cation, ref["r_a"])
    in_window = "YES" if T_MIN <= t <= T_MAX else "no"
    print(f"{name:<14} {ref['r_a']:>8.2f} {ref['r_b']:>8.2f} {t:>7.4f} {mu:>7.4f} {in_window:>8}  {ref['status']}")


# ── Full (A, B) screening ────────────────────────────────────────────────────

print("\n── Full (A²⁻, B⁻) tolerance factor map ──\n")

# Header
b_names = list(B_SITE.keys())
header = f"{'A-site':<8}" + "".join(f"{b:>10}" for b in b_names)
print(header)
print("-" * len(header))

results = []

for a_name, r_a in A_SITE.items():
    row_str = f"{a_name:<8}"
    for b_name in b_names:
        r_b = B_SITE[b_name]["r"]
        t = tolerance_factor(r_a, r_b)
        mu = octahedral_factor(R_LI, r_a)

        # Mark cells in the cubic window
        if T_MIN <= t <= T_MAX:
            marker = f"*{t:.3f}*"
        elif t > T_MAX:
            marker = f" {t:.3f}↑"
        else:
            marker = f" {t:.3f}↓"

        row_str += f"{marker:>10}"

        results.append({
            "A_site": a_name,
            "B_site": b_name,
            "r_A": r_a,
            "r_B": r_b,
            "r_B_source": B_SITE[b_name]["source"],
            "tolerance_factor": round(t, 4),
            "octahedral_factor": round(mu, 4),
            "in_cubic_window": T_MIN <= t <= T_MAX,
            "formula": f"Li₃({a_name.replace('²⁻','')})({{B_SITE[b_name]['formula']}})",
        })

    print(row_str)

print(f"\n  * = in cubic window [{T_MIN}–{T_MAX}]")
print(f"  ↑ = above window (face-sharing risk)")
print(f"  ↓ = below window (tilting/distortion)")


# ── Candidates in the cubic window ───────────────────────────────────────────

viable = [r for r in results if r["in_cubic_window"]]
non_viable_high = [r for r in results if r["tolerance_factor"] > T_MAX]
non_viable_low = [r for r in results if r["tolerance_factor"] < T_MIN]

print(f"\n── Candidates INSIDE cubic window ({len(viable)} of {len(results)}) ──\n")
print(f"{'A-site':<8} {'B-site':<12} {'r_A (Å)':>8} {'r_B (Å)':>8} {'t':>8} {'μ':>8}  Source")
print("-" * 70)

for r in sorted(viable, key=lambda x: x["tolerance_factor"]):
    print(f"{r['A_site']:<8} {r['B_site']:<12} {r['r_A']:>8.2f} {r['r_B']:>8.2f} "
          f"{r['tolerance_factor']:>8.4f} {r['octahedral_factor']:>8.4f}  {r['r_B_source']}")


# ── Near-misses (within 0.03 of window edge) ────────────────────────────────

print(f"\n── Near-misses (within 0.03 of window boundary) ──\n")

near = [r for r in results
        if not r["in_cubic_window"]
        and (abs(r["tolerance_factor"] - T_MIN) <= 0.03
             or abs(r["tolerance_factor"] - T_MAX) <= 0.03)]

if near:
    print(f"{'A-site':<8} {'B-site':<12} {'t':>8} {'gap':>8}  Note")
    print("-" * 55)
    for r in sorted(near, key=lambda x: x["tolerance_factor"]):
        if r["tolerance_factor"] < T_MIN:
            gap = T_MIN - r["tolerance_factor"]
            note = f"{gap:.3f} below {T_MIN}"
        else:
            gap = r["tolerance_factor"] - T_MAX
            note = f"{gap:.3f} above {T_MAX}"
        print(f"{r['A_site']:<8} {r['B_site']:<12} {r['tolerance_factor']:>8.4f} {gap:>8.3f}  {note}")
else:
    print("  (none)")


# ── Fang & Jena radii comparison for BH₄⁻, BF₄⁻, AlH₄⁻ ────────────────────

print("\n── Sensitivity: Jenkins vs Fang & Jena radii ──\n")
print(f"{'A-site':<8} {'B-site':<10} {'t(Jenkins)':>11} {'t(F&J)':>11} {'Δt':>7}  Jenkins? F&J?")
print("-" * 72)

for a_name, r_a in A_SITE.items():
    for b_name, b_info in B_SITE.items():
        if "r_fj" not in b_info:
            continue
        t_j = tolerance_factor(r_a, b_info["r"])
        t_fj = tolerance_factor(r_a, b_info["r_fj"])
        in_j = "YES" if T_MIN <= t_j <= T_MAX else "no"
        in_fj = "YES" if T_MIN <= t_fj <= T_MAX else "no"
        dt = t_fj - t_j
        print(f"{a_name:<8} {b_name:<10} {t_j:>11.4f} {t_fj:>11.4f} {dt:>+7.4f}  {in_j:<8} {in_fj}")


# ── Write results CSV ────────────────────────────────────────────────────────

OUT_CSV = "li3ab_tolerance_map.csv"
fieldnames = ["A_site", "B_site", "r_A", "r_B", "r_B_source",
              "tolerance_factor", "octahedral_factor", "in_cubic_window"]

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        row = {k: r[k] for k in fieldnames}
        writer.writerow(row)

print(f"\nFull results written to {OUT_CSV}")


# ── Summary ──────────────────────────────────────────────────────────────────

print("\n── Summary ──")
print(f"  Total (A, B) pairs screened: {len(results)}")
print(f"  In cubic window:             {len(viable)}")
print(f"  Above window (t > {T_MAX}):     {len(non_viable_high)}")
print(f"  Below window (t < {T_MIN}):     {len(non_viable_low)}")
print(f"  Near-misses (±0.03):         {len(near)}")

if viable:
    print(f"\n  Viable B-site anions: {sorted(set(r['B_site'] for r in viable))}")
    print(f"  Viable A-site anions: {sorted(set(r['A_site'] for r in viable))}")
