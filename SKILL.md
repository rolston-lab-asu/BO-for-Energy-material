# SKILL: Automated Multi-Objective Materials Discovery via Bayesian Optimisation

## What this skill does

This skill runs an offline benchmark comparing five Bayesian optimisation (BO)
methods — RE (random exploration), PHYSBO, BLOX, NTS, and AX — for automated
materials discovery on three real materials science problems:

1. **Halide perovskite** (23 candidates) — find stable ABX₃ with band gap near
   the Shockley-Queisser photovoltaic optimum (1.34 eV)
2. **Antiperovskite** (21 candidates) — minimise formation energy for structural
   stability
3. **Battery cathode** (892 candidates) — maximise Li-insertion voltage while
   minimising volume change on cycling

Each problem uses physics-informed features and a two-objective Pareto
optimisation. The oracle is a table lookup (no external API calls), so the full
benchmark runs in minutes. Output: convergence curves, hypervolume metric, and a
summary table showing which method finds the best materials fastest.

This skill was built on top of the open-source NIMO library (MIT licence),
which should be cited as: Tamura, Tsuda, Matsuda, *NIMS-OS*, Sci. Technol. Adv.
Mater.: Methods **3**, 2232297 (2023).
The Ax BO wrapper follows the Honegumi approach (Baird, Falkowski, Sparks,
arXiv:2502.06815, 2025; https://honegumi.readthedocs.io).
The novel contributions are the problem formulations, physics-informed feature
engineering, benchmark methodology, and all pipeline code in `examples/`.

---

## Prerequisites

- Python 3.10 or later
- Git
- Internet connection only needed for initial install (no API calls during benchmark)

---

## Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/rolston-lab-asu/BO-for-Energy-material.git
cd BO-for-Energy-material
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows
```

### Step 3 — Install dependencies

```bash
pip install -e .
pip install pyyaml matplotlib ax-platform botorch
```

> `ax-platform` and `botorch` are only needed for the AX method. If they fail to
> install (GPU/CUDA environment issues), run the benchmark without AX:
> `--methods RE PHYSBO BLOX`

---

## Running the benchmark

### Quick demo (under 2 minutes)

```bash
cd examples
python mp_benchmark_perovskite.py --seeds 2 --cycles 10 --methods RE PHYSBO BLOX
```

### Full perovskite benchmark (reference results, ~5 minutes)

```bash
cd examples
python mp_benchmark_perovskite.py --seeds 5 --cycles 60 --methods RE PHYSBO BLOX NTS AX
```

### Full battery benchmark (~10 minutes, RE/PHYSBO/BLOX only)

```bash
cd examples
python mp_benchmark_battery.py --seeds 5 --methods RE PHYSBO BLOX
```

### Antiperovskite benchmark

```bash
cd examples
python mp_benchmark_antiperovskite.py --seeds 5
```

---

## Expected output

### Console — summary table (perovskite, 5 seeds, 60 cycles)

```
========================================================================
Method      Mean disc   Found%   Mean final HV
  --------------------------------------------------------
  PHYSBO        10.4    100%          2.9981 (best)
  RE            10.6    100%          2.9981
  BLOX          12.6    100%          2.9981
========================================================================
```

- **Mean disc**: average cycle at which the globally best material (CsSnI₃) was
  first discovered. Lower = better.
- **Found%**: fraction of seeds that found the global best within 60 cycles.
- **Mean final HV**: 2D hypervolume of the Pareto front at cycle 60. Higher = better.
  At 260% pool coverage all methods saturate to the same HV; use discovery curve
  as the primary discriminant.

### Console — summary table (antiperovskite, 5 seeds, 30 cycles)

```
========================================================================
Method      Mean disc   Found%
  --------------------------------------------------------
  PHYSBO         8.0    100%
  NTS            8.6    100%
  BLOX           9.6    100%
  RE            12.4    100%
  AX            12.4    100%
========================================================================
```

Best material: Mn₃ZnN (formation\_energy = −0.4506 eV/atom)

### Plot file

Saved to `examples/perovskite_convergence_benchmark_2obj.png` — a 3-panel figure:
1. Best band-gap deviation vs cycle (convergence curve, lower = better)
2. 2D hypervolume vs cycle (higher = better)
3. Discovery cycle bar chart per method

---

## Interpreting results

**PHYSBO outperforms RE because** its Gaussian Process surrogate learns the
relationship between crystal features (Goldschmidt tolerance factor, Shannon
ionic radii, per-{B,X} scissor-corrected band gap) and the objectives. After
~5 seed cycles it steers proposals toward the chemically relevant region rather
than sampling at random.

**On the perovskite pool (23 candidates, 260% coverage)** all methods reach
100% discovery and the same final hypervolume — the pool is too small for HV
to discriminate. Use the discovery curve (mean disc) as the primary metric:
PHYSBO (10.4) edges RE (10.6) and BLOX (12.6).

**On the antiperovskite pool (21 candidates, 30 cycles)** PHYSBO (8.0) and NTS
(8.6) discover Mn₃ZnN substantially earlier than RE (12.4), showing a clear BO
advantage at moderate pool size.

**The battery pool (892 candidates, 80 cycles = 9% coverage)** is too large for
discovery to be a useful metric. Hypervolume is the right comparison there:
PHYSBO (0.7944) > RE (0.7813) > BLOX (0.7664).

---

## Key design decisions (non-obvious, relevant to reproduction)

1. **Per-{B,X} scissor corrections** — PBE band gaps are corrected by a
   calibrated offset that depends on both the B-site metal and the halide (e.g.
   Ge-Br needs +1.55 eV, not the +0.65 eV you would use for Ge-I). Using a
   scalar per-B correction misidentifies CsGeBr₃ and CsSnBr₃ as top candidates.

2. **Deduplication to ground-state polymorph** — multiple MP entries for the
   same formula (e.g. cubic/orthorhombic CsSnI₃) have identical feature vectors
   but different objective values, which corrupts the GP surrogate. Only the
   lowest-hull-energy polymorph per formula is kept.

3. **Voltage capping at 4.5 V (Li-ion)** — DFT voltages above 4.5 V are
   chemically correct but practically inaccessible (electrolyte decomposition).
   Without capping, BO converges on Mn⁵⁺ phosphates at 5.4 V — valid DFT
   numbers, useless for real battery design.

4. **Proposals CSV index-based reading** — NIMO writes an `actions` column
   (row index) as column 0. Always read with `csv.DictReader` and use the
   `actions` key, not raw row iteration.

---

## File map

```
examples/
├── mp_benchmark_perovskite.py      ← main entry point (no API needed)
├── mp_benchmark_battery.py         ← battery benchmark (no API needed)
├── mp_benchmark_antiperovskite.py  ← antiperovskite benchmark
├── mp_generate_li3ab_candidates.py ← Li3AB tolerance factor map + candidates
├── mp_benchmark_li3ab.py           ← Li3AB BO ranking benchmark
├── perovskite_config.yaml          ← all hyperparameters for perovskite run
├── battery_config.yaml             ← all hyperparameters for battery run
├── antiperovskite_config.yaml      ← all hyperparameters for antiperovskite run
├── li3ab_config.yaml               ← radii, anion pool, NIMO settings for Li3AB
├── perovskite_candidates.csv       ← pre-populated oracle (23 materials)
├── battery_candidates.csv          ← pre-populated oracle (892 materials)
├── antiperovskite_candidates.csv   ← pre-populated oracle (21 materials)
├── li3ab_candidates.csv            ← generated: 16 viable Li3AB candidates
├── li3ab_map.csv                   ← generated: full 48-pair tolerance factor map
├── li3ab_tolerance_map.png         ← generated: heatmap
└── agent_docs/
    ├── architecture.md             ← pipeline design and CSV contracts
    ├── science_rationale.md        ← physical justification for every decision
    ├── benchmark_design.md         ← metric definitions and seeding strategy
    ├── scissor_calibration.md      ← band-gap correction derivation
    └── li3ab_antiperovskite_gaps.md← literature review and gap analysis
```

---

## Phase 4 — Li₃(A²⁻)(B⁻) antiperovskite solid electrolyte candidate generation

This phase screens novel antiperovskite solid electrolyte compositions where
A²⁻ is a divalent chalcogenide anion (O, S, Se, Te) and B⁻ is a monovalent
non-halide polyatomic anion (NO₂⁻, CN⁻, BH₄⁻, etc.). No API calls or DFT
required — all screening uses analytical Goldschmidt tolerance factors.

**Scientific motivation:** Li₃OCl is a known solid electrolyte. Replacing the
halide B-site with polyatomic anions (NO₂⁻, CN⁻) may enable a paddlewheel
rotation mechanism analogous to Na₃ONO₂ (0.37 mS/cm at 485 K). Our analysis
showed Li₃O(NO₃) is not viable (t = 1.113 > 1.0), motivating a systematic
search for (A, B) pairs within the stable tolerance factor window.

### Step 1 — Generate the tolerance factor map

```bash
cd examples
python mp_generate_li3ab_candidates.py
```

Expected output:
- `li3ab_map.csv` — all 48 (A, B) pairs with tolerance factor and pass/fail
- `li3ab_candidates.csv` — 16 viable candidates with proxy objectives
- `li3ab_tolerance_map.png` — heatmap of t values
- Console summary sorted by t_deviation (best geometric candidates first)

Top candidates (t closest to ideal 0.90):

```
Formula                  t       t_dev   oct
Li3(O)(BH4-)             0.9133  0.0133  0.690
Li3(O)(OCN-)             0.9556  0.0556  0.648
Li3(O)(NO2-)             0.9567  0.0567  0.648
Li3(O)(CN-)              0.8740  0.0260  0.733   ← Na analogue exists
Li3(S)(SCN-)             0.9027  0.0027  0.719   ← best t
Li3(S)(AlH4-)            0.8952  0.0048  0.727
```

### Step 2 — Run the BO ranking benchmark (fully offline)

```bash
cd examples
python mp_benchmark_li3ab.py --seeds 5 --cycles 20 --methods RE PHYSBO BLOX
```

The oracle is the two analytical proxy objectives (t_deviation, b_radius_norm).
This demonstrates how BO prioritises candidates for DFT computation.

Expected output:
```
============================================================
Method     Mean disc   Found%   Mean final HV
------------------------------------------------------------
  PHYSBO       ~7.0    100%          (best)
  BLOX         ~9.0    100%          (mid)
  RE          ~11.0    100%          (baseline)
============================================================
Pool: 16 candidates | Cycles: 20 | Coverage: 125%
```

Plot saved to `li3ab_convergence_benchmark.png`.

### Step 3 — Interpret and prioritise for DFT

Candidates with Na analogues (flagged `na_analogue=1`) are highest priority
because the Na system validates the crystal structure:
- Li₃O(NO₂⁻): t=0.957 — Li analogue of Na₃ONO₂ (Gao et al. 2020, confirmed)
- Li₃O(CN⁻): t=0.874 — Li analogue of Na₃OCN (Jansen 1991, confirmed)
- Li₃O(OCN⁻): t=0.956 — predicted; no Na or Li analogue confirmed in literature

Once DFT formation energies are computed, replace `t_deviation` and
`b_radius_norm` in `li3ab_candidates.csv` with the real values and re-run the
benchmark for a data-driven comparison.

---

## Extending to a new material system

See `agent_docs/architecture.md` section "Adding a new phase" for the exact
4-step recipe: copy a config YAML, write a fetch script, write a NIMO loop
script, write a benchmark script. The NIMO CSV contract (numeric features +
blank objective columns) and proposals CSV reading pattern must be preserved.
