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
git clone https://github.com/<your-repo>/nimo.git
cd nimo
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
Method      Mean disc   Std disc   Found%   Mean final HV
  --------------------------------------------------------
  RE             ~33.0        ~8.0    80%          (lower)
  PHYSBO         ~10.4        ~3.0   100%          (best)
  BLOX           ~15.0        ~5.0   100%          (mid)
  NTS            ~18.0        ~6.0   100%          (mid)
  AX             ~20.0        ~7.0   100%          (mid)
========================================================================
```

- **Mean disc**: average cycle at which the globally best material (RbSnI₃) was
  first discovered. Lower = better.
- **Found%**: fraction of seeds that found the global best within 60 cycles.
- **Mean final HV**: 2D hypervolume of the Pareto front at cycle 60. Higher = better.

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
~6 seed cycles it steers proposals toward the chemically relevant region rather
than sampling at random.

**BLOX uses a Random Forest** acquisition function — competitive on small
datasets but slower to converge than the GP on this 23-candidate pool.

**RE is the baseline** — pure random sampling. On the 23-candidate perovskite
pool it finds the best material 80% of the time within 60 cycles; BO methods
find it in 100% of seeds and substantially earlier.

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
├── mp_benchmark_perovskite.py    ← main entry point (no API needed)
├── mp_benchmark_battery.py       ← battery benchmark (no API needed)
├── mp_benchmark_antiperovskite.py← antiperovskite benchmark
├── perovskite_config.yaml        ← all hyperparameters for perovskite run
├── battery_config.yaml           ← all hyperparameters for battery run
├── perovskite_candidates.csv     ← pre-populated oracle (23 materials)
├── battery_candidates.csv        ← pre-populated oracle (892 materials)
├── antiperovskite_candidates.csv ← pre-populated oracle (21 materials)
└── agent_docs/
    ├── architecture.md           ← pipeline design and CSV contracts
    ├── science_rationale.md      ← physical justification for every decision
    ├── benchmark_design.md       ← metric definitions and seeding strategy
    └── scissor_calibration.md    ← band-gap correction derivation
```

---

## Extending to a new material system

See `agent_docs/architecture.md` section "Adding a new phase" for the exact
4-step recipe: copy a config YAML, write a fetch script, write a NIMO loop
script, write a benchmark script. The NIMO CSV contract (numeric features +
blank objective columns) and proposals CSV reading pattern must be preserved.
