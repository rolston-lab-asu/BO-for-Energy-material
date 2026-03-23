# CLAUDE.md — NIMO Project

Working directory for active development: `/home/hithesh/project/nimo/examples/`

Python venv: `/home/hithesh/project/nimo/.venv`
Activate with: `source /home/hithesh/project/nimo/.venv/bin/activate`

MP API key: environment variable `MP_API_KEY` (never hardcode)

---

## What this project is

NIMO is a closed-loop Bayesian optimisation library for automated materials exploration.
The `examples/` folder contains a Materials Project (MP) integration pipeline that uses
NIMO to search for optimal materials without human intervention.

The MP pipeline has three active problems, each with its own fetch → loop → benchmark
workflow. Read `agent_docs/architecture.md` before making structural changes.

---

## Pipeline overview

| Phase | Problem | Config | Candidates | Pool size |
|-------|---------|--------|------------|-----------|
| 1 | Halide perovskite ABX₃ (PV band gap) | `perovskite_config.yaml` | `perovskite_candidates.csv` | 23 |
| 2 | Antiperovskite (formation energy) | `antiperovskite_config.yaml` | `antiperovskite_candidates.csv` | 21 |
| 3 | Li-intercalation battery cathode | `battery_config.yaml` | `battery_candidates.csv` | 892 |

Each phase follows the same 3-script pattern:
1. `mp_fetch_*.py` — query MP API, compute features, blank objectives → CSV
2. `mp_nimo_*.py` — NIMO loop: propose → oracle query → update CSV
3. `mp_benchmark_*.py` — offline benchmark (table-lookup oracle, no API calls)

Battery also has `mp_bulk_fetch_battery.py` to pre-populate all objectives in one
MP re-scan (needed before benchmarking).

---

## Key known issues / fixes already applied

- `thermo_type` from MP `insertion_electrodes` endpoint returns `None` — filter removed
- `battery_ids=[list]` API filter is broken (returns 1 result regardless of list size) —
  bulk populate uses full re-scan + local match instead
- NIMO proposals file has an `actions` column as column 0 — `read_proposals()` must use
  `csv.DictReader` and skip the `actions` key, not raw CSV row iteration
- Shannon ionic radii for Sn²⁺ (CN=VI) missing from pymatgen — manual fallback: 1.18 Å
- Perovskite scissors must be per-{B,X} pair, not per-B scalar (Ge-Br needs +1.55 eV,
  not +0.50 eV; Sn-Br needs +1.15 eV, not +0.75 eV)

---

## Benchmark results (as of 2026-03-19)

### Perovskite (23 candidates, 60 cycles, 5 seeds, RE/PHYSBO/BLOX)
| Method | Mean disc | Found% | Mean HV | p vs RE |
|--------|-----------|--------|---------|---------|
| PHYSBO | 10.4 | 100% | 2.9981 | 1.000 |
| BLOX | 12.6 | 100% | 2.9981 | 1.000 |
| RE | 10.6 | 100% | 2.9981 | — |

Best material: CsSnI₃ (bg=1.20 eV, bg_dev=0.14 eV from 1.34 eV SQ target)
Note: RbSnI₃ ground-state polymorph is orthorhombic (bg=2.76 eV after scissors) — not the global best.
Note: p=1.000 on HV expected at 260% coverage — all methods saturate. Use discovery curve, not final HV.
Note: RE disc improved from ~33 (old) to 10.6 because global best is now CsSnI₃ (common compound, easy random find).

### Antiperovskite (21 candidates, 30 cycles, 5 seeds, all methods)
| Method | Mean disc | Found% |
|--------|-----------|--------|
| PHYSBO | 8.0 | 100% |
| NTS | 8.6 | 100% |
| BLOX | 9.6 | 100% |
| RE | 12.4 | 100% |
| AX | 12.4 | 100% |

Best material: Mn₃ZnN (formation_energy = −0.4506 eV/atom)

### Battery Li-ion (892 candidates, 80 cycles, 5 seeds)
| Method | Mean disc | Found% | Mean HV | p vs RE |
|--------|-----------|--------|---------|---------|
| PHYSBO | — | 0% | 0.7944 | pending |
| RE | — | 0% | 0.7813 | — |
| BLOX | — | 0% | 0.7664 | pending |

Discovery = "never" expected (9% coverage). HV is the real metric. Wilcoxon pending.

### Na-ion (206 candidates, 5 seeds)
| Cycles | Coverage | PHYSBO HV | RE HV | p vs RE |
|--------|----------|-----------|-------|---------|
| 40 | 19% | 1.2826 | 1.2732 | 0.075 |
| 80 | 39% | 1.2848 | 1.2807 | 0.116 |

Discovery uninformative (global best at voltage cap, trivially found by all methods).
BO advantage diminishes with coverage — consistent with GP theory.

---

## Detailed context

| File | When to read |
|------|-------------|
| `agent_docs/architecture.md` | Adding new phases, changing CSV schema, cross-cutting changes |
| `agent_docs/mp_api_quirks.md` | Any Materials Project API work |
| `agent_docs/scissor_calibration.md` | Perovskite band-gap corrections |
| `agent_docs/benchmark_design.md` | Benchmark methodology, hypervolume, discovery metric |
