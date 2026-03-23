# BO for Energy Materials

Bayesian optimisation pipeline for automated materials exploration, built on top of [NIMO](https://github.com/NIMS-DA/nimo). This repository integrates NIMO with the [Materials Project (MP) API](https://materialsproject.org/) to run closed-loop, human-free searches for optimal energy materials across three material classes.

---

## What this project does

Each material class follows the same three-phase workflow:

1. **Fetch** — query the MP API, compute features, blank objectives → CSV
2. **Loop** — NIMO proposes candidates → oracle fills objectives → CSV updated
3. **Benchmark** — offline benchmark using a table-lookup oracle (no API calls needed)

### Active pipelines

| Phase | Problem | Config | Candidates | Pool size |
|-------|---------|--------|------------|-----------|
| 1 | Halide perovskite ABX₃ (PV band gap) | `perovskite_config.yaml` | `perovskite_candidates.csv` | 23 |
| 2 | Antiperovskite (formation energy) | `antiperovskite_config.yaml` | `antiperovskite_candidates.csv` | 21 |
| 3 | Li-intercalation battery cathode | `battery_config.yaml` | `battery_candidates.csv` | 892 |

---

## Benchmark results

### Perovskite — 23 candidates, 60 cycles, 5 seeds

| Method | Mean discovery cycle | Found % | Mean HV |
|--------|---------------------|---------|---------|
| PHYSBO | 10.4 | 100% | 2.9981 |
| BLOX   | 12.6 | 100% | 2.9981 |
| RE     | 10.6 | 100% | 2.9981 |

Best material: **CsSnI₃** (band gap = 1.20 eV, deviation 0.14 eV from 1.34 eV SQ target)

### Antiperovskite — 21 candidates, 30 cycles, 5 seeds

| Method | Mean discovery cycle | Found % |
|--------|---------------------|---------|
| PHYSBO | 8.0  | 100% |
| NTS    | 8.6  | 100% |
| BLOX   | 9.6  | 100% |
| RE     | 12.4 | 100% |
| AX     | 12.4 | 100% |

Best material: **Mn₃ZnN** (formation energy = −0.4506 eV/atom)

### Battery Li-ion — 892 candidates, 80 cycles, 5 seeds

| Method | Found % | Mean HV |
|--------|---------|---------|
| PHYSBO | 0% | 0.7944 |
| RE     | 0% | 0.7813 |
| BLOX   | 0% | 0.7664 |

At ~9% pool coverage, direct discovery is not expected. Hypervolume is the primary metric.

### Na-ion — 206 candidates, 5 seeds

| Cycles | Coverage | PHYSBO HV | RE HV |
|--------|----------|-----------|-------|
| 40 | 19% | 1.2826 | 1.2732 |
| 80 | 39% | 1.2848 | 1.2807 |

---

## Required Packages

- Python >= 3.6
- matplotlib
- numpy
- physbo >= 3.1.0
- scikit-learn
- scipy
- pyDOE3
- mp-api

---

## Setup

```bash
git clone https://github.com/Battery-Degradation-Rolston-Lab/BO-for-Energy-material.git
cd BO-for-Energy-material
pip install nimo
```

Set your Materials Project API key as an environment variable (never hardcode it):

```bash
export MP_API_KEY=your_key_here
```

---

## Acknowledgments

This project builds on [NIMO](https://github.com/NIMS-DA/nimo) (MIT License) and the following methods and libraries:

**PHYSBO** — GP-based Bayesian optimization (core engine for PHYSBO, NTS, PTR, BOMP, SLESA, COMBI):
> Motoyama et al., *Computer Physics Communications* **278**, 108405 (2022). https://doi.org/10.1016/j.cpc.2022.108405

**BLOX** — Boundless objective-free exploration (Random Forest + Stein Novelty):
> Terayama et al., *Chemical Science* **11**, 5959–5968 (2020). https://doi.org/10.1039/D0SC00982B

**PDC** — Phase diagram construction via uncertainty sampling:
> Terayama et al., *Physical Review Materials* **3**, 033802 (2019). https://doi.org/10.1103/PhysRevMaterials.3.033802
> Tamura et al., *Sci. Technol. Adv. Mater.: Methods* **2**, 153–161 (2022). https://doi.org/10.1080/27660400.2022.2076548

**AX** — Adaptive Experimentation Platform (BoTorch GP, continuous-to-discrete mapping):
> Baird, Falkowski & Sparks, arXiv:2502.06815 (2025). https://honegumi.readthedocs.io

**Materials Project** — Crystal structure and property data:
> https://materialsproject.org

---

## License

Distributed under the MIT License.
