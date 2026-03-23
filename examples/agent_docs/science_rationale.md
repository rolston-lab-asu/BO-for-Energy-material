# Science Rationale — NIMO MP Pipeline

Complete record of every scientific decision made in this project: what was done,
why it was done, and what the physical basis is. Written so a future session can
understand not just what the code does but why it is correct.

---

## 1. Why cathode, not anode

**Decision:** Focus on Li/Na-ion intercalation cathodes.

**Reasoning:**
Cell voltage = V_cathode − V_anode. The anode (graphite) is fixed at ~0.1 V vs Li/Li⁺
and is already near its theoretical intercalation limit (372 mAh/g). Every 0.1 V gained
on the cathode goes directly into cell energy density. The cathode is the bottleneck.

Anode alternatives are either:
- **Graphite** — mature, not an intercalation TM material, not in insertion_electrodes
- **Silicon** — alloying/conversion mechanism, not in insertion_electrodes, ΔV ~300%
- **Li metal** — plating/stripping, dendrite problem, no DFT descriptor for this
- **LTO/TiO₂** — small known space, already well-understood

**Na-ion rationale:** CATL and BYD announced commercial Na-ion cells in 2023–2024.
Na-ion cathode space is far less explored than Li-ion — more room for BO to find
non-obvious candidates. Same pipeline, same features, different working ion.

---

## 2. Why these objectives

### Li-ion: maximise voltage + minimise ΔV

**Voltage (average_voltage):**
Energy density ∝ voltage × capacity. Higher voltage = more energy per charge.
DFT-computed by MP from the energy difference between lithiated and delithiated states.
No scissor correction needed (unlike band gaps) — formation energy differences cancel
systematic DFT errors to first order.

**Volume change (max_delta_volume):**
Fractional volume change on lithiation. Large ΔV → mechanical stress → particle
cracking → capacity fade on cycling. LiFePO₄ (commercial EV cathode) has ΔV ≈ 3%.
LiCoO₂ (laptop/phone) ΔV ≈ 2%. Materials with ΔV > 20% are generally not viable.

**Why both simultaneously (Pareto, not single objective):**
These objectives trade off. High-voltage materials (Ni-rich NMC, ~4.2 V) tend to have
larger ΔV. Stable low-ΔV materials (LiFePO₄, ~3.4 V) have lower voltage. The Pareto
front maps this trade-off. A single-objective search would find the voltage extreme
(Mn⁵⁺ phosphate at 5.5 V, unusable) or the ΔV extreme (near-zero ΔV at very low voltage).

### Perovskite: minimise formation energy + minimise |band_gap − 1.34 eV|

**Formation energy:** thermodynamic stability proxy. More negative = more stable.
**Band gap deviation from 1.34 eV:** Shockley-Queisser optimum for single-junction PV.
Both minimised simultaneously → stable material with ideal PV gap.

---

## 3. Voltage capping at practical_max

**Problem discovered:** DFT voltages have no electrolyte stability constraint.
MP returned Mn(PO₄)₂ at 5.4995 V (Mn⁵⁺/Mn⁴⁺ couple in phosphate). This is a real
DFT number but completely unusable — all liquid electrolytes decompose above ~4.5 V,
and most solid electrolytes above ~5 V.

**Effect on NIMO:** Without capping, the GP learned "high oxidation state Mn phosphate
→ high voltage" and proposed materials in the 5–5.5 V range. These are valid DFT
predictions but irrelevant to practical battery design. Discovery metric became
"did you find Mn⁵⁺ chemistry" rather than "did you find the best usable cathode."

**Fix:** Cap voltage at `voltage_practical_max` before writing to NIMO CSV:
```python
practical_volt = min(float(volt), practical_max)
```
All 168/892 Li-ion materials above 4.5 V are treated as equivalent at the cap.
NIMO now optimises within the usable window. Global best becomes Al₃Cr₃(SbO₈)₂ at 4.5 V.

**Values:**
- Li-ion: 4.5 V (conventional liquid electrolyte stability limit)
- Na-ion: 4.0 V (Na-ion electrolytes typically stable to 4.0–4.3 V)

**Bug fixed alongside this:** Discovery metric was comparing raw voltage against capped
global best (always "never"). Fixed to compare `min(best_v, practical_max)`.

---

## 4. Feature engineering

### Original 9 features — what they capture and what they miss

| Feature | Physical meaning | Limitation |
|---------|-----------------|------------|
| tm_eneg | Pauling electronegativity of TM | Bulk average, misses redox couple |
| tm_ionic_radius | Atomic radius of TM | Doesn't distinguish oxidation states |
| tm_max_oxidation_state | Max common ox state | Max, not actual discharge state |
| anion_eneg | Highest-eneg non-TM element | Picks O/F/S but ignores polyanion central atom |
| li_per_fu | Li count in discharge formula | Good capacity proxy |
| spacegroup_number | SG of Li-free framework | Integer, good structural fingerprint |
| crystal_system | Triclinic…cubic (1–7) | Coarse structural class |
| volume_per_atom | Framework vol/atom (Å³) | Good density proxy |
| density | Mass density g/cm³ | Good for GP smoothness |

**Root cause of poor BO performance:** Mn²⁺/Mn³⁺ at 3.0 V and Mn³⁺/Mn⁴⁺ at 4.0 V
look identical in these 9 features. LiFePO₄ (3.4 V) and Li₂FeSO₄ (3.9 V) look nearly
identical except anion_eneg. The GP surrogate was blind to the two main voltage drivers.

### New physics features (added 2026-03-18)

#### polyanion_eneg — Goodenough inductive effect

**Physical basis:**
Goodenough (1997) showed that covalent bonding between O²⁻ and a counter-cation
(P⁵⁺, S⁶⁺, Si⁴⁺…) in a polyanion XO₄ withdraws electron density from the TM via
the X–O–TM bridge. This raises the TM d-band energy and therefore the Li insertion
voltage. Strength scales with electronegativity of X.

**Quantification:**
```
Simple oxide (no polyanion):  baseline voltage
Silicate SiO₄⁴⁻  (Si 1.90):  +0.2 V above oxide
Phosphate PO₄³⁻  (P  2.19):  +0.5–0.7 V above oxide  → LiFePO₄ at 3.5 V
Sulfate SO₄²⁻   (S  2.58):  +0.8–1.0 V above oxide  → Li₂Fe(SO₄)₂ at 3.9 V
Fluorosulfate    (S+F 3.98): +1.0–1.2 V above oxide  → LiVPO₄F at 4.2 V
```

**Implementation:** Parse `framework_formula` for P, S, Si, As, B, Mo, W, Ge, Nb, Ta, Sb.
Return highest-stoichiometry polyanion's central atom Pauling eneg. 0.0 for simple oxides.
No API call — derived entirely from composition already in the CSV.

#### structure_prototype — framework topology

**Physical basis:**
The crystal structure topology determines:
1. Li diffusion pathway → kinetics
2. Structural rigidity on lithiation → ΔV
3. Voltage plateau shape → how flat the charge/discharge curve is

Key prototype families:
- **Olivine (SG 62, Pnma):** LiFePO₄. 1D Li channels, very rigid PO₄ framework, ΔV ~3%
- **Spinel (SG 227, Fd-3m):** LiMn₂O₄. 3D Li network, moderate rigidity, ΔV ~6%
- **Layered (SG 166, R-3m):** LiCoO₂/NMC. 2D Li planes between MO₂ sheets, shear possible, ΔV ~2–10%
- **Monoclinic layered (SG 12, C2/m):** Li₂MnO₃ family. Similar to layered
- **NASICON (SG 167, R-3c):** Na₃V₂(PO₄)₃ family. 3D open framework, excellent ΔV

**Implementation:** Dictionary lookup from spacegroup_number already in CSV. Zero cost.

#### tm_oxidation_discharge — actual redox couple

**Physical basis:**
The intercalation voltage is determined by which TM oxidation state transition occurs:
```
Mn²⁺ → Mn³⁺  :  ~3.0 V   (e.g. MnO cathodes)
Fe²⁺ → Fe³⁺  :  ~3.4 V   (LiFePO₄)
Co²⁺ → Co³⁺  :  ~3.9 V   (LiCoO₂)
Ni²⁺ → Ni³⁺  :  ~3.8 V
Ni³⁺ → Ni⁴⁺  :  ~4.6 V   (Ni-rich NMC at high charge)
Mn³⁺ → Mn⁴⁺  :  ~4.0 V   (spinel LiMn₂O₄)
V³⁺  → V⁴⁺   :  ~3.5 V
V⁴⁺  → V⁵⁺   :  ~4.5 V   (VOPO₄)
```

`tm_max_oxidation_state` (old feature) is the max *possible* oxidation state, not the
one actually occurring. Fe can reach Fe⁶⁺ but no cathode cycles Fe⁶⁺/Fe⁵⁺.
`tm_oxidation_discharge` is the actual state in the discharged structure from MP's
DFT-assigned oxidation states (`possible_species` field), which pins the actual couple.

**Implementation:** `possible_species` from MP summary endpoint → parse "Fe2+", "Mn3+",
etc. → average TM oxidation states → store as float. 831/892 filled; 61 imputed with
column mean (3.16). 25 Na-ion imputed with mean 3.25.

### Impact of new features

| Metric | 9 features | 12 features | Change |
|--------|-----------|-------------|--------|
| PHYSBO HV (160 cycles) | 0.8116 | 0.8160 | +0.5% |
| PHYSBO discovery (160 cycles) | 0% (never) | 100% | +100% |
| RE HV | 0.8076 | 0.8076 | 0% (expected) |
| PHYSBO vs RE gap | +0.004 | +0.008 | 2× wider |

RE is unaffected (doesn't use features). The doubling of the BO advantage confirms
the new features are capturing real physical structure in the data.

---

## 5. Perovskite band gap corrections (scissor)

### Why PBE underestimates band gaps

MP provides PBE (GGA) band gaps without spin-orbit coupling (SOC). PBE systematically
underestimates gaps due to self-interaction error in the exchange-correlation functional.
The error is not constant — it depends on how strongly the B-site metal's s-orbital
hybridises with the X-site halide p-orbital.

### Why per-{B,X} pair corrections are necessary

As X goes I → Br → Cl:
- The B-s/X-p hybridisation weakens (halide p-orbitals become less diffuse)
- The B-s self-interaction error grows (more localised orbital)
- The required correction increases

A single scalar per B-site (the first approach) captures the B-site physics but not
the halide dependence. This caused:
- CsSnBr₃: single-scalar correction gave 1.35 eV, experimental is 1.75 eV (+0.40 eV error)
- CsGeBr₃: correction gave 1.29 eV, experimental is 2.32 eV (+1.03 eV error)

Both materials appeared as top PV candidates when they should not have been.

### Calibrated per-{B,X} offsets (eV)

| B \ X | I    | Br   | Cl   | Source |
|-------|------|------|------|--------|
| Pb    | 0.20 | 0.45 | 0.50 | PBE-noSOC vs experiment |
| Sn    | 0.75 | 1.15 | 1.00 | Stoumpos 2013, Gupta 2016 |
| Ge    | 0.65 | 1.55 | 1.60 | HSE calcs, ACS Omega 2022 |
| Bi    | 0.30 | 0.40 | 0.50 | conservative estimate |

**Effect on NIMO:** After correction, PHYSBO discovery improved from ~40 cycles → ~10 cycles.
The old correction was steering NIMO toward wrong materials (CsSnBr₃, CsGeBr₃).

---

## 6. Shannon ionic radii for perovskite features

### Why atomic radii were wrong for Goldschmidt/octahedral factors

**Goldschmidt tolerance factor:**
```
t = (r_A + r_X) / (√2 · (r_B + r_X))
Stable cubic perovskite: 0.80 ≤ t ≤ 1.06
```

**Octahedral factor:**
```
μ = r_B / r_X
Stable BX₆ octahedra: 0.41 ≤ μ ≤ 0.73
```

Using pymatgen *atomic* radii (metallic/covalent): Pb = 1.80 Å, I = 1.40 Å.
This gives μ = 1.80/1.40 = 1.28 → above 0.73 for ALL 51 perovskites.
Octahedral factor was useless as a feature (no variance).

**Why Shannon ionic radii are correct:**
These are coordination-number-specific ionic radii measured from crystal structures.
Physically correct for the ionic bonding in perovskites.

Correct assignments:
- A-site (+1 cation, cuboctahedral corner, CN=XII): e.g. Cs⁺ = 1.88 Å
- B-site (+2 cation, octahedral body-centre, CN=VI): e.g. Pb²⁺ = 1.19 Å
- X-site (−1 halide, face-centre, CN=VI): e.g. I⁻ = 2.20 Å

With correct radii: Pb²⁺/I⁻ gives μ = 1.19/2.20 = 0.54 ✓ (within 0.41–0.73)

**Sn²⁺ missing from pymatgen:** Shannon (1976) lists Sn²⁺ CN=VI = 1.18 Å but pymatgen
only has Sn⁴⁺. Hardcoded fallback: `_SHANNON_FALLBACK = {("Sn", +2, "VI"): 1.18}`.

---

## 7. Deduplication decisions

### Battery: one entry per framework_formula, keep highest voltage

MP's `insertion_electrodes` endpoint returns one document per voltage sub-range per
material. A single framework like LiFePO₄ may appear as multiple entries covering
different voltage plateaus. Keeping one entry (highest voltage) gives one row per
unique cathode framework in the search space.

### Perovskite: one entry per formula, keep lowest energy_above_hull

Multiple MP entries for the same formula (e.g. 3× cubic/orthorhombic CsSnI₃) represent
polymorphs. Keeping all conflates two different questions:
1. "Which composition is best?"
2. "Which polymorph of that composition is best?"

NIMO's GP cannot distinguish polymorphs of the same composition from composition features
alone (same tm_eneg, same a_radius, etc.) — they have identical feature vectors. This
caused the GP to effectively see 3 copies of CsSnI₃ with different objective values,
confusing the surrogate.

Fix: keep the ground-state polymorph (lowest hull energy) per formula.
Result: 51 → 23 unique compositions. PHYSBO discovery improved 40 cycles → 10 cycles.

---

## 8. Why BO outperforms RE and when it does not

### BO advantage mechanism

PHYSBO places a Gaussian Process surrogate over the feature → objective mapping.
After each observation, it updates the posterior and proposes the next experiment via
Thompson sampling (for multi-objective: one GP per objective, independent sampling).

The GP learns structure: "high polyanion_eneg + Mn³⁺/Mn⁴⁺ → high voltage". Once
learned, it proposes experiments in regions of high predicted utility rather than
random positions. On smooth feature-objective landscapes, this is far more efficient
than random search.

### When the advantage is small

1. **Coverage too high:** 160/892 = 18% coverage. Random search hits good materials
   by luck when sampling 1 in 5 candidates. BO's steering advantage is diluted.
   On small pools (23 perovskites, 60/23 = 260% coverage) BO is critical.

2. **Poor features:** With 9 features that don't encode the dominant physics (redox
   couple, inductive effect), the GP builds a weak surrogate. BO degenerates toward
   random. Adding 3 physics features doubled the BO advantage over RE.

3. **Multi-objective complexity:** The GP must model two objectives simultaneously.
   More training data needed to resolve the full Pareto front than a single peak.

### Method comparison summary

| Method | Mechanism | Best for | Worst for |
|--------|-----------|----------|-----------|
| RE | Random sampling | Large diverse pools, any budget | Never optimal |
| PHYSBO | GP + Thompson sampling | Smooth landscapes, moderate budget | Very small datasets |
| BLOX | Random forest acquisition | Non-smooth, many discrete features | Small N (< 100 training points) |
| NTS | GP + neighbourhood search | Multi-modal landscapes | Slow on large pools |
| AX | BoTorch Sobol + GP | Well-specified objectives, large budget | Small pools (Sobol init > pool size) |

---

## 9. Na-ion vs Li-ion parameter differences

| Parameter | Li-ion | Na-ion | Reason |
|-----------|--------|--------|--------|
| Voltage window | 2.5–5.5 V | 1.5–5.0 V | Na/Na⁺ reference is 0.3 V above Li/Li⁺; Na insertion is thermodynamically less favourable |
| Practical cap | 4.5 V | 4.0 V | Na-ion electrolytes are less oxidatively stable |
| Hull threshold | 0.10 eV/atom | 0.15 eV/atom | Na-ion cathode space less mature; more metastable phases are experimentally relevant |
| Max ΔV | 0.40 | 0.50 | Na⁺ ionic radius 1.02 Å vs Li⁺ 0.76 Å — larger ion causes more lattice distortion |
| TM pool | 8 elements | 9 elements (+ Cu) | Na₂CuO₂ and related Cu-oxide frameworks are known Na cathodes |
| Pool size | 892 | 206 | Na-ion space less explored in MP |

**Feature set is identical** — same 12 features. The inductive effect, structural
prototype, and redox couple physics apply equally to Na insertion. The GP transfers.

---

## 10. MP API design decisions

### Single batch query for initial fetch

Both Li and Na pipelines use one `insertion_electrodes.search()` call with voltage/volume
filters rather than per-material queries. This returns O(1000) documents in ~2 seconds.
Per-material queries would take O(1000) × 0.5s inter-call delay = ~8 minutes.

### Oracle queries (NIMO loop only)

The NIMO loop queries one battery_id at a time (`battery_ids=[single_id]`). This is
correct — the loop needs to simulate an experiment (one at a time). Batch oracle
queries are only used in `mp_bulk_fetch_battery.py` for benchmark pre-population.

### battery_ids list filter broken

`insertion_electrodes.search(battery_ids=[id1, id2, ...])` returns only 1 result
regardless of list length. Probable cause: the filter uses AND semantics or has a
server-side bug. Workaround: re-run the full voltage/volume scan and match locally.

### thermo_type always None

The `thermo_type` field on `InsertionElectrodeDoc` is no longer populated by the API.
Do not filter on it. Use `stability_discharge ≤ hull_threshold` instead.
