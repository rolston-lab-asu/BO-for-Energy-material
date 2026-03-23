# Structural Stability Mapping of Li₃AB Antiperovskites with Polyatomic B-Site Anions

## Literature Review & Computational Gap Analysis

**Date:** 2026-03-20
**Context:** Prospective design of non-halide Li₃(A²⁻)(B⁻) antiperovskite solid electrolytes

---

## 1. Background

Lithium-rich antiperovskites (Li₃AB, where A is a divalent anion and B is a monovalent
anion) are promising solid electrolyte candidates for all-solid-state batteries. The
prototypical compounds Li₃OCl and Li₃OBr were first reported by Zhao & Daemen [1] with
claimed ionic conductivity of 1.94×10⁻³ S/cm for Li₃OCl₀.₅Br₀.₅, though this value
has not been independently reproduced [4].

The field has since expanded along several axes: hydroxide variants (Li₂OHCl) [4,8],
hydride-chalcogenides (Li₃HCh, Ch = S, Se, Te) requiring 5 GPa synthesis [5], and
computationally predicted cluster-ion compositions (Li₃SBF₄, Li₃OBH₄) [2] that remain
unsynthesized.

**Our direction:** Design Li₃AB antiperovskites where B is a monovalent non-halide anion,
using tolerance factor, octahedral tilt, and Shannon ionic radii as synthesizability
constraints. Preliminary work showed that Li₃O(NO₃) is not viable because the effective
ionic radius of NO₃⁻ pushes the Goldschmidt tolerance factor above 1.0.

---

## 2. State of the Art

### 2.1 Experimentally realized Li₃AB antiperovskites

| Composition | B-site | σ_RT (S/cm) | Synthesis | Status |
|-------------|--------|-------------|-----------|--------|
| Li₃OCl₀.₅Br₀.₅ | Halide | 1.94×10⁻³ (claimed) | Solid-state | Disputed [4] |
| Li₂OHCl | OH⁻ | 1.4×10⁻⁶ | Solid-state | Reproducible [4,19] |
| Li₃HS, Li₃HSe, Li₃HTe | H⁻ | Low (RT) | 5 GPa, 700°C | Confirmed [5] |

Only Li₂OHCl has reproducible conductivity at ambient pressure. The original Li₃OCl
results remain unverified [4], and the hydride-chalcogenides require extreme synthesis
conditions [5]. No Li-based non-halide antiperovskite has achieved high conductivity
under ambient conditions.

### 2.2 Na-system analogues (experimental, non-halide B-site)

Na₃ONO₂ achieves 0.37 mS/cm at 485 K via a paddlewheel mechanism where rotational
NO₂⁻ groups facilitate Na⁺ migration [17]. Na₃OCN shows similar conductivity jumps near
500 K [18]. These demonstrate that polyatomic B-site anions can enable fast ion transport
through rotational coupling — but no Li-based equivalent has been synthesized or computed.

### 2.3 Computational screening studies

Three large-scale screening studies have been published:

**Fang & Jena (2017)** [2] introduced the "super-LRAP" concept with cluster ions at the
B-site. They tested only 3 anions (BH₄⁻, BF₄⁻, AlH₄⁻) with A = O and S, predicting
Li₃SBF₄ at ~10⁻² S/cm (Ea = 0.210 eV). Phonon stability and AIMD conductivity were
computed. None have been synthesized.

**Lee et al. (2024)** [11] screened 18,133 hypothetical X₃BA antiperovskites using
active learning with genetic algorithms and Bayesian optimization. The search space is
monoatomic-only — polyatomic ions were not included. Oracle: DFT (formation energy,
Ehull) then AIMD (conductivity). Identified 7 candidates with >4 mS/cm.

**Lin, Zhang & Dong (2025)** [12] screened 12,840 candidates using ROOST (a
composition-based ML model) with sequential filtering: tolerance factor → thermodynamic
stability → electronic conductivity → mechanical stability → electrochemical stability →
ionic conductivity. Validated 8 top SSEs with DFT + AIMD. ROOST cannot distinguish
polyatomic ion compositions (it encodes stoichiometry only, so Li₃O(NO₃) and Li₃O(BH₄)
are indistinguishable).

### 2.4 Double antiperovskites

Li₆OS(BH₄)₂ has been synthesized as a thin film (bulk modulus 128.5 GPa, band gap
4.03 eV) [15]. Li₆.₅OS₁.₅I₁.₅ shows 2–3 orders of magnitude higher conductivity than
Li₃OCl [16]. The double antiperovskite space has not been systematically screened.

### 2.5 Tolerance factor limitations

Coutinho Dutra & Dawson (2023) [7] explicitly state that the Goldschmidt tolerance
factor is "not a suitable descriptor for antiperovskite materials, especially those
containing heavy halides and cluster ions." Bartel et al. (2019) [13] proposed a new
tolerance factor for regular perovskites, but it was not designed for antiperovskites and
does not handle polyatomic ions. No validated replacement exists for antiperovskites with
polyatomic B-site anions.

---

## 3. Identified Computational Gaps

### Gap 1: No systematic polyatomic B-site screening for Li₃AB

The two large screening studies [11,12] are restricted to monoatomic ions. Fang & Jena
[2] tested only 3 cluster anions. The following monovalent polyatomic anions have NOT
been computed in Li₃AB antiperovskite structures:

- NO₂⁻ (verified experimentally in Na system [17])
- CN⁻ (verified experimentally in Na system [18])
- OCN⁻, N₃⁻, SCN⁻
- NO₃⁻ (ruled out by tolerance factor for A=O, but unexplored with A=S,Se,Te)

### Gap 2: No structural stability descriptor for polyatomic B-site ions

The Goldschmidt tolerance factor breaks down for cluster ions [7], but no replacement
has been proposed for the Li antiperovskite system. The existing screening studies either
exclude cluster ions entirely [11,12] or use crude effective radius estimates [2].
Kieslich et al. extended tolerance factor to hybrid organic-inorganic perovskites by
treating molecular ions as rigid cylinders, but this has not been adapted for
antiperovskites.

### Gap 3: Tolerance factor failure boundary unmapped across A-site

Our preliminary analysis showed Li₃O(NO₃) fails because t > 1.0. Fang & Jena [2]
found empirically that switching A from O²⁻ to S²⁻ stabilized BF₄⁻ compositions.
However, nobody has systematically mapped: for each polyatomic B⁻ ion, which A²⁻ site
(O, S, Se, Te) brings t into the stable cubic window? This (A, B) → t map is the
fundamental missing piece.

### Gap 4: A-site chalcogenide + polyatomic B-site unexplored

Beyond hydride (H⁻), no study has combined A = {S, Se, Te} with polyatomic B-site ions.
Fang & Jena [2] covered A = {O, S} × B = {BH₄, BF₄, AlH₄} — a 2×3 grid. The full
space is A = {O, S, Se, Te} × B = {~10–15 polyatomic anions} = 40–60 compositions,
most of which are completely unexplored.

### Gap 5: Synthesizability beyond energy above hull

Every screening study uses Ehull as the sole stability proxy [11,12]. Geometric
constraints (tolerance factor, octahedral factor) are used only as coarse filters, not
as quantitative synthesizability descriptors. The Li₃SI synthesis failure [6] (reactants
melt before forming the antiperovskite phase) shows that Ehull alone is insufficient —
kinetic accessibility matters.

---

## 4. Proposed Computational Study

**Title:** Structural stability mapping of Li₃AB antiperovskites with polyatomic B-site
anions: bridging the gap between geometric screening and DFT validation

### Scope

1. **Polyatomic ion enumeration:** Compile monovalent non-halide anions with literature
   effective ionic radii (~10–15 candidates from Jenkins/Goldschmidt tables).

2. **Tolerance factor map:** For each (A²⁻, B⁻) pair, compute t using both the standard
   Goldschmidt formula and effective molecular radii. Map the cubic stability boundary
   across all A-site chalcogenides (O, S, Se, Te).

3. **DFT relaxation + phonon stability:** For candidates inside the geometric window
   (~10–20 structures), perform full structural relaxation using the Li₃OCl Pm-3m
   prototype, followed by phonon calculations to confirm dynamic stability.

4. **AIMD ionic conductivity:** For dynamically stable candidates (~3–5), compute
   room-temperature Li⁺ conductivity and migration barriers via ab initio molecular
   dynamics.

5. **Validation against known systems:** Compare predicted tolerance factor boundaries
   with experimental outcomes for Na₃ONO₂ [17], Na₃OCN [18], Li₃HS/HSe/HTe [5], and
   the Li₃SI synthesis failure [6].

### Differentiation from existing work

- First systematic computation of polyatomic B-site ions beyond BH₄/BF₄/AlH₄
- First (A, B) tolerance factor stability map for antiperovskites with cluster ions
- Li-side analogues of experimentally verified Na systems (NO₂⁻, CN⁻)
- Geometric synthesizability constraints as a complement to Ehull

---

## References

[1] Y. Zhao, L. L. Daemen, "Superionic Conductivity in Lithium-Rich Anti-Perovskites,"
    J. Am. Chem. Soc. 134, 15042–15047 (2012). DOI: 10.1021/ja305709z

[2] H. Fang, P. Jena, "Li-rich antiperovskite superionic conductors based on cluster
    ions," Proc. Natl. Acad. Sci. U.S.A. 114, 11046–11051 (2017).
    DOI: 10.1073/pnas.1704086114

[3] A. Emly, E. Kioupakis, A. Van der Ven, "Phase Stability and Transport Mechanisms
    in Antiperovskite Li₃OCl and Li₃OBr Superionic Conductors," Chem. Mater. 25,
    4663–4670 (2013). DOI: 10.1021/cm4016222

[4] Y. Zheng, Y. Perry, Y. Wu, "Antiperovskite Superionic Conductors: A Critical
    Review," ACS Mater. Au (2021). DOI: 10.1021/acsmaterialsau.1c00026

[5] S. Gao, T. Broux, S. Fujii et al., "Hydride-based antiperovskites with soft anionic
    sublattices as fast alkali ionic conductors," Nat. Commun. 12, 201 (2021).
    DOI: 10.1038/s41467-020-20370-2

[6] L. Yin, M. Murphy, K. Kim et al., "Synthesis of Antiperovskite Solid Electrolytes:
    Comparing Li₃SI, Na₃SI, and Ag₃SI," Inorg. Chem. 59, 11244–11247 (2020).
    DOI: 10.1021/acs.inorgchem.0c01705

[7] A. C. Coutinho Dutra, J. A. Dawson, "Computational Design of Antiperovskite Solid
    Electrolytes," J. Phys. Chem. C 127, 18256–18270 (2023).
    DOI: 10.1021/acs.jpcc.3c04953

[8] W. Xia, Y. Zhao, F. Zhao et al., "Antiperovskite Electrolytes for Solid-State
    Batteries," Chem. Rev. 122, 3763–3819 (2022). DOI: 10.1021/acs.chemrev.1c00594

[9] J. A. Dawson, T. Famprikis, K. E. Johnston, "Anti-perovskites for solid-state
    batteries: recent developments, current challenges and future prospects," J. Mater.
    Chem. A 9, 18746–18772 (2021). DOI: 10.1039/D1TA03680G

[10] Z. Deng et al., "Anti-perovskite materials for energy storage batteries," InfoMat
     4, e12252 (2022). DOI: 10.1002/inf2.12252

[11] Lee, Shin, Kim, Cho, Lee, "Discovering virtual antiperovskites as solid-state
     electrolytes through active learning," Energy Storage Mater. 70, 103535 (2024).
     DOI: 10.1016/j.ensm.2024.103535

[12] C. Lin, L. Zhang, Y. Dong, "Compositional machine learning and high-throughput
     screening aided discovery of novel anti-perovskite solid-state electrolytes,"
     J. Energy Storage 125, 116990 (2025). DOI: 10.1016/j.est.2025.116990

[13] C. J. Bartel et al., "New tolerance factor to predict the stability of perovskite
     oxides and halides," Sci. Adv. 5, eaav0693 (2019).
     DOI: 10.1126/sciadv.aav0693

[14] C. Guan, H. Jing, Y. Yang, R. Ouyang, H. Zhu, "Data-Driven Theoretical Design of
     Anion Cluster-Based Sodium Antiperovskite Superionic Conductors," ACS Appl. Mater.
     Interfaces 16, 70665–70674 (2024). DOI: 10.1021/acsami.4c16856

[15] M. M. Islam, A. A. Maruf, J. Pokharel, Y. Zhou, "Superhalogen-based Li-rich double
     antiperovskite Li₆OS(BH₄)₂ as solid electrolyte," MRS Commun. 12, 1140–1146
     (2022). DOI: 10.1557/s43579-022-00290-6

[16] H. Xu, M. Xuan, W. Xiao et al., "Lithium Ion Conductivity in Double Antiperovskite
     Li₆.₅OS₁.₅I₁.₅: Alloying and Boundary Effects," ACS Appl. Energy Mater. 2,
     6288–6294 (2019). DOI: 10.1021/acsaem.9b00861

[17] L. Gao, H. Zhang, Y. Wang et al., "Mechanism of enhanced ionic conductivity by
     rotational nitrite group in antiperovskite Na₃ONO₂," J. Mater. Chem. A 8,
     21265–21272 (2020). DOI: 10.1039/D0TA07110B

[18] M. Jansen, "Volume Effect or Paddle-Wheel Mechanism — Fast Alkali-Metal Ionic
     Conduction in Solids with Rotationally Disordered Complex Anions," Angew. Chem.
     Int. Ed. Engl. 30, 1547–1558 (1991). DOI: 10.1002/anie.199115471

[19] L. Gao, X. Zhang, J. Zhu et al., "Boosting lithium ion conductivity of
     antiperovskite solid electrolyte by potassium ions substitution for cation
     clusters," Nat. Commun. 14 (2023). DOI: 10.1038/s41467-023-42385-1

[20] A. Merchant, S. Batzner, S. S. Schoenholz et al., "Scaling deep learning for
     materials discovery," Nature 624, 80–85 (2023). DOI: 10.1038/s41586-023-06735-9
