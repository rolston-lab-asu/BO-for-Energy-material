# Perovskite Scissor Calibration

## Why scissors are needed

MP provides PBE (GGA) band gaps without spin-orbit coupling (SOC).
PBE systematically underestimates gaps, and the error is **B-site AND halide dependent**:
as X goes I → Br → Cl, B-s/X-p hybridisation weakens, B-s self-interaction error grows,
and the needed correction increases. A single scalar per B-site is wrong.

## Per-{B,X} offsets (eV) — current calibration

| B \ X | I    | Br   | Cl   |
|-------|------|------|------|
| Pb    | 0.20 | 0.45 | 0.50 |
| Sn    | 0.75 | 1.15 | 1.00 |
| Ge    | 0.65 | 1.55 | 1.60 |
| Bi    | 0.30 | 0.40 | 0.50 |

Corrected gap = PBE gap + offset.
Stored as `band_gap` in CSV. Raw PBE stored as `band_gap_raw` for traceability.

## Calibration sources

| System | Exp. gap (eV) | PBE gap (eV) | Offset | Source |
|--------|--------------|--------------|--------|--------|
| CsSnI₃ | 1.28 | 0.50 | +0.75 | Stoumpos 2013 |
| CsSnBr₃ | 1.75 | 0.60 | +1.15 | Gupta ACS-EL 2016 |
| CsSnCl₃ | — | — | +1.00 | trend estimate |
| CsGeI₃ | 1.60 | ~1.0 | +0.65 | HSE calc |
| CsGeBr₃ | 2.32 | 0.78 | +1.55 | HSE (ACS Omega 2022) |
| CsGeCl₃ | — | 2.15 | +1.60 | HSE (ACS Omega 2022) |
| CsPbI₃ | 1.73 | 1.55 | +0.20 | PBE-noSOC |
| CsPbBr₃ | 2.36 | 1.90 | +0.45 | PBE-noSOC |
| CsPbCl₃ | 3.00 | 2.50 | +0.50 | PBE-noSOC |

## History of errors

**Before per-{B,X} scissors (single scalar per B-site):**
- Sn offset was +0.75 eV for all halides
- CsSnBr₃ corrected to 1.35 eV → appeared as best PV candidate
- Actual experimental gap: 1.75 eV — **0.40 eV off**
- CsGeBr₃ corrected to 1.29 eV with +0.50 → actual experimental: 2.32 eV — **1.0 eV off**

Root cause: Ge-Br and Ge-Cl have much larger PBE errors than Ge-I because
the Ge-s/X-p hybridisation collapses faster for heavier halides.

**After per-{B,X} scissors:**
- CsSnBr₃ corrected to 2.15 eV — correctly deprioritised
- CsGeI₃ remains a viable candidate (~1.65 eV)
- PHYSBO discovery improved from ~40 cycles → ~10 cycles after dedup

## Implementation in mp_nimo_perovskite.py

```python
SCISSOR_OFFSETS = cfg["scissor"]["offsets"]   # nested dict: B → X → float
B_COLUMN        = cfg["scissor"]["B_column"]  # "B"
X_COLUMN        = cfg["scissor"]["X_column"]  # "X"

def query_mp(material_id, b_elem, x_elem, ...):
    offset = SCISSOR_OFFSETS.get(b_elem, {}).get(x_elem, 0.0)
    band_gap_corrected = band_gap_raw + offset
```
