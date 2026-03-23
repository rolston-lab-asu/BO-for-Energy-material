# Architecture — NIMO MP Pipeline

## Directory layout

```
examples/
├── agent_docs/              ← context docs (read on demand)
├── *_config.yaml            ← single source of truth per phase
├── mp_fetch_*.py            ← fetch candidates from MP, compute features
├── mp_nimo_*.py             ← NIMO optimisation loop (calls MP oracle)
├── mp_benchmark_*.py        ← offline benchmark (no API, table lookup)
├── mp_bulk_fetch_battery.py ← pre-populate all battery objectives
└── *_candidates.csv         ← living dataset (objectives filled in-place)
```

## Data flow per phase

```
mp_fetch_*.py
    → queries MP API (one batch or per-chemsys)
    → filters (hull, TM presence, thermo stability)
    → deduplicates (one entry per composition)
    → computes 9–14 features from composition + structure
    → writes *_candidates.csv with objective columns BLANK

mp_nimo_*.py  (one cycle = one MP API call)
    → builds NIMO working CSV from candidates (features + blank/filled objectives)
    → calls nimo.selection(method, ...) → writes proposals CSV
    → reads proposals CSV (DictReader, skip 'actions' col)
    → calls MP oracle for each proposed battery_id / material_id
    → fills objective values back into candidates CSV
    → calls nimo.history() for PHYSBO internal state
    → saves checkpoint JSON for resume

mp_benchmark_*.py  (no API)
    → requires fully populated candidates CSV
    → oracle = dict keyed by battery_id / material_id
    → runs N seeds × M cycles per method in tempdir
    → reports hypervolume + discovery cycle per method
```

## NIMO CSV contract

The NIMO library expects a CSV where:
- All columns except the last `num_objectives` are features (must be numeric)
- The last `num_objectives` columns are objectives (blank = unmeasured)
- Both objectives are **minimised** — negate voltage before writing

Battery objectives written to NIMO CSV:
```
col[-2] = neg_avg_voltage = -average_voltage   (minimise → maximise V)
col[-1] = max_delta_volume                      (minimise directly)
```

Perovskite objectives:
```
col[-2] = formation_energy_per_atom
col[-1] = band_gap_dev = |band_gap_corrected - bg_target|
```

## Proposals CSV format

NIMO writes proposals with an `actions` column (row index) as **column 0**,
then the feature columns. Always read with `csv.DictReader` and skip `actions`:

```python
def read_proposals(proposals_path, candidates):
    proposed = []
    with open(proposals_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["actions"])
            proposed.append((idx, candidates[idx]))
    return proposed
```

The battery loop uses a float-matching fallback (legacy) — the benchmark uses
index-based matching (correct). Always use index-based.

## Feature sets

### Battery (9 features)
| Feature | Source | Notes |
|---------|--------|-------|
| tm_eneg | composition-weighted Pauling eneg of TM(s) | |
| tm_ionic_radius | composition-weighted atomic radius of TM(s) Å | |
| tm_max_oxidation_state | composition-weighted max common ox state | |
| anion_eneg | highest-eneg non-TM element | picks O/S/F |
| li_per_fu | Li atoms per formula unit (discharge) | |
| spacegroup_number | SG of Li-free framework | SpacegroupAnalyzer symprec=0.1 |
| crystal_system | int 1–7 (triclinic=1 … cubic=7) | |
| volume_per_atom | framework vol / nsites (Å³/atom) | |
| density | framework mass density g/cm³ | |

### Perovskite (14 features)
a_eneg, b_eneg, x_eneg, a_radius, b_radius, x_radius (Shannon ionic),
a_max_oxid, b_max_oxid, tolerance_factor, octahedral_factor,
spacegroup_number, crystal_system, volume_per_atom, density

Shannon ionic radii: A-site CN=XII (+1), B-site CN=VI (+2), X-site CN=VI (−1).
Sn²⁺ CN=VI missing from pymatgen → hardcoded fallback 1.18 Å.

## Checkpoint / resume

All NIMO loops save `*_checkpoint.json` after each cycle:
```json
{
  "method": "PHYSBO",
  "next_cycle": 12,
  "curves": {"best_voltage": [...], "best_delta_volume": [...]},
  "all_queried": [[cycle, formula, id, tm, volt, mdv], ...]
}
```
On restart the script detects the checkpoint and continues from `next_cycle`.
Checkpoint is deleted on successful completion.

## Adding a new phase

1. Copy `battery_config.yaml` → `newphase_config.yaml`. Define:
   - `files`: candidates, proposals, nimo_csv, checkpoint, plot_prefix
   - `nimo`: num_cycles, seed_cycles, num_objectives (always 2), num_proposals, default_method
   - `feature_cols`: must match CSV column names exactly
   - `api`: max_retries, retry_base_delay, inter_call_delay

2. Write `mp_fetch_newphase.py`:
   - One batch query (or per-chemsys loop)
   - Filter → deduplicate → compute features
   - Write CSV with objectives blank

3. Write `mp_nimo_newphase.py`:
   - Use index-based `read_proposals()` (DictReader + actions col)
   - Negate any objectives that should be maximised before writing NIMO CSV
   - Include checkpoint save/load

4. Write `mp_benchmark_newphase.py`:
   - Copy battery benchmark, change oracle key and objective logic
   - HV reference point: (max_obj1 + margin, max_obj2 + margin)
