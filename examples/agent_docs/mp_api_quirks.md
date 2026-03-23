# MP API Quirks

All issues discovered through live use of `mp-api` in this project.
Check here before writing any new MP query code.

---

## insertion_electrodes endpoint

### thermo_type is always None
`doc.thermo_type` returns `None` for all insertion electrode docs regardless of
the actual thermodynamic type. The field is no longer populated by the API.
**Do not filter on thermo_type.** Use hull stability instead:
```python
stab = doc.stability_discharge
if stab is None or float(stab) > HULL_THRESHOLD:
    continue
```

### battery_ids filter with a list is broken
`mpr.materials.insertion_electrodes.search(battery_ids=[id1, id2, ...])` returns
only 1 result regardless of list length. The filter behaves like AND, not OR.

**Workaround for bulk populate:** re-run the full voltage/volume-range scan
(same as initial fetch), get all 2184 docs, then match to candidates locally:
```python
docs = mpr.materials.insertion_electrodes.search(
    working_ion="Li",
    average_voltage=(2.5, 5.5),
    max_delta_volume=(0.0, 0.40),
    all_fields=False,
    fields=["battery_id", "average_voltage", "max_delta_volume"],
)
lookup = {doc.battery_id: (doc.average_voltage, doc.max_delta_volume) for doc in docs}
```

**Single-ID queries still work correctly** (the NIMO oracle uses `battery_ids=[single_id]`).

### battery_id format
`battery_id = f"{mpid}_{working_ion}"` e.g. `mp-26531_Li`.
The `_Li` suffix is always the working ion symbol.

---

## summary endpoint (used for perovskites)

### Per-chemsys queries
Perovskite fetch uses `mpr.materials.summary.search(chemsys="Cs-Pb-I", num_elements=3, ...)`.
This returns all materials in the ternary system. The `num_elements=3` filter is critical
to exclude binary and quaternary phases from the results.

### symmetry field
`doc.symmetry.number` → spacegroup number (int)
`doc.symmetry.crystal_system.value.lower()` → crystal system string

---

## General

### Rate limits
Use `inter_call_delay: 0.5` for electrode endpoint (stricter).
Use `inter_call_delay: 0.3` for summary endpoint.
Both configured in `api` section of yaml.

### Retry pattern
All query functions use exponential backoff:
```python
wait = base_delay * (2 ** attempt)   # 2, 4, 8 seconds
```
`max_retries: 3` is sufficient for transient errors.

### host_structure vs framework
- Battery: `doc.host_structure` → pymatgen Structure of the Li-free framework
- Use `SpacegroupAnalyzer(host_structure, symprec=0.1)` — symprec=0.1 is more
  tolerant than the default 0.01 and handles slightly distorted battery structures

### framework field
`doc.framework` → pymatgen Composition of the framework (no Li).
Used to compute TM features and anion electronegativity.
