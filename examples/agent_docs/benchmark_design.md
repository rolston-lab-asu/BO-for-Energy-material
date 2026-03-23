# Benchmark Design

## Purpose

The benchmark scripts compare NIMO selection methods (RE, PHYSBO, BLOX, NTS, AX)
using a fully-populated candidates CSV as the oracle. No MP API calls are made —
each "experiment" is an instant table lookup. This allows 5 seeds × 80 cycles to
run in minutes instead of days.

## Prerequisites

The candidates CSV must have **all objective columns populated** before benchmarking.
- Perovskite / antiperovskite: run the NIMO loop to completion (small pools)
- Battery: run `mp_bulk_fetch_battery.py` (single MP re-scan, fills all 892 in ~2s)

## Metrics

### Hypervolume (primary metric for 2-objective problems)

2D hypervolume dominated by the current Pareto front relative to a reference point.
Both objectives are minimised. For battery: objectives are `(-average_voltage, max_delta_volume)`.

```python
def hypervolume_2d(points, ref):
    # points: list of (obj1, obj2) — both minimised
    # ref: (ref1, ref2) — strictly worse than all Pareto points
    dominated = [(v, m) for v, m in points if v < ref[0] and m < ref[1]]
    pts = sorted(dominated, key=lambda p: p[0])
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return hv
```

Reference point = (max_obj1 + margin, max_obj2 + margin) over the full oracle.
Margins: 0.1 for voltage (V), 0.01 for delta_volume (fraction).

Higher hypervolume = better Pareto front = more efficient search.

### Discovery cycle

1-indexed cycle at which the global best single-objective value is first found
(within 1e-4 tolerance). "never" if not found within NUM_CYCLES.

On large pools (892 battery candidates, 80 cycles = 9% coverage), discovery is
almost always "never" and should not be used as the primary metric. Hypervolume is.

On small pools (23 perovskites, 60 cycles = 260% coverage), discovery is reliable.

## Seeding

RE (random exploration) uses `re_seed = seed_offset * 1000 + cycle` to ensure
reproducible but varied random sequences across seeds.
AI methods (PHYSBO, BLOX, etc.) are deterministic given the same training data;
variation comes from different RE seed phases.

## Known method behaviour on battery pool (892 candidates)

- **PHYSBO** best HV (0.794): GP kernel learns smooth voltage/ΔV trade-off
- **RE** middle (0.781): large diverse pool means random 9% gets decent coverage
- **BLOX** worst (0.766): RF needs more data than 80 samples to generalise over
  9 continuous features on 892 candidates
- **NTS**: very slow on large pools (~10 min/seed), not yet benchmarked to completion
- **AX**: Sobol initialisation phase = 20 trials, may exceed pool size on small sets

## Running the benchmark

```bash
# Fast (RE, PHYSBO, BLOX only):
python mp_benchmark_battery.py --methods RE PHYSBO BLOX --seeds 5

# Full (add NTS and AX — takes ~1 hour for battery):
python mp_benchmark_battery.py --seeds 5

# Quick smoke-test (fewer cycles):
python mp_benchmark_battery.py --methods RE PHYSBO --seeds 2 --cycles 20
```
