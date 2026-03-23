"""
Bulk-populate average_voltage and max_delta_volume for all battery candidates
in a single MP API call (needed before running the benchmark).

Reads battery_candidates.csv, queries all battery_ids in one batch, writes
the results back.  No per-material sleep needed — one network round-trip.

Usage:
    export MP_API_KEY="your_key_here"
    python mp_bulk_fetch_battery.py [--config battery_config.yaml]
"""

import os
import sys
import csv
import argparse

import yaml
from mp_api.client import MPRester


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path):
    MP_API_KEY = os.environ.get("MP_API_KEY")
    if not MP_API_KEY:
        print("Error: set MP_API_KEY environment variable first.")
        sys.exit(1)

    cfg  = load_config(config_path)
    path = cfg["files"]["candidates"]

    with open(path) as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())

    blank = [r for r in rows
             if r["average_voltage"] == "" or r["max_delta_volume"] == ""]
    print(f"Candidates total  : {len(rows)}")
    print(f"Need values       : {len(blank)}")

    if not blank:
        print("All candidates already populated. Nothing to do.")
        return

    bat = cfg["battery"]
    print(f"Re-scanning MP insertion_electrodes (single batch, "
          f"voltage {bat['voltage_min']}–{bat['voltage_max']} V)…")

    lookup = {}
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.materials.insertion_electrodes.search(
            working_ion=bat["working_ion"],
            average_voltage=(bat["voltage_min"], bat["voltage_max"]),
            max_delta_volume=(0.0, bat["max_delta_volume"]),
            all_fields=False,
            fields=["battery_id", "average_voltage", "max_delta_volume"],
        )
    print(f"MP returned {len(docs)} docs; matching to {len(blank)} candidates…")
    for doc in docs:
        bid = doc.battery_id
        v   = float(doc.average_voltage)  if doc.average_voltage  is not None else None
        m   = float(doc.max_delta_volume) if doc.max_delta_volume is not None else None
        lookup[bid] = (v, m)

    filled = skipped = 0
    for r in rows:
        if r["average_voltage"] != "" and r["max_delta_volume"] != "":
            continue
        val = lookup.get(r["battery_id"])
        if val is None or val[0] is None or val[1] is None:
            skipped += 1
            continue
        r["average_voltage"]  = str(round(val[0], 6))
        r["max_delta_volume"] = str(round(val[1], 6))
        filled += 1

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Filled   : {filled}")
    print(f"Skipped  : {skipped}  (incomplete MP data)")
    print(f"Written  → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="battery_config.yaml")
    args = parser.parse_args()
    main(args.config)
