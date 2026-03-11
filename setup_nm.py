"""Populate well atoms for New Mexico wells with first production in 2023.

Usage:
    python setup_nm.py              # 50 wells (default)
    python setup_nm.py --count 100  # custom count
    python setup_nm.py --all        # all eligible wells

Creates wells/{entity_key}/ with production.csv, params.json,
trace.jsonl, and hindcast.json.
"""

import argparse
import csv
import json
from pathlib import Path

import duckdb

WAREHOUSE_PATH = "../data-warehousers/data/warehouse.duckdb"
WELLS_DIR = Path("wells/NM")

BASELINE_PARAMS = {
    "version": 1,
    "description": "Baseline — global defaults",
    "preprocessing": {
        "significance_threshold": 0.50,
        "smoothing_window": 3,
        "outlier_window": 3,
        "outlier_threshold": 0.30,
        "peak_merge_distance": 3,
        "min_clean_months": 3,
    },
    "fitting": {
        "d_min": 0.05,
        "di_initial": 0.50,
        "b_initial": 0.50,
        "qi_guess_strategy": "first",
        "qi_multiplier_upper": 5.0,
        "di_upper_bound": 0.99,
        "b_upper_bound": 2.0,
    },
}


def select_wells(con, count=50):
    """Select NM wells with first oil production in 2023 and 18+ months history."""
    rows = con.execute("""
        WITH candidates AS (
            SELECT
                entity_key,
                MIN(production_date) AS first_prod,
                COUNT(*) AS months,
                AVG(oil_bbl) AS avg_oil,
                MAX(oil_bbl) AS peak_oil
            FROM production_monthly
            WHERE state = 'NM'
              AND oil_bbl > 0
            GROUP BY entity_key
            HAVING EXTRACT(YEAR FROM MIN(production_date)) = 2023
               AND COUNT(*) >= 18
        )
        SELECT entity_key, months, avg_oil, peak_oil
        FROM candidates
        ORDER BY months DESC, avg_oil DESC
        LIMIT ?
    """, [count]).fetchall()

    print(f"  Selected {len(rows)} NM wells (2023 vintage, 18+ months)")
    return rows


def populate_atom(con, entity_key):
    """Create one well's atom directory. Returns month count."""
    atom_dir = WELLS_DIR / entity_key
    atom_dir.mkdir(parents=True, exist_ok=True)

    rows = con.execute("""
        SELECT production_date, oil_bbl, gas_mcf, water_bbl
        FROM production_monthly
        WHERE entity_key = ?
        ORDER BY production_date
    """, [entity_key]).fetchall()

    with open(atom_dir / "production.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "oil_bbl", "gas_mcf", "water_bbl"])
        for row in rows:
            dt = row[0]
            date_str = f"{dt.year:04d}-{dt.month:02d}"
            oil = row[1] if row[1] is not None else 0.0
            gas = row[2] if row[2] is not None else 0.0
            water = row[3] if row[3] is not None else 0.0
            writer.writerow([date_str, oil, gas, water])

    params = {**BASELINE_PARAMS, "well_type": "nm_2023", "state": "NM", "vintage": 2023}
    (atom_dir / "params.json").write_text(json.dumps(params, indent=2))
    (atom_dir / "trace.jsonl").write_text("")
    (atom_dir / "hindcast.json").write_text("[]")

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Setup NM 2023 wells")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    count = 999999 if args.all else args.count

    con = duckdb.connect(WAREHOUSE_PATH, read_only=True)
    wells = select_wells(con, count)

    WELLS_DIR.mkdir(exist_ok=True)

    print(f"Populating {len(wells)} atoms...")
    for entity_key, months, avg_oil, peak_oil in wells:
        n = populate_atom(con, entity_key)
        print(f"  {entity_key} — {n} months, avg {avg_oil:.0f} bbl, peak {peak_oil:.0f} bbl")

    con.close()
    print(f"Done. {len(wells)} wells in {WELLS_DIR}/")


if __name__ == "__main__":
    main()
