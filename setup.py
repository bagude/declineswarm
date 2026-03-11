"""Populate well atom directories from warehouse.duckdb.

Run once before the swarm:
    python setup.py

Creates wells/{entity_key}/ with production.csv, params.json,
trace.jsonl, and hindcast.json for 50 stratified wells.
"""

import csv
import json
from pathlib import Path

import duckdb

WAREHOUSE_PATH = "../data-warehousers/data/warehouse.duckdb"
WELLS_DIR = Path("wells")

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


def select_wells(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
    """Select 50 wells stratified by production characteristics.

    Returns list of (entity_key, well_type_label) tuples.
    """
    # Build per-well summary stats
    con.execute("""
        CREATE TEMP TABLE well_stats AS
        WITH numbered AS (
            SELECT
                entity_key,
                well_type,
                total_oil_bbl,
                production_date,
                ROW_NUMBER() OVER (PARTITION BY entity_key ORDER BY production_date) AS month_num,
                COUNT(*) OVER (PARTITION BY entity_key) AS total_months
            FROM production_monthly
            WHERE well_type = 'OIL'
              AND total_oil_bbl IS NOT NULL
        ),
        stats AS (
            SELECT
                entity_key,
                total_months,
                AVG(total_oil_bbl) AS avg_oil,
                MAX(CASE WHEN month_num = 1 THEN total_oil_bbl END) AS month1_oil,
                MAX(CASE WHEN month_num = 12 THEN total_oil_bbl END) AS month12_oil,
                AVG(CASE WHEN month_num <= 12 THEN total_oil_bbl END) AS avg_first_year
            FROM numbered
            GROUP BY entity_key, total_months
        )
        SELECT
            entity_key,
            total_months,
            avg_oil,
            month1_oil,
            month12_oil,
            avg_first_year,
            CASE
                WHEN avg_oil > 15000 AND total_months >= 18
                    THEN 'highrate'
                WHEN avg_oil < 300 AND total_months >= 24
                    THEN 'stripper'
                WHEN avg_first_year > 50
                     AND month1_oil > 0
                     AND month12_oil IS NOT NULL
                     AND month12_oil < 0.5 * month1_oil
                     AND total_months >= 24
                    THEN 'shale'
                WHEN avg_oil BETWEEN 10 AND 500
                     AND total_months >= 36
                    THEN 'conventional'
                ELSE NULL
            END AS category
        FROM stats
    """)

    selected: list[tuple[str, str]] = []
    targets = [
        ("highrate", 10),
        ("shale", 15),
        ("stripper", 10),
        ("conventional", 15),
    ]

    for category, count in targets:
        rows = con.execute(
            """SELECT entity_key FROM well_stats
               WHERE category = ?
               ORDER BY entity_key
               LIMIT ?""",
            [category, count],
        ).fetchall()
        for (ek,) in rows:
            selected.append((ek, category))
        print(f"  {category}: selected {len(rows)}/{count}")

    con.execute("DROP TABLE IF EXISTS well_stats")
    return selected


def populate_atom(
    con: duckdb.DuckDBPyConnection,
    entity_key: str,
    well_type: str,
) -> int:
    """Create one well's atom directory. Returns month count."""
    atom_dir = WELLS_DIR / entity_key
    atom_dir.mkdir(parents=True, exist_ok=True)

    # Pull production history
    rows = con.execute(
        """SELECT production_date, total_oil_bbl, total_gas_mcf, water_bbl
           FROM production_monthly
           WHERE entity_key = ?
           ORDER BY production_date""",
        [entity_key],
    ).fetchall()

    # Write production.csv
    with open(atom_dir / "production.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "oil_bbl", "gas_mcf", "water_bbl"])
        for row in rows:
            dt = row[0]
            date_str = f"{dt.year:04d}-{dt.month:02d}" if hasattr(dt, "year") else str(dt)[:7]
            oil = row[1] if row[1] is not None else 0.0
            gas = row[2] if row[2] is not None else 0.0
            water = row[3] if row[3] is not None else 0.0
            writer.writerow([date_str, oil, gas, water])

    # Write params.json
    params = {**BASELINE_PARAMS, "well_type": well_type}
    (atom_dir / "params.json").write_text(json.dumps(params, indent=2))

    # Initialize empty trace and hindcast
    (atom_dir / "trace.jsonl").write_text("")
    (atom_dir / "hindcast.json").write_text("[]")

    return len(rows)


def main():
    print(f"Connecting to warehouse: {WAREHOUSE_PATH}")
    con = duckdb.connect(WAREHOUSE_PATH, read_only=True)

    print("Selecting wells...")
    wells = select_wells(con)
    print(f"Total selected: {len(wells)}")

    if not wells:
        print("ERROR: No wells selected!")
        con.close()
        return

    WELLS_DIR.mkdir(exist_ok=True)

    print("\nPopulating atoms...")
    by_type: dict[str, list[str]] = {}
    for entity_key, well_type in wells:
        months = populate_atom(con, entity_key, well_type)
        by_type.setdefault(well_type, []).append(entity_key)
        print(f"  {entity_key} ({well_type}) — {months} months")

    con.close()

    print("\n=== Summary ===")
    for wt, keys in sorted(by_type.items()):
        print(f"  {wt}: {len(keys)} wells")
        print(f"    sample: {keys[0]}")
    print(f"  TOTAL: {len(wells)} wells in {WELLS_DIR}/")


if __name__ == "__main__":
    main()
