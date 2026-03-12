"""Add a well from a raw CSV file.

Usage:
    python setup_from_csv.py path/to/production.csv --state NM --api 30-025-50364
    python setup_from_csv.py data/*.csv --state NM   # bulk, API inferred from filename

Expects CSV with columns: date, oil_bbl, gas_mcf, water_bbl
(or: production_date, total_oil_bbl, total_gas_mcf, water_bbl -- auto-mapped)

Creates wells/{state}/{api}/ with production.csv, params.json,
trace.jsonl, and hindcast.json.
"""

import argparse
import csv
import json
from pathlib import Path

from defaults import BASELINE_PARAMS

WELLS_DIR = Path("wells")

# Column name mappings: source -> canonical
COLUMN_MAP = {
    "production_date": "date",
    "total_oil_bbl": "oil_bbl",
    "total_gas_mcf": "gas_mcf",
}

CANONICAL_COLS = ["date", "oil_bbl", "gas_mcf", "water_bbl"]


def normalize_header(raw_fields: list[str]) -> dict[str, str]:
    """Build a mapping from raw CSV column names to canonical names."""
    return {f: COLUMN_MAP.get(f, f) for f in raw_fields}


def populate_atom(src_csv: Path, state: str, api: str) -> int:
    """Read a source CSV and create the well atom directory."""
    atom_dir = WELLS_DIR / state / api
    atom_dir.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with open(src_csv) as fin:
        reader = csv.DictReader(fin)
        col_map = normalize_header(reader.fieldnames)
        mapped_cols = set(col_map.values())

        missing = set(CANONICAL_COLS) - mapped_cols
        if missing:
            print(f"  WARNING {src_csv}: missing columns {missing}, will default to 0")

        with open(atom_dir / "production.csv", "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(CANONICAL_COLS)
            for raw_row in reader:
                row = {col_map.get(k, k): v for k, v in raw_row.items()}
                date_str = str(row.get("date", ""))[:7]  # YYYY-MM
                oil = float(row.get("oil_bbl", 0) or 0)
                gas = float(row.get("gas_mcf", 0) or 0)
                water = float(row.get("water_bbl", 0) or 0)
                writer.writerow([date_str, oil, gas, water])
                rows_written += 1

    params = {**BASELINE_PARAMS, "state": state}
    (atom_dir / "params.json").write_text(json.dumps(params, indent=2))
    (atom_dir / "trace.jsonl").write_text("")
    (atom_dir / "hindcast.json").write_text("[]")

    return rows_written


def main():
    parser = argparse.ArgumentParser(description="Add wells from CSV files")
    parser.add_argument("csvs", nargs="+", type=Path, help="CSV file(s) to import")
    parser.add_argument("--state", required=True, help="State code (NM, TX, etc.)")
    parser.add_argument("--api", default=None,
                        help="API number (required for single CSV, "
                             "omit for bulk -- inferred from filename)")
    args = parser.parse_args()

    for csv_path in args.csvs:
        if not csv_path.exists():
            print(f"  SKIP {csv_path} — not found")
            continue

        api = args.api if args.api else csv_path.stem
        months = populate_atom(csv_path, args.state, api)
        print(f"  {args.state}/{api} — {months} months")

    print("Done.")


if __name__ == "__main__":
    main()
