from __future__ import annotations

from typing import List

import pandas as pd

from src.utils.custom_logger import get_logger
from src.utils.custom_exception import CustomException
from pathlib import Path
from typing import List

logger = get_logger(__name__)


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Raise CustomException if any required column is missing."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise CustomException(f"Missing required columns: {missing}")

    logger.debug("Required columns present: %s", required_columns)





def generate_source_report(df: pd.DataFrame, out_dir: Path, required_columns: List[str], name: str = "data", write_csv: bool = True) -> pd.DataFrame:
    """Create a per-source report showing missing/invalid counts.

    The report includes:
      - source_file
      - total_rows
      - missing_<column> counts for each required column
      - fully_empty_rows (all required columns missing)
      - rows_with_any_missing_required
      - missing_fraction_pct

    Returns a pandas DataFrame with one row per source_file and optionally
    writes a CSV to `out_dir/<name>_source_report.csv`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if "source_file" not in df.columns:
        raise CustomException("DataFrame missing 'source_file' column; enable provenance tracking in ProcessRaw.")

    grouped = df.groupby("source_file")
    records = []
    for src, group in grouped:
        total = len(group)
        missing_counts = {f"missing_{c}": int(group[c].isna().sum()) for c in required_columns}
        fully_empty = int(group[required_columns].isna().all(axis=1).sum())
        any_missing = int(group[required_columns].isna().any(axis=1).sum())
        records.append({
            "source_file": src,
            "total_rows": total,
            **missing_counts,
            "fully_empty_rows": fully_empty,
            "rows_with_any_missing_required": any_missing,
            "missing_fraction_pct": (any_missing / total * 100.0) if total else 0.0,
        })

    report_df = pd.DataFrame.from_records(records)
    if write_csv:
        report_path = out_dir / f"{name}_source_report.csv"
        report_df.to_csv(report_path, index=False)
        logger.info("Wrote source report for %s to %s", name, report_path)

    return report_df
