from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd

from src.utils.custom_logger import get_logger
from src.utils.custom_exception import CustomException
from pathlib import Path

logger = get_logger(__name__)


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Raise CustomException if any required column is missing."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.error("Missing required columns: %s", missing)
        raise CustomException(f"Missing required columns: {missing}")

    logger.debug("Required columns present: %s", required_columns)





def generate_source_report(df: pd.DataFrame, out_dir: Path, required_columns: List[str], name: str = "data", write_csv: bool = True, per_source: Optional[List[Tuple[str, pd.DataFrame]]] = None) -> pd.DataFrame:
    """Create a per-source report showing missing/invalid counts.

    The report includes:
      - source_file
      - total_rows
      - missing_<column> counts for each required column
      - fully_empty_rows (all required columns missing)
      - rows_with_any_missing_required
      - missing_fraction_pct

    The function accepts either:
      - A dataframe that contains a `source_file` column (legacy behavior), or
      - A `per_source` sequence of (source_file_path, dataframe) tuples computed
        by the caller (preferred when provenance should not be embedded in the
        large concatenated dataframe).

    Returns a pandas DataFrame with one row per source_file and optionally
    writes a CSV to `out_dir/<name>_source_report.csv`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []

    # If per_source is provided, compute per-file stats from that list. This
    # avoids requiring a 'source_file' column injected into the concatenated
    # dataframe (keeping the DataFrame clean from provenance metadata).
    if per_source is not None:
        for src, group in per_source:
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
    else:
        if "source_file" not in df.columns:
            raise CustomException("DataFrame missing 'source_file' column; enable provenance tracking in ProcessRaw.")

        grouped = df.groupby("source_file")
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
