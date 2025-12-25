from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import DATA_RAW_UNZIP_PATH, DATA_PROCESSED_PATH, PROJECT_ROOT
from src.utils.custom_exception import raise_from_exception, CustomException
from src.utils.custom_logger import get_logger

from src.config.settings import DATA_INTERIM_PATH
from src.process.validate_data import validate_required_columns, generate_source_report

logger = get_logger(__name__)


class ProcessRaw:
    """Load and concatenate raw CSV files, then save the processed DataFrame.

    It walks `DATA_RAW_UNZIP_PATH`, reads CSV files (default sep=';'),
    concatenates them into a single DataFrame, and saves the result to
    `DATA_PROCESSED_PATH` as `processed.csv` (unless a different name is
    provided).
    """

    def __init__(self, raw_path: Optional[Path] = None, processed_path: Optional[Path] = None, interim_path: Optional[Path] = None) -> None:
        self.path_raw_data: Path = Path(raw_path) if raw_path else DATA_RAW_UNZIP_PATH
        self.path_processed_path: Path = Path(processed_path) if processed_path else DATA_PROCESSED_PATH
        self.path_interim_path: Path = Path(interim_path) if interim_path else DATA_INTERIM_PATH

        # Ensure directories exist
        self.path_processed_path.mkdir(parents=True, exist_ok=True)
        self.path_interim_path.mkdir(parents=True, exist_ok=True)

    def concat(self, sep: str = ";") -> pd.DataFrame:
        """Read all CSV files under `path_raw_data` and concatenate into a DataFrame.

        Parameters
        ----------
        sep : str
            Delimiter to use when reading CSV files.

        Returns
        -------
        pd.DataFrame
            Concatenated dataframe of all CSVs found.
        """
        try:
            df_list = []
            provenance = []
            files_found = 0
            for root, _dirs, files in os.walk(self.path_raw_data):
                for file in files:
                    # Only attempt to read CSV-like files
                    if not file.lower().endswith(".csv"):
                        continue

                    files_found += 1
                    file_path = Path(root) / file
                    # Render a path relative to project root for cleaner logs
                    try:
                        rel_path = str(file_path.relative_to(PROJECT_ROOT))
                    except Exception:
                        rel_path = str(file_path.name)

                    logger.debug("Reading raw file: %s", rel_path)

                    try:
                        df_read = pd.read_csv(file_path, sep=sep, encoding="utf-8")
                    except UnicodeDecodeError:
                        # Fallback to latin1 if utf-8 fails
                        logger.debug("utf-8 failed for %s; trying latin-1", rel_path)
                        df_read = pd.read_csv(file_path, sep=sep, encoding="latin-1")

                    # Track provenance separately (do NOT add a 'source_file' column to the dataframe)
                    provenance.append((rel_path, df_read.copy()))
                df_list.append(df_read)

            if files_found == 0:
                raise CustomException(f"No CSV files found in {self.path_raw_data}")

            df = pd.concat(df_list, ignore_index=True)
            logger.info("Concatenated %d files into dataframe with %d rows and %d columns",
                        len(df_list), len(df), len(df.columns))

            # Drop and rename domain-specific columns for consistency
            def _drop_and_rename(df: pd.DataFrame) -> pd.DataFrame:
                # Columns to drop (if present)
                drop_cols = ["TP_FUNDO_CLASSE", "ID_SUBCLASE", "ID_SUBCLASSE"]
                present_drop = [c for c in drop_cols if c in df.columns]
                if present_drop:
                    logger.info("Dropping columns: %s", present_drop)
                    df = df.drop(columns=present_drop)

                # Mapping to tidy, meaningful snake_case names (removes 'CLASSE')
                col_map = {
                    'TP_FUNDO_CLASSE': 'fund_type',
                    'CNPJ_FUNDO_CLASSE': 'fund_cnpj',
                    'ID_SUBCLASSE': 'subclass_id',
                    'DT_COMPTC': 'report_date',
                    'VL_TOTAL': 'total_value',
                    'VL_QUOTA': 'quota_value',
                    'VL_PATRIM_LIQ': 'net_asset_value',
                    'CAPTC_DIA': 'daily_inflow',
                    'RESG_DIA': 'daily_redemptions',
                    'NR_COTST': 'num_shareholders',
                }

                present_rename = {k: v for k, v in col_map.items() if k in df.columns}
                if present_rename:
                    df = df.rename(columns=present_rename)
                    logger.info("Renamed columns: %s", present_rename)

                # Ensure snake_case and no spaces
                df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
                logger.debug("Final columns after rename: %s", list(df.columns))
                return df

            df = _drop_and_rename(df)

            # Remove any mis-parsed combined-header columns (they contain commas)
            comma_cols = [c for c in df.columns if "," in c]
            if comma_cols:
                logger.warning("Found mis-parsed combined columns, dropping: %s", comma_cols)
                df = df.drop(columns=comma_cols)

            # Cast report_date (and common variants) to datetime
            for date_col in ("report_date", "dt_comptc", "data_competencia"):
                if date_col in df.columns:
                    logger.debug("Casting column %s to datetime", date_col)
                    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                    logger.info("Column %s cast to datetime; nulls after cast: %d", date_col, df[date_col].isna().sum())

            # Validate required schema columns (manifest functionality removed)
            try:
                validate_required_columns(df, ["fund_cnpj", "report_date"])
                logger.info("Validation passed for required columns")
            except CustomException as e:
                logger.warning("Schema validation issue: %s", e)

            # Generate per-source report of missing/invalid rows
            try:
                report_df = generate_source_report(df, out_dir=self.path_interim_path, required_columns=["fund_cnpj", "report_date"], name="concat", write_csv=True, per_source=provenance)
                report_path = self.path_interim_path / "concat_source_report.csv"
                logger.info("Source report saved to %s", report_path)
                # Log top offending files
                top = report_df.sort_values("rows_with_any_missing_required", ascending=False).head(5)
                for _, row in top.iterrows():
                    logger.info("Source %s: %d missing rows (%.2f%%)", row["source_file"], int(row["rows_with_any_missing_required"]), float(row["missing_fraction_pct"]))
            except Exception as e:
                logger.warning("Failed to generate source report: %s", e)

            return df

        except Exception as e:
            # Use helper to log and raise a CustomException
            raise_from_exception("Failed to concatenate raw CSV files", e)

    def save(
        self,
        df: pd.DataFrame,
        filename: str = "interim.parquet",
        fmt: str = "parquet",
        sample_csv_lines: int = 10,
        sep: str = ";",
        allow_full_csv: bool = False,
        target: str = "interim",
    ) -> Path:
        """Save dataframe in the chosen format and also write a small CSV sample.

        CSV Safety Policy
        ------------------
        For safety, the method will **never** write a full CSV file by default when
        the DataFrame is large. To force writing a full CSV, pass
        `allow_full_csv=True` (not recommended). Instead, prefer Parquet output.

        New parameter
        -------------
        target : str
            Destination directory for the output file. One of `'processed'` or
            `'interim'`. Defaults to `'interim'` to preserve historical behaviour.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to save.
        filename : str
            File name for the main output. For `parquet` this should end with
            `.parquet`; for `csv` with `.csv`.
        fmt : str
            Output format: `'parquet'` or `'csv'`.
        sample_csv_lines : int
            Number of rows to write to a sample CSV file for quick inspection in
            VSCode.
        sep : str
            Delimiter used when writing CSV sample.
        allow_full_csv : bool
            If `True`, allow writing the full CSV even when the dataframe is
            large. Default is `False`.

        Returns
        -------
        Path
            Path to the main output file.
        """
        try:
            if target not in {"processed", "interim"}:
                raise CustomException(f"Unsupported target: {target}; use 'processed' or 'interim'")

            base_dir = self.path_processed_path if target == "processed" else self.path_interim_path
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / filename
            nrows = len(df)

            if fmt == "parquet":
                # Try pyarrow then fastparquet
                try:
                    df.to_parquet(out_path, index=False, compression="snappy")
                    try:
                        out_rel = str(out_path.relative_to(PROJECT_ROOT))
                    except Exception:
                        out_rel = str(out_path.name)
                    logger.info("Saved processed dataframe as parquet to %s (pyarrow)", out_rel)
                except Exception as e1:
                    # Try fastparquet
                    try:
                        df.to_parquet(out_path, index=False, engine="fastparquet", compression="snappy")
                        try:
                            out_rel = str(out_path.relative_to(PROJECT_ROOT))
                        except Exception:
                            out_rel = str(out_path.name)
                        logger.info("Saved processed dataframe as parquet to %s (fastparquet)", out_rel)
                    except Exception as e2:
                        raise_from_exception("Failed to write parquet file; please install pyarrow or fastparquet", e2)

            elif fmt == "csv":
                # Safety: don't write big CSVs unless explicitly allowed
                if nrows > sample_csv_lines and not allow_full_csv:
                    raise CustomException(
                        f"Refusing to write full CSV ({nrows} rows)."
                        " To allow this, set `allow_full_csv=True` (not recommended)."
                    )
                df.to_csv(out_path, index=False, sep=sep, encoding="utf-8")
                logger.info("Saved processed dataframe as csv to %s", out_path)
            else:
                raise CustomException(f"Unsupported save format: {fmt}")

            # Write at most one CSV:
            # - If the main file is parquet, write a small sample CSV for quick inspection.
            # - If the main file is CSV, do NOT write an additional sample to avoid duplicates.
            if fmt == "parquet":
                # Save small sample CSV into interim (better place for inspection artifacts)
                sample_path = self.path_interim_path / f"{out_path.stem}_sample.csv"
                df.head(sample_csv_lines).to_csv(sample_path, index=False, sep=sep, encoding="utf-8")
                try:
                    sample_rel = str(sample_path.relative_to(PROJECT_ROOT))
                except Exception:
                    sample_rel = str(sample_path.name)
                logger.info("Saved sample CSV (%d rows) to %s", sample_csv_lines, sample_rel)
            else:
                logger.debug("Skipping additional sample CSV because main file is CSV")

            return out_path
        except CustomException:
            # Re-raise configuration/usage errors without wrapping
            raise
        except Exception as e:
            raise_from_exception(f"Failed to save processed dataframe to {filename}", e)
