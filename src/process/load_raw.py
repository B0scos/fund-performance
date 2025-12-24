"""Utilities to load raw CSV files, concatenate them and save processed output."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.settings import DATA_RAW_UNZIP_PATH, DATA_PROCESSED_PATH
from src.utils.custom_exception import raise_from_exception, CustomException
from src.utils.custom_logger import get_logger

from src.config.settings import DATA_INTERIM_PATH

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
            files_found = 0
            for root, _dirs, files in os.walk(self.path_raw_data):
                for file in files:
                    # Only attempt to read CSV-like files
                    if not file.lower().endswith(".csv"):
                        continue

                    files_found += 1
                    file_path = Path(root) / file
                    logger.debug("Reading raw file: %s", file_path)

                    try:
                        df_read = pd.read_csv(file_path, sep=sep, encoding="utf-8")
                    except UnicodeDecodeError:
                        # Fallback to latin1 if utf-8 fails
                        logger.debug("utf-8 failed for %s; trying latin-1", file_path)
                        df_read = pd.read_csv(file_path, sep=sep, encoding="latin-1")

                    logger.info("Loaded %s rows from %s", len(df_read), file_path)
                    df_list.append(df_read)

            if files_found == 0:
                raise CustomException(f"No CSV files found in {self.path_raw_data}")

            df = pd.concat(df_list, ignore_index=True)
            logger.info("Concatenated %d files into dataframe with %d rows and %d columns",
                        len(df_list), len(df), len(df.columns))
            return df

        except Exception as e:
            # Use helper to log and raise a CustomException
            raise_from_exception("Failed to concatenate raw CSV files", e)

    def save(
        self,
        df: pd.DataFrame,
        filename: str = "processed.parquet",
        fmt: str = "parquet",
        sample_csv_lines: int = 10,
        sep: str = ";",
        allow_full_csv: bool = False,
    ) -> Path:
        """Save dataframe in the chosen format and also write a small CSV sample.

        CSV Safety Policy
        ------------------
        For safety, the method will **never** write a full CSV file by default when
        the DataFrame is large. To force writing a full CSV, pass
        `allow_full_csv=True` (not recommended). Instead, prefer Parquet output.

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
            out_path = self.path_processed_path / filename
            nrows = len(df)

            if fmt == "parquet":
                # Try pyarrow then fastparquet
                try:
                    df.to_parquet(out_path, index=False, compression="snappy")
                    logger.info("Saved processed dataframe as parquet to %s (pyarrow)", out_path)
                except Exception as e1:
                    # Try fastparquet
                    try:
                        df.to_parquet(out_path, index=False, engine="fastparquet", compression="snappy")
                        logger.info("Saved processed dataframe as parquet to %s (fastparquet)", out_path)
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
                sample_path = self.path_processed_path / f"{out_path.stem}_sample.csv"
                df.head(sample_csv_lines).to_csv(sample_path, index=False, sep=sep, encoding="utf-8")
                logger.info("Saved sample CSV (%d rows) to %s", sample_csv_lines, sample_path)
            else:
                logger.debug("Skipping additional sample CSV because main file is CSV")

            return out_path
        except CustomException:
            # Re-raise configuration/usage errors without wrapping
            raise
        except Exception as e:
            raise_from_exception(f"Failed to save processed dataframe to {filename}", e)


if __name__ == "__main__":
    pr = ProcessRaw()
    df = pr.concat()
    pr.save(df)
    logger.info("Processing complete. Output written to %s", pr.path_processed_path)


