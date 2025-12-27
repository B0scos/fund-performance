from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Dict, Any
import logging
import pandas as pd

from src.utils.custom_logger import get_logger
from src.utils.custom_exception import CustomException

# Import ProcessRaw so DataCleaner can optionally persist cleaned data
from src.process.load_raw import ProcessRaw
from src.config.settings import data_split_cutoff, thresh_hold_num_shareholds


logger = get_logger(__name__)


@dataclass
class DataCleanerConfig:
    """Configuration for the DataCleaner.

    Attributes:
        required_columns: Columns that must exist in the input DataFrame.
        drop_all_empty: If True, rows that are fully empty for required
            columns should be dropped (implementation left to you).
        dedupe_subset: If provided, deduplication will use these columns.
        outlier_params: Free-form dict for outlier detection configuration.
        impute_strategy: Mapping of column -> strategy for imputations.
    """

    required_columns: List[str] = field(default_factory=list)
    drop_all_empty: bool = True
    dedupe_subset: Optional[List[str]] = None
    outlier_params: Dict[str, Any] = field(default_factory=dict)
    impute_strategy: Dict[str, Any] = field(default_factory=dict)


class DataCleaner:
    """
    Usage:
        dc = DataCleaner(config=DataCleanerConfig(required_columns=["fund_cnpj", "report_date"]))
        cleaned = dc.run(raw_df)

    Extend/override any of the protected methods to implement behavior:
        - _validate_input
        - _drop_empty_rows
        - _deduplicate
        - _flag_outliers
        - _impute_missing
        - _standardize_columns

    The public `run` method executes the pipeline in order and returns
    the cleaned DataFrame. Currently all steps are implemented as
    no-ops and simply return the input DataFrame so you can implement
    logic incrementally.
    """

    def __init__(self, config: Optional[DataCleanerConfig] = None, logger: Optional[logging.Logger] = None):
        self.config = config or DataCleanerConfig()
        # Use the provided logger or fall back to the module logger created by get_logger
        self.logger = logger or get_logger(__name__)

    
    def run(
        self,
        df: pd.DataFrame,
        save: bool = False,
        filename: str = "cleaned.parquet",
        fmt: str = "parquet",
        allow_full_csv: bool = False,
    ) -> pd.DataFrame:
        """Run the cleaning pipeline and return a cleaned DataFrame.

        Optional persistence
        --------------------
        If `save=True`, the cleaned DataFrame will be saved to
        `data/processed` using the existing `ProcessRaw.save` helper.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe to clean.
        save : bool
            If True, save the cleaned DataFrame to `data/processed`.
        filename : str
            Filename to write (defaults to `cleaned.parquet`).
        fmt : str
            Output format: 'parquet' or 'csv'.
        allow_full_csv : bool
            Only relevant for CSV writes; allow writing full CSV if True.
        """
        self.logger.debug("Starting DataCleaner.run()")

        self._validate_input(df)

        df = self._drop_empty_rows(df)
        df = self._deduplicate(df)
        df = self._filter_min_shareholders_pre_cutoff(df)
        df = self._flag_outliers(df)

        # Optionally persist the cleaned dataframe into data/processed
        if save:
            self.logger.debug("Saving cleaned dataframe to processed: %s (fmt=%s)", filename, fmt)
            try:
                pr = ProcessRaw()
                out_path = pr.save(df, filename=filename, fmt=fmt, allow_full_csv=allow_full_csv, target="processed")
                try:
                    rel = str(out_path.relative_to(pr.path_processed_path.parent))
                except Exception:
                    rel = str(out_path)
                self.logger.info("Cleaned dataframe saved to %s", rel)
            except Exception as exc:
                # Wrap in CustomException to keep consistent behaviour
                self.logger.error("Failed to save cleaned dataframe: %s", exc)
                raise CustomException("Failed to save cleaned dataframe") from exc

        self.logger.debug("Finished DataCleaner.run()")
        return df

    # ---------- Validation ----------
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate the input DataFrame.

        - Ensures `df` is a pandas DataFrame
        - Verifies required columns exist (if any configured)
        """
        if not isinstance(df, pd.DataFrame):
            raise CustomException("Input must be a pandas DataFrame")

        missing = [c for c in self.config.required_columns if c not in df.columns]
        if missing:
            msg = f"Missing required columns: {missing}"
            self.logger.error(msg)
            raise CustomException(msg)

    def _drop_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where required columns are empty.

        TODO: Implement your logic here. This method currently returns the
        original DataFrame unchanged but logs the intended behavior.
        """
        self.logger.debug("_drop_empty_rows called (no-op)")
        subset = self.config.required_columns or df.columns.tolist()
        return df.dropna(subset=subset, how='all')


    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates rows.
        """
        self.logger.debug("_deduplicate called (no-op)")
        return df.drop_duplicates()

    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag outliers according to `self.config.outlier_params`.

        Should add boolean columns like `<col>_is_outlier` or an
        `outlier_score` column depending on your approach.

        TODO: Implement your logic here.
        """
        self.logger.debug("_flag_outliers called (no-op)")
        return df
    
    def _filter_min_shareholders_pre_cutoff(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove a fund entirely if:
        1) It ever had <= thresh_hold_num_shareholds shareholders on any date
        earlier than data_split_cutoff; OR
        2) The fund was created after data_split_cutoff
        (i.e. it has no observations before cutoff).
        """
        try:
            required = {"fund_cnpj", "num_shareholders", "report_date"}
            if not required.issubset(df.columns):
                return df

            cutoff = pd.to_datetime(data_split_cutoff)
            df["report_date"] = pd.to_datetime(df["report_date"])

            # funds that violate shareholder rule before cutoff
            bad_low_shareholders = (
                df.loc[
                    (df["report_date"] < cutoff)
                    & (df["num_shareholders"] <= thresh_hold_num_shareholds),
                    "fund_cnpj",
                ]
                .unique()
            )

            # funds born after cutoff (no records before cutoff)
            fund_first_dates = df.groupby("fund_cnpj")["report_date"].min()
            born_after_cutoff = fund_first_dates[fund_first_dates >= cutoff].index

            bad_cnpjs = set(bad_low_shareholders) | set(born_after_cutoff)

            filtered = df[~df["fund_cnpj"].isin(bad_cnpjs)]

            self.logger.info(
                "Removed %d funds violating shareholder rule or born after cutoff %s",
                len(bad_cnpjs),
                cutoff,
            )

            return filtered

        except Exception as exc:
            self.logger.error("Failed shareholder cutoff filter: %s", exc)
            raise CustomException("Error filtering funds by shareholder rule") from exc
        
    # ---------- Utilities ----------
    @staticmethod
    def summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Return a small summary dictionary useful for logging or manifests.

        Example output:
            {"rows": 123, "columns": 10, "nulls_per_col": {...}}
        """
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "nulls_per_col": df.isnull().sum().to_dict(),
        }


__all__ = ["DataCleaner", "DataCleanerConfig"]
