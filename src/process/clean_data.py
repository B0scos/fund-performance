from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Dict, Any
import logging
import pandas as pd

from src.utils.custom_logger import get_logger
from src.utils.custom_exception import CustomException


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
    """Backbone class for cleaning tabular data.

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
        self.logger = logger or logger

    # ---------- Public API ----------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the cleaning pipeline and return a cleaned DataFrame.

        This orchestrates validation + cleaning steps. Each step logs its
        start/end and returns the (possibly mutated) DataFrame. Because
        the concrete logic is intentionally not implemented here, the
        steps are no-ops by default.
        """
        self.logger.debug("Starting DataCleaner.run()")

        self._validate_input(df)

        df = self._drop_empty_rows(df)
        df = self._standardize_columns(df)
        df = self._impute_missing(df)
        df = self._deduplicate(df)
        df = self._flag_outliers(df)

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

    # ---------- Cleaning steps (placeholders) ----------
    def _drop_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where required columns are empty.

        TODO: Implement your logic here. This method currently returns the
        original DataFrame unchanged but logs the intended behavior.
        """
        self.logger.debug("_drop_empty_rows called (no-op)")
        # Example implementation hint (do not enable here):
        # subset = self.config.required_columns or df.columns.tolist()
        # return df.dropna(subset=subset, how='all')
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize/rename/format columns (e.g., whitespace, case).

        TODO: Implement your logic here.
        """
        self.logger.debug("_standardize_columns called (no-op)")
        return df

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values according to `self.config.impute_strategy`.

        TODO: Implement your logic here.
        """
        self.logger.debug("_impute_missing called (no-op)")
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate rows.

        Hint: use `self.config.dedupe_subset` to restrict keys.
        TODO: Implement your logic here.
        """
        self.logger.debug("_deduplicate called (no-op)")
        return df

    def _flag_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag outliers according to `self.config.outlier_params`.

        Should add boolean columns like `<col>_is_outlier` or an
        `outlier_score` column depending on your approach.

        TODO: Implement your logic here.
        """
        self.logger.debug("_flag_outliers called (no-op)")
        return df

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
