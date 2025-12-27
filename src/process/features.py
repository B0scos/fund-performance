from __future__ import annotations


import pandas as pd

from src.utils.custom_logger import get_logger

import numpy as np
from functools import partial

logger = get_logger(__name__)


class FeaturesCreation:
    """
    Build financial features for funds dataframe.
    Assumes dataframe contains:
        - 'report_date'
        - 'fund_cnpj'
        - 'quota_value'
        - 'total_value'
        - 'net_asset_value'
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.sort_values("report_date")

    def run(self) -> pd.DataFrame:
        logger.info("Starting FeaturesCreation.run()")

        self._add_return('quota_value')
        self._add_gross_by_net()
        self._add_vol_std([5, 10, 15, 22, 50], "return")
        self._add_drawdown('quota_value')

        return self.df.reset_index(drop=True)
        
    def aggregate_features(self) -> pd.DataFrame:
        """
        Aggregate daily fund-level metrics into fund-level summary features.

        This method groups the dataset by ``fund_cnpj`` and computes a set of
        descriptive statistics commonly used for fund analysis. The aggregation
        captures central tendency, dispersion, risk behavior, liquidity activity,
        and investor base characteristics. After aggregation, infinite values are
        converted to NaN and any rows containing missing values are removed to
        ensure a clean feature set for downstream modeling.

        Returns
        -------
        pd.DataFrame
            A dataframe indexed by ``fund_cnpj`` containing the aggregated
            statistics for each fund. Each row represents one fund and all
            resulting columns are numeric features.

        Notes
        -----
        Metrics produced:
            - Returns: mean and standard deviation
            - Drawdowns: maximum (min value) and average
            - Liquidity: mean inflow and redemption volume
            - Exposure: mean gross/net ratio
            - Investor base: average number of shareholders
        """
        logger.info("Starting FeaturesCreation.aggregate()")

        agg_features = self.df.groupby("fund_cnpj").agg(
            mean_return=("return", "mean"),
            std_return=("return", "std"),
            max_drawdown=("drawdown", "min"),
            avg_drawdown=("drawdown", "mean"),
            avg_gross_by_net=("gross_by_net", "mean"),
            avg_inflow=("daily_inflow", "mean"),
            avg_redemption=("daily_redemptions", "mean"),
            avg_shareholders=("num_shareholders", "mean"),
        )


        # handle NaNs and infinities from funds with insufficient history
        agg_features = agg_features.replace([np.inf, -np.inf], np.nan).dropna()
        return agg_features

        


    def _add_return(self, col: str) -> pd.DataFrame:
        logger.info("Creating 'return' feature")
        self.df["return"] = (
            self.df.groupby("fund_cnpj")[col]
            .pct_change()
        )
        return self.df

    def _add_gross_by_net(self) -> pd.DataFrame:
        logger.info("Creating 'gross_by_net' feature")
        self.df["gross_by_net"] = self.df["total_value"] / self.df["net_asset_value"]
        return self.df

    def _add_vol_std(self, list_windows: list, col: str) -> pd.DataFrame:
        logger.info("Creating 'vol' features based on standard deviation")
        for win in list_windows:
            self.df[f"vol_{win}"] = self.df[col].rolling(win).std()
        return self.df

    def _add_drawdown(self, col: str) -> pd.DataFrame:
        logger.info("Creating 'drawdown' feature")
        peak = (
            self.df.groupby("fund_cnpj")[col]
            .transform("cummax")
        )

        self.df["drawdown"] = (self.df[col] / peak) - 1
        return self.df
