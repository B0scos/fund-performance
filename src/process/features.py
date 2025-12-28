from __future__ import annotations

import pandas as pd
import numpy as np
from functools import partial

from src.utils.custom_logger import get_logger

logger = get_logger(__name__)

def calculate_correlation_features(group: pd.DataFrame) -> pd.Series:
    """
    Calculate correlation features from a group DataFrame.
    
    Parameters
    ----------
    group : pd.DataFrame
        A pandas DataFrame containing data for a single fund.
    
    Returns
    -------
    pd.Series
        A pandas Series with correlation features.
    """
    features = {}
    
    # Calculate correlations only if we have at least 2 observations
    if len(group) > 1:
        # Return vs volatility correlations
        features['corr_return_vol5'] = group['return'].corr(group['vol_5'])
        features['corr_return_vol10'] = group['return'].corr(group['vol_10'])
        features['corr_return_vol15'] = group['return'].corr(group['vol_15'])
        # features['auto_corr_1'] = group['return'].autocorr(1)
        # features['auto_corr_2'] = group['return'].autocorr(2)
        # features['auto_corr_5'] = group['return'].autocorr(5)
        

    else:
        # For groups with only 1 observation, set correlations to NaN
        features['corr_return_vol5'] = np.nan
        features['corr_return_vol10'] = np.nan
        features['corr_return_vol15'] = np.nan

    
    return pd.Series(features)

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
        self._add_vol_std([5, 10, 15], "return")
        self._add_drawdown('quota_value')
        self._add_time_in_drawdown('drawdown')

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

        # Calculate basic aggregates
        agg_features = self.df.groupby("fund_cnpj").agg(
            ## features based on returns
            mean_return=("return", "mean"),
            median_return=("return", "median"),
            std_return=("return", "std"),
            skew_return=("return", "skew"),
            kurt_return=("return", pd.Series.kurt),
        
            ## features based on drawdown
            max_drawdown=("drawdown", "min"),
            avg_drawdown=("drawdown", "mean"),
            avg_time_drawdown=('time_in_drawdown', 'mean'),
            max_time_drawdown=('time_in_drawdown', 'max'),

            ## features based on funds
            avg_gross_by_net=("gross_by_net", "mean"),
            avg_inflow=("daily_inflow", "mean"),
            avg_redemption=("daily_redemptions", "mean"),
            avg_shareholders=("num_shareholders", "mean"),

            ## features based on SD
            mean_std_5=('vol_5', 'mean'),
            mean_std_10=('vol_10', 'mean'),
            mean_std_15=('vol_15', 'mean'),
            std_std_5=('vol_5', 'std'),
            std_std_10=('vol_10', 'std'),
            std_std_15=('vol_15', 'std'),
        )

        # Calculate correlation features separately
        logger.info("Calculating correlation features")
        corr_features = self.df.groupby("fund_cnpj").apply(calculate_correlation_features)
        
        # Merge correlation features with basic aggregates
        agg_features = pd.concat([agg_features, corr_features], axis=1)

        # Calculate derived ratios
        logger.info("Calculating derived ratios")
        agg_features['sharpe'] = agg_features['mean_return'] / agg_features['std_return']
        agg_features['sharpe_mean_Std_5'] = agg_features['mean_return'] / agg_features['mean_std_5']
        agg_features['ret_by_DD'] = agg_features['mean_return'] / agg_features['avg_drawdown']
        agg_features['ret_by_timedd'] = agg_features['mean_return'] / agg_features['avg_time_drawdown']
        agg_features['ret_by_timedd_max'] = agg_features['mean_return'] / agg_features['max_time_drawdown']

        # handle NaNs and infinities from funds with insufficient history
        logger.info("Cleaning data: replacing infinities with NaN")
        agg_features = agg_features.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Data shape before dropping NaN: {agg_features.shape}")
        agg_features = agg_features.dropna()
        logger.info(f"Data shape after dropping NaN: {agg_features.shape}")
        
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

        self.df["drawdown"] = (peak / self.df[col]) - 1
        return self.df

    def _add_time_in_drawdown(self, col: str) -> pd.DataFrame:
        """
        Calculate consecutive days in drawdown for each fund.
        
        A drawdown is when current value is below previous peak (drawdown > 0).
        This function counts consecutive days where drawdown > 0 for each fund.
        """
        logger.info("Creating 'time_in_drawdown' feature")
        
        def calculate_consecutive_drawdown(group_drawdown: pd.Series) -> pd.Series:
            """Calculate consecutive days in drawdown for a single fund."""
            # Initialize result array
            result = np.zeros(len(group_drawdown))
            
            # Track current consecutive count
            current_streak = 0
            
            for i in range(len(group_drawdown)):
                if group_drawdown.iloc[i] > 0:  # In drawdown
                    current_streak += 1
                    result[i] = current_streak
                else:  # Not in drawdown
                    current_streak = 0
                    result[i] = 0
            
            return pd.Series(result, index=group_drawdown.index)
        
        # Apply to each fund group
        self.df['time_in_drawdown'] = (
            self.df.groupby('fund_cnpj')[col]
            .apply(calculate_consecutive_drawdown)
            .reset_index(level=0, drop=True)
        )
        
        return self.df