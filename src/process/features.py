from __future__ import annotations

import pandas as pd
import numpy as np
from functools import partial
from typing import Optional, List, Dict, Any
import warnings

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
        # Use numpy for faster correlation calculation
        return_vals = group['return'].values
        vol_5_vals = group['vol_5'].values
        vol_10_vals = group['vol_10'].values
        vol_15_vals = group['vol_15'].values
        
        # Calculate correlations with NaN handling
        features['corr_return_vol5'] = np.corrcoef(return_vals, vol_5_vals)[0, 1] if len(return_vals) > 1 else np.nan
        features['corr_return_vol10'] = np.corrcoef(return_vals, vol_10_vals)[0, 1] if len(return_vals) > 1 else np.nan
        features['corr_return_vol15'] = np.corrcoef(return_vals, vol_15_vals)[0, 1] if len(return_vals) > 1 else np.nan
    else:
        # For groups with only 1 observation, set correlations to NaN
        features['corr_return_vol5'] = np.nan
        features['corr_return_vol10'] = np.nan
        features['corr_return_vol15'] = np.nan
    
    return pd.Series(features)


def safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = np.nan) -> pd.Series:
    """
    Safely divide two series, handling divide-by-zero and NaN cases.
    
    Parameters
    ----------
    numerator : pd.Series
        Numerator series
    denominator : pd.Series
        Denominator series
    default : float, default=np.nan
        Value to return when denominator is zero
    
    Returns
    -------
    pd.Series
        Result of safe division
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = numerator / denominator
        result[denominator == 0] = default
        return result


class FeaturesCreation:
    """
    Build financial features for funds dataframe.
    
    Assumes dataframe contains:
        - 'report_date'
        - 'fund_cnpj'
        - 'quota_value'
        - 'total_value'
        - 'net_asset_value'
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with fund data
    volatility_windows : List[int], optional
        Windows for volatility calculation, default is [5, 10, 15]
    min_periods_for_correlation : int, optional
        Minimum periods required for correlation calculation, default is 2
    
    Attributes
    ----------
    df : pd.DataFrame
        Processed dataframe with features
    original_shape : tuple
        Original shape of input dataframe
    volatility_windows : List[int]
        Windows for volatility calculation
    min_periods_for_correlation : int
        Minimum periods for correlation calculation
    """

    def __init__(self, df: pd.DataFrame, 
                 volatility_windows: Optional[List[int]] = None,
                 min_periods_for_correlation: int = 2) -> None:
        # Validate required columns exist
        required_cols = {'report_date', 'fund_cnpj', 'quota_value', 
                         'total_value', 'net_asset_value'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Validate data types
        if not pd.api.types.is_numeric_dtype(df['quota_value']):
            raise TypeError("quota_value must be numeric")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['fund_cnpj', 'report_date']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate fund-date combinations. Keeping first occurrence.")
            df = df.drop_duplicates(subset=['fund_cnpj', 'report_date'], keep='first')
        
        # Store parameters
        self.volatility_windows = volatility_windows or [5, 10, 15]
        self.min_periods_for_correlation = min_periods_for_correlation
        
        # Store original data
        self.original_shape = df.shape
        self.df = df.copy().sort_values(["fund_cnpj", "report_date"])
        
        logger.info(f"FeaturesCreation initialized with {self.original_shape} rows")

    def run(self, features_to_create: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create all features in the dataframe.
        
        Parameters
        ----------
        features_to_create : List[str], optional
            List of features to create. If None, creates all features.
            Options: ['returns', 'gross_by_net', 'volatility', 'drawdown', 'time_in_drawdown']
        
        Returns
        -------
        pd.DataFrame
            Dataframe with created features
        """
        try:
            logger.info(f"Starting feature creation on {self.original_shape}")
            
            # Define feature creation methods
            feature_methods = {
                'returns': partial(self._add_return, 'quota_value'),
                'gross_by_net': self._add_gross_by_net,
                'volatility': partial(self._add_vol_std, self.volatility_windows, "return"),
                'drawdown': partial(self._add_drawdown, 'quota_value'),
                'time_in_drawdown': partial(self._add_time_in_drawdown, 'drawdown'),
            }
            
            # Determine which features to create
            if features_to_create is None:
                features_to_create = list(feature_methods.keys())
            
            # Create features
            for feature in features_to_create:
                if feature in feature_methods:
                    logger.info(f"Creating feature: {feature}")
                    feature_methods[feature]()
                else:
                    logger.warning(f"Unknown feature: {feature}")
            
            final_shape = self.df.shape
            logger.info(f"Feature creation complete. Final shape: {final_shape}")
            
            return self.df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Feature creation failed: {e}")
            raise

    def aggregate_features(self, drop_threshold: float = 0.5) -> pd.DataFrame:
        """
        Aggregate daily fund-level metrics into fund-level summary features.
        
        Parameters
        ----------
        drop_threshold : float, default=0.5
            Maximum percentage of NaN values allowed per fund before dropping
        
        Returns
        -------
        pd.DataFrame
            A dataframe indexed by ``fund_cnpj`` containing aggregated statistics for each fund.
        """
        try:
            logger.info("Starting fund-level feature aggregation")
            
            # Calculate basic aggregates in one pass
            agg_dict = {
                ## Returns
                'mean_return': ('return', 'mean'),
                'median_return': ('return', 'median'),
                'std_return': ('return', 'std'),
                'skew_return': ('return', 'skew'),
                'kurt_return': ('return', lambda x: x.kurt()),
                
                ## Drawdown
                'max_drawdown': ('drawdown', 'max'),
                'avg_drawdown': ('drawdown', 'mean'),
                'avg_time_drawdown': ('time_in_drawdown', 'mean'),
                'max_time_drawdown': ('time_in_drawdown', 'max'),
                
                ## Funds
                'avg_gross_by_net': ('gross_by_net', 'mean'),
                'avg_inflow': ('daily_inflow', 'mean'),
                'avg_redemption': ('daily_redemptions', 'mean'),
                'avg_shareholders': ('num_shareholders', 'mean'),
                
                ## Volatility
                'mean_std_5': ('vol_5', 'mean'),
                'mean_std_10': ('vol_10', 'mean'),
                'mean_std_15': ('vol_15', 'mean'),
                'std_std_5': ('vol_5', 'std'),
                'std_std_10': ('vol_10', 'std'),
                'std_std_15': ('vol_15', 'std'),
            }
            
            agg_features = self.df.groupby("fund_cnpj").agg(**agg_dict)
            
            # Calculate correlation features
            logger.info("Calculating correlation features")
            corr_features = self.df.groupby("fund_cnpj").apply(calculate_correlation_features)
            agg_features = pd.concat([agg_features, corr_features], axis=1)
            
            # Calculate derived ratios safely
            logger.info("Calculating derived ratios")
            
            # Sharpe ratios
            agg_features['sharpe'] = safe_divide(
                agg_features['mean_return'], 
                agg_features['std_return']
            )
            agg_features['sharpe_mean_std_5'] = safe_divide(
                agg_features['mean_return'], 
                agg_features['mean_std_5']
            )
            agg_features['sharpe_mean_std_10'] = safe_divide(
                agg_features['mean_return'], 
                agg_features['mean_std_10']
            )
            agg_features['sharpe_mean_std_15'] = safe_divide(
                agg_features['mean_return'], 
                agg_features['mean_std_15']
            )
            
            # Drawdown ratios (use absolute value for drawdown)
            agg_features['ret_by_DD'] = safe_divide(
                agg_features['mean_return'], 
                agg_features['avg_drawdown'].abs()
            )
            agg_features['ret_by_max_DD'] = safe_divide(
                agg_features['mean_return'], 
                agg_features['max_drawdown'].abs()
            )
            agg_features['ret_by_timedd'] = safe_divide(
                agg_features['mean_return'], 
                agg_features['avg_time_drawdown']
            )
            agg_features['ret_by_timedd_max'] = safe_divide(
                agg_features['mean_return'], 
                agg_features['max_time_drawdown']
            )
            
            # Volatility ratios
            agg_features['volatility_ratio_5_10'] = safe_divide(
                agg_features['mean_std_5'],
                agg_features['mean_std_10']
            )
            agg_features['volatility_ratio_10_15'] = safe_divide(
                agg_features['mean_std_10'],
                agg_features['mean_std_15']
            )
            
            # Clean data
            logger.info("Cleaning aggregated data")
            
            # Replace infinities with NaN
            agg_features = agg_features.replace([np.inf, -np.inf], np.nan)
            
            # Drop funds with too many missing values
            nan_pct_per_fund = agg_features.isnull().mean(axis=1)
            funds_to_keep = nan_pct_per_fund <= drop_threshold
            
            logger.info(f"Dropping {len(funds_to_keep) - funds_to_keep.sum()} funds with >{drop_threshold:.0%} missing values")
            agg_features = agg_features[funds_to_keep]
            
            # Drop columns that are all NaN
            cols_before = agg_features.shape[1]
            agg_features = agg_features.dropna(axis=1, how='all')
            cols_dropped = cols_before - agg_features.shape[1]
            
            if cols_dropped > 0:
                logger.info(f"Dropped {cols_dropped} columns that were all NaN")
            
            logger.info(f"Final aggregated shape: {agg_features.shape}")
            
            return agg_features
            
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
            raise

    def _add_return(self, col: str) -> pd.DataFrame:
        """
        Calculate daily returns.
        
        Parameters
        ----------
        col : str
            Column name to calculate returns from
        
        Returns
        -------
        pd.DataFrame
            Updated dataframe with return column
        """
        logger.info("Creating 'return' feature")
        self.df["return"] = (
            self.df.groupby("fund_cnpj")[col]
            .pct_change()
        )
        return self.df

    def _add_gross_by_net(self) -> pd.DataFrame:
        """
        Calculate gross to net asset value ratio.
        
        Returns
        -------
        pd.DataFrame
            Updated dataframe with gross_by_net column
        """
        logger.info("Creating 'gross_by_net' feature")
        self.df["gross_by_net"] = safe_divide(
            self.df["total_value"], 
            self.df["net_asset_value"]
        )
        return self.df

    def _add_vol_std(self, list_windows: List[int], col: str) -> pd.DataFrame:
        """
        Calculate rolling volatility (standard deviation).
        
        Parameters
        ----------
        list_windows : List[int]
            List of window sizes for rolling calculation
        col : str
            Column name to calculate volatility from
        
        Returns
        -------
        pd.DataFrame
            Updated dataframe with volatility columns
        """
        logger.info(f"Creating volatility features for windows: {list_windows}")
        
        for win in list_windows:
            self.df[f"vol_{win}"] = (
                self.df.groupby("fund_cnpj")[col]
                .rolling(window=win, min_periods=max(2, win//2))
                .std()
                .reset_index(level=0, drop=True)
            )
        
        return self.df

    def _add_drawdown(self, col: str) -> pd.DataFrame:
        """
        Calculate drawdown from peak.
        
        Parameters
        ----------
        col : str
            Column name to calculate drawdown from
        
        Returns
        -------
        pd.DataFrame
            Updated dataframe with drawdown column
        """
        logger.info("Creating 'drawdown' feature")
        
        # Calculate rolling maximum (peak)
        peak = (
            self.df.groupby("fund_cnpj")[col]
            .transform(lambda x: x.expanding().max())
        )
        
        # Calculate drawdown (negative for losses)
        self.df["drawdown"] = safe_divide(peak, self.df[col]) - 1
        
        return self.df

    def _add_time_in_drawdown(self, col: str) -> pd.DataFrame:
        """
        Calculate consecutive days in drawdown for each fund.
        
        Parameters
        ----------
        col : str
            Column name indicating drawdown status
        
        Returns
        -------
        pd.DataFrame
            Updated dataframe with time_in_drawdown column
        """
        logger.info("Creating 'time_in_drawdown' feature")
        
        def vectorized_consecutive_drawdown(group: pd.Series) -> pd.Series:
            """Vectorized calculation of consecutive days in drawdown."""
            # Create boolean mask for drawdown periods
            in_dd = group > 0
            
            # Use cumulative sum with reset on False
            cum_sum = in_dd.cumsum()
            reset = (~in_dd).cumsum()
            
            # Subtract the last reset value for each group of consecutive True
            result = cum_sum - cum_sum.where(~in_dd).ffill().fillna(0)
            
            return result.astype(int)
        
        # Apply the vectorized function
        self.df['time_in_drawdown'] = (
            self.df.groupby('fund_cnpj')[col]
            .transform(vectorized_consecutive_drawdown)
        )
        
        return self.df

    @staticmethod
    def get_feature_descriptions() -> Dict[str, str]:
        """
        Returns descriptions for all created features.
        
        Returns
        -------
        Dict[str, str]
            Dictionary mapping feature names to descriptions
        """
        return {
            # Daily features
            'return': 'Daily return based on quota value (pct_change)',
            'gross_by_net': 'Ratio of total value to net asset value',
            'vol_5': '5-day rolling standard deviation of returns',
            'vol_10': '10-day rolling standard deviation of returns',
            'vol_15': '15-day rolling standard deviation of returns',
            'drawdown': 'Current drawdown from peak (positive for losses)',
            'time_in_drawdown': 'Consecutive days in drawdown',
            
            # Aggregate features
            'mean_return': 'Average daily return',
            'median_return': 'Median daily return',
            'std_return': 'Standard deviation of daily returns',
            'skew_return': 'Skewness of daily returns',
            'kurt_return': 'Kurtosis of daily returns',
            'max_drawdown': 'Maximum historical drawdown',
            'avg_drawdown': 'Average drawdown',
            'avg_time_drawdown': 'Average time spent in drawdown',
            'max_time_drawdown': 'Maximum consecutive time in drawdown',
            'avg_gross_by_net': 'Average gross to net asset ratio',
            'avg_inflow': 'Average daily inflow',
            'avg_redemption': 'Average daily redemptions',
            'avg_shareholders': 'Average number of shareholders',
            'mean_std_5': 'Average 5-day volatility',
            'mean_std_10': 'Average 10-day volatility',
            'mean_std_15': 'Average 15-day volatility',
            'std_std_5': 'Std of 5-day volatilities',
            'std_std_10': 'Std of 10-day volatilities',
            'std_std_15': 'Std of 15-day volatilities',
            'corr_return_vol5': 'Correlation between returns and 5-day volatility',
            'corr_return_vol10': 'Correlation between returns and 10-day volatility',
            'corr_return_vol15': 'Correlation between returns and 15-day volatility',
            
            # Derived ratios
            'sharpe': 'Sharpe ratio (mean_return / std_return)',
            'sharpe_mean_std_5': 'Modified Sharpe (mean_return / mean_std_5)',
            'sharpe_mean_std_10': 'Modified Sharpe (mean_return / mean_std_10)',
            'sharpe_mean_std_15': 'Modified Sharpe (mean_return / mean_std_15)',
            'ret_by_DD': 'Return to average drawdown ratio',
            'ret_by_max_DD': 'Return to maximum drawdown ratio',
            'ret_by_timedd': 'Return to average time in drawdown ratio',
            'ret_by_timedd_max': 'Return to max time in drawdown ratio',
            'volatility_ratio_5_10': 'Ratio of 5-day to 10-day volatility',
            'volatility_ratio_10_15': 'Ratio of 10-day to 15-day volatility',
        }

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics for all features.
        
        Returns
        -------
        pd.DataFrame
            Dataframe with feature summaries
        """
        if not hasattr(self, 'df') or self.df.empty:
            raise ValueError("No features created yet. Run the feature creation first.")
        
        # Get numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Create summary dataframe
        summary = pd.DataFrame(index=numeric_cols)
        summary['dtype'] = self.df[numeric_cols].dtypes
        summary['non_null_count'] = self.df[numeric_cols].count()
        summary['null_count'] = self.df[numeric_cols].isnull().sum()
        summary['null_pct'] = (summary['null_count'] / len(self.df)) * 100
        summary['unique_count'] = self.df[numeric_cols].nunique()
        
        # Add statistics for numeric columns
        summary['mean'] = self.df[numeric_cols].mean()
        summary['std'] = self.df[numeric_cols].std()
        summary['min'] = self.df[numeric_cols].min()
        summary['25%'] = self.df[numeric_cols].quantile(0.25)
        summary['50%'] = self.df[numeric_cols].quantile(0.50)
        summary['75%'] = self.df[numeric_cols].quantile(0.75)
        summary['max'] = self.df[numeric_cols].max()
        
        # Add description if available
        descriptions = self.get_feature_descriptions()
        summary['description'] = summary.index.map(lambda x: descriptions.get(x, ''))
        
        return summary.sort_values('null_pct', ascending=True)

    def get_fund_summary(self, fund_cnpj: str) -> Dict[str, Any]:
        """
        Get summary statistics for a specific fund.
        
        Parameters
        ----------
        fund_cnpj : str
            Fund identifier
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with fund statistics
        """
        if fund_cnpj not in self.df['fund_cnpj'].unique():
            raise ValueError(f"Fund {fund_cnpj} not found in data")
        
        fund_data = self.df[self.df['fund_cnpj'] == fund_cnpj].copy()
        
        summary = {
            'fund_cnpj': fund_cnpj,
            'observation_count': len(fund_data),
            'date_range': {
                'start': fund_data['report_date'].min(),
                'end': fund_data['report_date'].max(),
                'days': (fund_data['report_date'].max() - fund_data['report_date'].min()).days
            },
            'return_stats': {
                'mean': fund_data['return'].mean() if 'return' in fund_data.columns else None,
                'std': fund_data['return'].std() if 'return' in fund_data.columns else None,
                'min': fund_data['return'].min() if 'return' in fund_data.columns else None,
                'max': fund_data['return'].max() if 'return' in fund_data.columns else None,
            },
            'drawdown_stats': {
                'max_drawdown': fund_data['drawdown'].max() if 'drawdown' in fund_data.columns else None,
                'avg_drawdown': fund_data['drawdown'].mean() if 'drawdown' in fund_data.columns else None,
            } if 'drawdown' in fund_data.columns else None,
        }
        
        return summary

    def save_features(self, filepath: str, format: str = 'parquet') -> None:
        """
        Save the features dataframe to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save the file
        format : str, default='parquet'
            File format ('parquet', 'csv', or 'pickle')
        """
        if format == 'parquet':
            self.df.to_parquet(filepath, index=False)
        elif format == 'csv':
            self.df.to_csv(filepath, index=False)
        elif format == 'pickle':
            self.df.to_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Features saved to {filepath} ({format})")

    @classmethod
    def load_features(cls, filepath: str, format: str = 'parquet') -> 'FeaturesCreation':
        """
        Load features from disk and create a FeaturesCreation instance.
        
        Parameters
        ----------
        filepath : str
            Path to load the file from
        format : str, default='parquet'
            File format ('parquet', 'csv', or 'pickle')
        
        Returns
        -------
        FeaturesCreation
            Instance with loaded features
        """
        if format == 'parquet':
            df = pd.read_parquet(filepath)
        elif format == 'csv':
            df = pd.read_csv(filepath)
        elif format == 'pickle':
            df = pd.read_pickle(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Features loaded from {filepath} ({format}), shape: {df.shape}")
        return cls(df)