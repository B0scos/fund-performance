import pandas as pd
from src.models.model import PCAWrapper, RobustScalerWrapper

def pre_processing(n_components : int, df_train : pd.DataFrame, df_test : pd.DataFrame, df_val : pd.DataFrame):
    """
    Load and preprocess data with PCA and Robust Scaling.
    
    Parameters
    ----------
    n_components : int
        Number of PCA components to retain.

    df_train : pd.DataFrame
        Train data

    df_test : pd.DataFrame
        Test data
        
    df_val : pd.DataFrame
        Validation data
    
    Returns
    -------
    tuple
        (train_scaled, test_scaled, val_scaled) - Scaled DataFrames
    
    Notes
    -----
    - PCA is fitted on train data and applied to all sets
    - RobustScaler is fitted on train data and applied to all sets
    - Currently overwrites PCA results with original data (bug)
    """


    keep_cols = ['mean_return', 'median_return', 'std_return', 'avg_time_drawdown', 'sharpe', 'max_drawdown']

    # Load data
    df_train = df_train[keep_cols]
    df_test = df_test[keep_cols]
    df_val = df_val[keep_cols]


    pca = PCAWrapper(
        df_train,
        df_test,
        df_val,
        n_components=n_components
    )
    train_pca, test_pca, val_pca = pca.fit_transform()

    train_pca = df_train
    test_pca = df_test
    val_pca = df_val

    scaler = RobustScalerWrapper(train_pca, test_pca, val_pca)
    train_scaled, test_scaled, val_scaled = scaler.fit_transform()

    return train_scaled, test_scaled, val_scaled

def scalling(df_train : pd.DataFrame, df_test : pd.DataFrame, df_val : pd.DataFrame):
    keep_cols = ['mean_return', 'median_return', 'std_return', 'avg_time_drawdown', 'sharpe', 'max_drawdown']
    df_train = df_train[keep_cols]
    df_test = df_test[keep_cols]
    df_val = df_val[keep_cols]

    scaler = RobustScalerWrapper(df_train, df_test, df_val)

    train_scaled, test_scaled, val_scaled = scaler.fit_transform()

    return train_scaled, test_scaled, val_scaled