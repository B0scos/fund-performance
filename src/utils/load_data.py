from src.config.settings import (
DATA_TRAIN_PATH_WITH_FEATURES,
DATA_TEST_PATH_WITH_FEATURES,
DATA_VALIDATION_PATH_WITH_FEATURES)
import pandas as pd

def load_data_with_features():
    """
    
    Loads the feature ready data
    from the pathes

    Returns:
        df_train, df_test, df_val
    
    """

    df_train = pd.read_parquet(DATA_TRAIN_PATH_WITH_FEATURES).dropna()
    df_test = pd.read_parquet(DATA_TEST_PATH_WITH_FEATURES).dropna()
    df_val = pd.read_parquet(DATA_VALIDATION_PATH_WITH_FEATURES).dropna()

    return df_train, df_test, df_val