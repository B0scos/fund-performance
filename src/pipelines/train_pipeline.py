from src.config.settings import DATA_TRAIN_PATH_WITH_FEATURES, DATA_TEST_PATH_WITH_FEATURES, DATA_VALIDATION_PATH_WITH_FEATURES
import pandas as pd
from src.model.model import PCAWrapper, RobustScalerWrapper

def preprocessing(n_components):
    df_train = pd.read_parquet(DATA_TRAIN_PATH_WITH_FEATURES)
    df_test = pd.read_parquet(DATA_TEST_PATH_WITH_FEATURES)
    df_val = pd.read_parquet(DATA_VALIDATION_PATH_WITH_FEATURES)

    robust_scaler = RobustScalerWrapper(df_train, df_test, df_val)

    train_scalled, test_scalled, val_scalled = robust_scaler.fit_transform()


    pca = PCAWrapper(train_scalled, test_scalled, val_scalled, n_components=n_components)
    

    train_pca, test_pca, val_pca = pca.fit_transform()

    print(train_pca, test_pca, val_pca)




