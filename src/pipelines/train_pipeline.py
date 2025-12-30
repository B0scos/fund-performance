from src.config.settings import DATA_TRAIN_PATH_WITH_FEATURES, DATA_TEST_PATH_WITH_FEATURES, DATA_VALIDATION_PATH_WITH_FEATURES
import pandas as pd
from src.model.model import PCAWrapper, RobustScalerWrapper

def pre_processing(n_components):
    # Load data
    df_train = pd.read_parquet(DATA_TRAIN_PATH_WITH_FEATURES)
    df_test = pd.read_parquet(DATA_TEST_PATH_WITH_FEATURES)
    df_val = pd.read_parquet(DATA_VALIDATION_PATH_WITH_FEATURES)


    # pca = PCAWrapper(
    #     df_train,
    #     df_test,
    #     df_val,
    #     n_components=n_components
    # )
    # train_pca, test_pca, val_pca = pca.fit_transform()

    train_pca = df_train
    test_pca = df_test
    val_pca = df_val

    scaler = RobustScalerWrapper(train_pca, test_pca, val_pca)
    train_scaled, test_scaled, val_scaled = scaler.fit_transform()

    return train_scaled, test_scaled, val_scaled
