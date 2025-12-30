import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from src.utils.custom_logger import get_logger

logger = get_logger(__name__)


class PCAWrapper:
    """
    Wrapper for performing PCA consistently on train / test / validation sets.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataframe used to fit PCA.
    test_df : pd.DataFrame
        Test dataframe transformed using the PCA fitted on train.
    val_df : pd.DataFrame
        Validation dataframe transformed using the PCA fitted on train.
    **pca_kwargs :
        Optional keyword arguments passed directly to sklearn.decomposition.PCA.
        If none are provided, PCA() uses sklearn defaults.

    Methods
    -------
    fit_transform():
        Fits PCA on train and transforms train, test and validation.
        Returns (train_pca, test_pca, val_pca)

    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame, **pca_kwargs):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.val_df = val_df.copy()

        # Schema enforcement
        if not (list(train_df.columns) == list(test_df.columns) == list(val_df.columns)):
            raise ValueError("Train / Test / Val must have identical columns in the same order.")

        # Numeric validation
        if not all(pd.api.types.is_numeric_dtype(self.train_df[col]) for col in self.train_df.columns):
            raise ValueError("PCAWrapper only supports numeric features.")

        if self.train_df.isnull().any().any():
            raise ValueError("Training dataset contains NaN values. Handle missing values first.")

        self.pca = PCA(**pca_kwargs)

    def fit_transform(self):
        logger.info(f"Fitting PCA on shape {self.train_df.shape}")

        train = self.pca.fit_transform(self.train_df)
        test  = self.pca.transform(self.test_df)
        val   = self.pca.transform(self.val_df)

        cols = [f"pca_{i+1}" for i in range(train.shape[1])]

        train_df = pd.DataFrame(train, index=self.train_df.index, columns=cols)
        test_df  = pd.DataFrame(test,  index=self.test_df.index,  columns=cols)
        val_df   = pd.DataFrame(val,   index=self.val_df.index,   columns=cols)

        explained = self.pca.explained_variance_ratio_.sum()
        logger.info(f"PCA finished. Explained variance ratio sum = {explained:.4f}")

        return train_df, test_df, val_df


class RobustScalerWrapper:
    """
    Wrapper for applying RobustScaler consistently on
    train / test / validation ensuring no data leakage.

    Parameters
    ----------
    train_df : pd.DataFrame
        Data used to fit the scaler.
    test_df : pd.DataFrame
        Data transformed using scaler fit on train.
    val_df : pd.DataFrame
        Data transformed using scaler fit on train.
    **scaler_kwargs :
        Optional keyword arguments passed to sklearn.preprocessing.RobustScaler.

    Methods
    -------
    fit_transform():
        Fits scaler on train and transforms all datasets.
        Returns (train_scaled, test_scaled, val_scaled)
    """

    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame, **scaler_kwargs):
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.val_df = val_df.copy()

        # Validate schema
        if not (list(train_df.columns) == list(test_df.columns) == list(val_df.columns)):
            raise ValueError("Train / Test / Val must have identical columns in the same order.")

        # Validate numeric
        if not all(pd.api.types.is_numeric_dtype(self.train_df[col]) for col in self.train_df.columns):
            raise ValueError("RobustScalerWrapper only supports numeric features.")

        # Validate NaN
        if self.train_df.isnull().any().any():
            raise ValueError("Training dataset contains NaN values. Handle missing values first.")

        self.scaler = RobustScaler(**scaler_kwargs)

    def fit_transform(self):
        logger.info(f"Fitting RobustScaler on data shape {self.train_df.shape}")

        train_scaled = self.scaler.fit_transform(self.train_df)
        test_scaled  = self.scaler.transform(self.test_df)
        val_scaled   = self.scaler.transform(self.val_df)

        cols = self.train_df.columns

        train_scaled = pd.DataFrame(train_scaled, index=self.train_df.index, columns=cols)
        test_scaled  = pd.DataFrame(test_scaled,  index=self.test_df.index,  columns=cols)
        val_scaled   = pd.DataFrame(val_scaled,   index=self.val_df.index,   columns=cols)

        logger.info("RobustScaler transformation complete.")
        return train_scaled, test_scaled, val_scaled
