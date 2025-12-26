import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

from src.utils.custom_logger import get_logger
from src.utils.custom_exception import CustomException
from pathlib import Path


logger = get_logger(__name__)


def data_spliter(
    df: pd.DataFrame,
    val_cutoff: str,
    test_ratio: float = 0.2,
    date_col: str = "report_date",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into:
        - train (subset of data < cutoff)
        - test  (subset of data < cutoff)
        - validation (data >= cutoff)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    val_cutoff : str
        Cutoff date for validation (inclusive for validation subset).
    test_ratio : float, optional
        Proportion of pre-cutoff data allocated to test set.
    date_col : str, optional
        Name of the datetime column.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, test_df, val_df)

    Raises
    ------
    CustomException
        If date column is missing, cutoff filtering fails,
        or no data remains for train/test split.
    """

    try:
        logger.info("Starting dataset split process...")

        df = df.copy()

        if date_col not in df.columns:
            raise CustomException(f"Column '{date_col}' does not exist in dataframe.")


        # validation = >= cutoff
        val_df = df[df[date_col] >= val_cutoff]

        # train/test = < cutoff
        train_test_df = df[df[date_col] < val_cutoff]

        if len(train_test_df) == 0:
            raise CustomException("No data available for train/test before cutoff.")

        train_df, test_df = train_test_split(
            train_test_df,
            test_size=test_ratio,
            random_state=random_state,
            shuffle=True,  # safe; temporal separation already enforced
        )

        logger.info(
            f"Split complete | train={len(train_df)} | test={len(test_df)} | val={len(val_df)}"
        )

        return train_df, test_df, val_df

    except Exception as e:
        logger.exception("Error occurred during dataset splitting.")
        raise CustomException(f"Dataset split failed: {str(e)}") from e


def save_dataframe_parquet(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
) -> None:
    """
    Save a pandas DataFrame as a Parquet file.
    Ensures the directory exists; creates it if necessary.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be saved.
    path : str | Path
        Full output file path (.parquet) or directory path.
    index : bool, optional
        Whether to write the DataFrame index.

    Returns
    -------
    None

    Raises
    ------
    CustomException
        If saving fails, path is invalid, or dataframe is empty.
    """
    try:
        if df is None or df.empty:
            raise CustomException("Cannot save empty or None DataFrame.")

        output_path = Path(path)

        # If user passes a directory instead of file, fail loudly.
        if output_path.is_dir():
            raise CustomException(
                f"Provided path is a directory. You must include a filename: {output_path}"
            )

        if output_path.suffix.lower() != ".parquet":
            raise CustomException(
                f"Output path must end with .parquet, got '{output_path.suffix}'"
            )

        # Ensure parent folder exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving DataFrame to parquet: {output_path}")

        df.to_parquet(output_path, index=index)

        logger.info(
            f"Parquet save successful | rows={len(df)} | path='{output_path.resolve()}'"
        )

    except Exception as e:
        logger.exception("Failed to save DataFrame as parquet.")
        raise CustomException(f"Failed to save parquet: {str(e)}") from e