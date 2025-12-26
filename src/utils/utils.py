import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.custom_logger import get_logger
from src.utils.custom_exception import CustomException


logger = get_logger(__name__)

def data_spliter(
    df: pd.DataFrame,
    val_cutoff: str,
    test_ratio: float = 0.2,
    date_col: str = "report_date",
    random_state: int = 42,
):
    """
    Split dataset into:
        - train
        - test (from pre-cutoff segment)
        - validation (post-cutoff segment)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    val_cutoff : str
        Cutoff date for validation split (inclusive for validation).
    test_ratio : float, optional
        Proportion of test size from pre-cutoff data.
    date_col : str, optional
        Name of the datetime column.
    random_state : int, optional
        Random seed for reproducibility.
    stratify_col : str | None
        Column name to stratify split (classification).

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    val_df : pd.DataFrame
    """

    # Ensure datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])


    # Validation split
    val_df = df[df[date_col] >= pd.to_datetime(val_cutoff)]
    train_test_df = df[df[date_col] < pd.to_datetime(val_cutoff)]

    if len(train_test_df) == 0:
        CustomException("No data left for train/test before cutoff.")



    train_df, test_df = train_test_split(
        train_test_df,
        test_size=test_ratio,
        random_state=random_state,
        shuffle=True,  # time split already handled; this is ok within pre-cutoff
    )

    return train_df, test_df, val_df