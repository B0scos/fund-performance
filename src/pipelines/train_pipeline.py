from sklearn.cluster import KMeans
import pandas as pd
from src.utils.custom_logger import get_logger

logger = get_logger(__name__)


def train_pipeline(train_df : pd.DataFrame, test_df : pd.DataFrame, model):

    logger.info("Starting the train_pipeline")

    model = KMeans()
        
    train_clusters = model.fit_predict(train_df)
    test_clusters = model.predict(test_df)

