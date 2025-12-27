from src.process.load_raw import ProcessRaw
from src.process.clean_data import DataCleaner, DataCleanerConfig
from src.utils.utils import data_spliter, save_dataframe_parquet
from src.config.settings import train_test_split_ratio, data_split_cutoff
from src.utils.custom_exception import CustomException
from src.utils.custom_logger import get_logger
from src.config.settings import DATA_TRAIN_PATH, DATA_TEST_PATH, DATA_VALIDATION_PATH
from src.config.settings import DATA_TRAIN_PATH_WITH_FEATURES, DATA_TEST_PATH_WITH_FEATURES, DATA_VALIDATION_PATH_WITH_FEATURES
import pandas as pd
from src.config.settings import DATA_INTERIM_PATH

from src.process.features import FeaturesCreation

logger = get_logger(__name__)

def data_pipeline(skip_concat = False):


    try:

        if skip_concat:
            logger.info("data_pipeline.py started")
            pr = ProcessRaw()

            df = pr.concat()
            
            pr.save(df, filename="interim.parquet", fmt="parquet", target="interim")
            
            # Run data cleaning (skeleton; no destructive logic by default)
            cfg = DataCleanerConfig(required_columns=["fund_cnpj", "report_date"])
            dc = DataCleaner(config=cfg)
            cleaned = dc.run(df, save=True, filename="cleaned.parquet", fmt="parquet")
        else:
            cleaned = pd.read_parquet(DATA_INTERIM_PATH)


        # splitting the data
        train_df, test_df, val_df = data_spliter(cleaned, data_split_cutoff, train_test_split_ratio)

        # saving the data
        save_dataframe_parquet(train_df, DATA_TRAIN_PATH)
        save_dataframe_parquet(test_df, DATA_TEST_PATH)
        save_dataframe_parquet(val_df, DATA_VALIDATION_PATH)

        features_creation_train = FeaturesCreation(train_df)
        features_creation_test = FeaturesCreation(test_df)
        features_creation_val = FeaturesCreation(val_df)

        features_creation_train.run()
        features_creation_test.run()
        features_creation_val.run()

        train_df_features = features_creation_train.aggregate_features()
        test_df_features = features_creation_test.aggregate_features()
        val_df_features = features_creation_val.aggregate_features()


        # saving the data after feature eng
        save_dataframe_parquet(train_df_features, DATA_TRAIN_PATH_WITH_FEATURES, index=True)
        save_dataframe_parquet(test_df_features, DATA_TEST_PATH_WITH_FEATURES, index=True)
        save_dataframe_parquet(val_df_features, DATA_VALIDATION_PATH_WITH_FEATURES, index=True)
        

    except Exception as e:
        CustomException(f"data_pipeline failed : {e}")