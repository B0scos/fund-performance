from src.process.load_raw import ProcessRaw
from src.process.clean_data import DataCleaner, DataCleanerConfig
from src.utils.utils import data_spliter, save_dataframe_parquet
from src.config.settings import train_test_split_ratio, data_split_cutoff
from src.utils.custom_exception import CustomException
from src.utils.custom_logger import get_logger
from src.config.settings import DATA_TRAIN_PATH, DATA_TEST_PATH, DATA_VALIDATION_PATH


logger = get_logger(__name__)

def data_pipeline():


    try:
        logger.info("")
        pr = ProcessRaw()

        df = pr.concat()
        
        pr.save(df, filename="interim.parquet", fmt="parquet", target="interim")
        
        # Run data cleaning (skeleton; no destructive logic by default)
        cfg = DataCleanerConfig(required_columns=["fund_cnpj", "report_date"])
        dc = DataCleaner(config=cfg)
        cleaned = dc.run(df, save=True, filename="cleaned.parquet", fmt="parquet")


        train_df, test_df, val_df = data_spliter(cleaned, data_split_cutoff, train_test_split_ratio)

        save_dataframe_parquet(train_df, DATA_TRAIN_PATH)
        save_dataframe_parquet(test_df, DATA_TEST_PATH)
        save_dataframe_parquet(val_df, DATA_VALIDATION_PATH)



    except Exception as e:
        CustomException(f"data_pipeline failed : {e}")