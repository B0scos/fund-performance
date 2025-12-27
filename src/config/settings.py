from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


## data pathes

# raw data pathes
DATA_RAW_ZIP_PATH = PROJECT_ROOT.parent / "data" / "raw_zip"
DATA_RAW_UNZIP_PATH = PROJECT_ROOT.parent / "data" / "raw_unzip"

# interim / processed path
DATA_INTERIM_PATH = PROJECT_ROOT.parent / "data" / "interim"
DATA_PROCESSED_PATH = PROJECT_ROOT.parent / "data" / "processed"

# train test split validation path
DATA_TRAIN_PATH = PROJECT_ROOT.parent / "data" / "splitted" / "train.parquet"
DATA_TEST_PATH = PROJECT_ROOT.parent / "data" / "splitted" / "test.parquet"
DATA_VALIDATION_PATH = PROJECT_ROOT.parent / "data" / "splitted" / "validation.parquet"

# train test split validation after feature eng path
DATA_TRAIN_PATH_WITH_FEATURES = PROJECT_ROOT.parent / "data" / "splitted_features" / "train.parquet"
DATA_TEST_PATH_WITH_FEATURES = PROJECT_ROOT.parent / "data" / "splitted_features" / "test.parquet"
DATA_VALIDATION_PATH_WITH_FEATURES = PROJECT_ROOT.parent / "data" / "splitted_features" / "validation.parquet"

## data spliting settings
data_split_cutoff = '2025-06-01'
train_test_split_ratio = 0.2

## cleaning option

# funds with less than this number will be removed
thresh_hold_num_shareholds = 15
