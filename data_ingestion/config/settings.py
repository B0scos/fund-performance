"""
Centralized configuration settings for the CVM pipeline.
All paths and settings are defined here.
"""

from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT.parent / "data" 
LOGS_DIR = PROJECT_ROOT / "logs"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_UNZIP_DIR = DATA_DIR / "raw_unzip"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
STATE_FILE = DATA_DIR / "pipeline_state.json"

# Fund catalog
FUND_CATALOG_PATH = DATA_DIR / "fund_catalog.parquet"

# Logging
LOG_FILE_FORMAT = "cvm_pipeline_{timestamp}.log"
LOG_ROTATION_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# Download settings
MAX_WORKERS = 4
DOWNLOAD_TIMEOUT = 60
DOWNLOAD_RETRIES = 3

# Processing settings
CHUNK_SIZE = 10000  # For chunked processing
DEFAULT_DATE_RANGE = {
    "start": "2020-01-01",
    "end": "2025-12-31"
}

# Create directories
for directory in [DATA_DIR, LOGS_DIR, RAW_DATA_DIR, RAW_UNZIP_DIR, PROCESSED_DATA_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)