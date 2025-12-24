# Fund Data Ingestion Pipeline

A Python-based data ingestion and processing pipeline for fund performance data. Reads raw CSV files, validates schemas, tracks data provenance, and outputs processed Parquet files with metadata.

## Features

✅ **Robust CSV ingestion** with multi-encoding support and delimiter detection  
✅ **Data provenance tracking** — source file attribution for every row  
✅ **Schema validation** — required columns and type casting  
✅ **Per-source quality report** — missing/invalid row counts by file  
✅ **Safe output handling** — Parquet primary format + sample CSV for inspection  
✅ **Comprehensive logging** — rotating file handler + stdout output  

## Project Structure

```
.
├── main.py                          # Entry point; runs the pipeline
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py              # Path configuration & project constants
│   │
│   ├── process/
│   │   ├── load_raw.py              # ProcessRaw: CSV loading & concatenation
│   │   ├── validate_data.py         # Validation utilities & source reporting
│   │   └── clean_data.py            # DataCleaner skeleton (to be implemented)
│   │
│   └── utils/
│       ├── custom_logger.py         # Logging setup (file + stdout)
│       └── custom_exception.py      # CustomException with traceback logging
│
├── data/
│   ├── raw_unzip/                   # Input: raw CSV files (by date folder)
│   │   ├── latest_sample.csv
│   │   └── inf_diario_fi_YYYYMM/    # Monthly data folders
│   │
│   ├── interim/                     # Intermediate artifacts
│   │   ├── interim.parquet          # Concatenated + renamed data
│   │   ├── concat_source_report.csv # Per-source quality report
│   │   └── interim_sample.csv       # Sample (10 rows) for quick inspection
│   │
│   ├── processed/                   # Final cleaned output (reserved)
│   │
│   └── cache/                       # Placeholder for caching
│
├── notebooks/
│   └── 1_raw_data_checking.ipynb   # EDA & data exploration
│
└── logs/
    └── app.log                      # Application logs (rotating)
```

## Data Pipeline Flow

```
Raw CSVs (data/raw_unzip)
        ↓
    read & concatenate
        ↓
    rename columns (snake_case)
        ↓
    cast dates to datetime
        ↓
    validate required schema
        ↓
    generate per-source report
        ↓
    save as Parquet + sample CSV
        ↓
    Interim output (data/interim)
```

## Setup

### 1. Environment

```bash
# Create virtual environment (if not already done)
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit [src/config/settings.py](src/config/settings.py) if needed:
- `PROJECT_ROOT` — auto-detected from file location
- `DATA_RAW_ZIP_PATH` — where zip files are stored (for download integration)
- `DATA_RAW_UNZIP_PATH` — input folder with raw CSVs (default: `data/raw_unzip`)
- `DATA_INTERIM_PATH` — intermediate outputs (default: `data/interim`)
- `DATA_PROCESSED_PATH` — final clean outputs (default: `data/processed`)

### 3. Run the Pipeline

```bash
python main.py
```

This will:
1. Load all CSV files from `data/raw_unzip` (recursively)
2. Concatenate them into a single DataFrame (6.2M+ rows)
3. Rename columns to meaningful snake_case (e.g., `CNPJ_FUNDO_CLASSE` → `fund_cnpj`)
4. Drop non-essential columns (e.g., `TP_FUNDO_CLASSE`, `ID_SUBCLASSE`)
5. Cast date columns to datetime format
6. Validate required columns (`fund_cnpj`, `report_date`) exist
7. Generate a per-source report showing missing/invalid row counts
8. Save concatenated output as `data/interim/interim.parquet` + `interim_sample.csv`

### Logs

Application logs are written to `logs/app.log` with a rotating handler (5MB per file, 5 backups). Check there for detailed processing steps, warnings, and errors.

## Key Modules

### `src.process.load_raw` — ProcessRaw

Main ingestion class.

**Usage:**
```python
from src.process.load_raw import ProcessRaw

pr = ProcessRaw()
df = pr.concat(sep=";")  # Load & concatenate all CSVs
pr.save(df)              # Save to interim.parquet + sample CSV
```

**Methods:**
- `concat(sep=";")` → pd.DataFrame  
  Loads all CSV files, concatenates them, applies transformations (rename, cast, validate), generates source report.
  
- `save(df, filename="interim.parquet", fmt="parquet", sample_csv_lines=10, allow_full_csv=False)` → Path  
  Saves DataFrame as Parquet (primary) and writes a small CSV sample to `data/interim`. Refuses to write large CSVs unless `allow_full_csv=True` (safety policy).

### `src.process.validate_data` — Validation & Reporting

Provides schema validation and per-source quality reporting.

**Functions:**
- `validate_required_columns(df, required_columns)` → None  
  Raises `CustomException` if any required column is missing.
  
- `generate_source_report(df, out_dir, required_columns, name="data", write_csv=True)` → pd.DataFrame  
  Creates a per-source report with:
  - `total_rows` per source
  - Missing counts for each required column
  - `fully_empty_rows` (all required columns missing)
  - `rows_with_any_missing_required` (at least one required column missing)
  - `missing_fraction_pct` (percentage of rows with missing required data)
  
  Optionally writes to CSV (default: `data/interim/concat_source_report.csv`).

**Example Output (concat_source_report.csv):**
```
source_file,total_rows,missing_fund_cnpj,missing_report_date,fully_empty_rows,rows_with_any_missing_required,missing_fraction_pct
inf_diario_fi_202501.csv,560683,0,0,0,0,0.0
latest_sample.csv,1000,1000,1000,1000,1000,100.0
```

### `src.utils.custom_logger` — Logging

Sets up rotating file handler (logs/app.log) and stdout streaming.

**Usage:**
```python
from src.utils.custom_logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
```

### `src.utils.custom_exception` — Exception Handling

Custom exception class that logs full tracebacks on instantiation.

**Usage:**
```python
from src.utils.custom_exception import CustomException

if something_bad:
    raise CustomException("User-friendly error message")
```

## Current Data Issues & Notes

- **Missing dates:** ~1,000 rows have unparseable or missing `report_date` values; these are logged but not dropped (yet).
- **Mis-parsed headers:** Some files have combined column headers (e.g., `"cnpj_fundo_classe,dt_comptc,vl_quota,cnpj_fundo"` in a single cell), which are automatically detected and dropped.
- **Type warnings:** pandas warnings about mixed types in numeric columns; safe to ignore (data is coerced on read).
- **Sample file provenance:** `latest_sample.csv` contains 100% missing required fields; consider whether this is placeholder data or should be cleaned.

## Next Steps (Roadmap)

- [ ] **Drop empty rows** — Remove rows where all/most required fields are missing
- [ ] **Advanced date reparsing** — Try alternate date formats or heuristics on invalid rows
- [ ] **DataCleaner implementation** — Duplicate detection, outlier flagging, deduplication
- [ ] **Unit tests** — Add test coverage for validation, date parsing, and per-source reporting
- [ ] **CLI interface** — Add command-line flags for configuration (output format, validation level, etc.)
- [ ] **Great Expectations integration** — Declarative data quality framework (optional)

## Troubleshooting

**Issue:** "No CSV files found in..."  
→ Ensure raw CSV files exist in `data/raw_unzip`.

**Issue:** "Missing required columns"  
→ Check logs and `concat_source_report.csv` to identify which source files have missing `fund_cnpj` or `report_date`.

**Issue:** Large memory usage  
→ Pipeline reads all files and concatenates in memory; consider chunking or streaming for very large datasets.

**Issue:** Encoding errors  
→ The pipeline tries UTF-8 first, then falls back to latin-1; if still failing, check the raw file encoding.

## Dependencies

- **pandas** (2.3.3) — Data manipulation and CSV reading
- **fastparquet** (2025.12.0) — Parquet file I/O
- **requests** (2.32.5) — HTTP client (for future download functionality)

## License

TBD

## Contact

Project maintained by [your name/team]. For issues or questions, check the logs or open an issue.