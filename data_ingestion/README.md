# Data Ingestion — CVM Fund Quota Pipeline

This document describes the data ingestion component of the project (code under `data_ingestion/`). It focuses on responsibilities, installation, configuration, CLI usage, and developer notes to get started with ingestion-related work.

## Overview

The data ingestion module implements downloading, updating, storing, and initial housekeeping for CVM fund quota data. The module exposes a small CLI for common operations and centralizes configuration in `data_ingestion/config/settings.py`.

Core responsibilities
- Initial historical download (bulk ingestion)
- Regular monthly updates
- Pipeline status inspection
- Cleaning data directories (raw, processed, cache)

## Project layout (data_ingestion)

- `data_ingestion/main.py` — CLI entry point
- `data_ingestion/cli/commands.py` — CLI argument parsing and command handlers
- `data_ingestion/config/settings.py` — configuration and directory definitions
- `data/` — default data directory (created automatically):
  - `raw/` — raw downloaded files
  - `processed/` — processed datasets
  - `cache/` — temporary/cache files
  - `pipeline_state.json` — pipeline state and metadata

## Requirements

A Python 3.10+ environment is recommended. Install project dependencies from the repository root:

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

Key settings are defined in `data_ingestion/config/settings.py`:

- `DATA_DIR` — base data directory (defaults to repository `data/`)
- `RAW_DATA_DIR`, `RAW_UNZIP_DIR`, `PROCESSED_DATA_DIR`, `CACHE_DIR` — subdirectories for files

`RAW_UNZIP_DIR` is where ZIP contents are extracted when using the `--extract` flag.- `STATE_FILE` — pipeline state file path (`data/pipeline_state.json`)
- `MAX_WORKERS`, `DOWNLOAD_TIMEOUT`, `DOWNLOAD_RETRIES` — downloader settings
- `DEFAULT_DATE_RANGE` — default start/end dates for the `init` command

Adjust settings in this file for custom data locations or different download settings.

## CLI Usage

The CLI is invoked via `python data_ingestion\main.py` from the repository root. The available commands are:

- `init` — Initial historical download
  - Flags: `--start`, `--end`, `--workers`, `--data-dir`, `--log-level`
  - Example: `python data_ingestion\main.py init --start 2020-01-01 --end 2025-12-31`
  - Note: The default range may download tens of gigabytes.

- `update` — Monthly update
  - Flags: `--include-current`, `--data-dir`, `--log-level`
  - Example: `python data_ingestion\main.py update --include-current`

- `status` — Check pipeline and data status
  - Flags: `--data-dir`
  - Example: `python data_ingestion\main.py status`

- `clean` — Clean up files in data directories
  - Flags: `--data-dir`, `--keep-raw`, `--keep-processed`
  - Example: `python data_ingestion\main.py clean --keep-raw`

Each command prints summaries and prompts for confirmation when potentially destructive operations are initiated.

## Operational notes

- The `init` command runs the pipeline's initial ingestion logic and will prompt before large downloads. Use `--workers` to tune concurrency. For raw-only download/unzip behavior, use the `--raw-only` flag together with `--extract` and optionally `--delete-zip` to remove ZIPs after extraction.

Important: In `--raw-only` mode the pipeline will "not" modify, cast, or concatenate any data. It will only download ZIP files into `data/raw/` and, if `--extract` is provided, extract contents exactly as-is into `data/raw_unzip/<zip_stem>/`. If `--delete-zip` is provided the original ZIP file will be deleted after successful extraction.

New: use the dedicated `fetch` command to strictly download and optionally extract files without any catalogue/processing steps. Example:

```
python data_ingestion\main.py fetch --start 2025-10-01 --end 2025-12-31 --workers 4 --extract --delete-zip
```- The `update` command runs a monthly update; add it to a scheduler (cron, Windows Task Scheduler) for automation.
- The `status` command reads `pipeline_state.json` and reports counts and timestamps for quick inspection.
- The `clean` command removes raw/processed/cache content unless `--keep-*` flags are used. It prompts before deletion.

## Development notes

- Code that performs the ingestion and processing lives in `core/` and `utils/` packages; CLI handlers call into `core.pipeline.CVMPipeline`.
- Add new CLI subcommands by extending `data_ingestion/cli/commands.py::create_parser` and adding a handler.
- Add tests under a `tests/` folder (not present yet). Unit tests should mock network calls and filesystem operations where appropriate.

## Troubleshooting

- If runs fail due to missing directories, ensure the environment has write permission and `DATA_DIR` exists (the settings file attempts to create directories on import).
- Inspect logs in the `logs/` directory created under the project root for detailed errors.
- If downloads are slow or failing frequently, try reducing `--workers` or increasing `DOWNLOAD_TIMEOUT` and `DOWNLOAD_RETRIES` in settings.

## Next steps (for contributors)

- Add end-to-end tests for initial and incremental ingestion.
- Implement more robust retry/backoff behavior in the downloader.
- Add metrics and better logging for long running downloads.

---

## Examples

### Sample `status` output

This is a representative example of what `python data_ingestion\main.py status` may print (values are illustration only):

```


### Sample `clean` interaction

When running `clean`, the CLI lists items to be removed and asks for confirmation. Example:

```
Will delete:
  - Raw files: 120 files
  - Processed data: 8 files
  - Cache: 10 files

Are you sure? (yes/no): 
```

## Configuration overrides

You can override the data directory on a per-command basis using `--data-dir`:

```
python data_ingestion\main.py init --data-dir C:\custom\data\path --start 2020-01-01 --end 2025-12-31
```

For persistent configuration changes, edit `data_ingestion/config/settings.py` (for example, set a different `DATA_DIR`, change `MAX_WORKERS`, or increase `DOWNLOAD_TIMEOUT`). Note that `settings.py` creates directories on import, so ensure the running user has write permissions to the configured paths.

## Troubleshooting checklist

- Command prints "Permission denied" or fails to create directories
  - Ensure the process has write permissions to `DATA_DIR` or set a different `--data-dir`.

- Downloads are slow or frequently time out
  - Reduce `--workers` or increase `DOWNLOAD_TIMEOUT` and `DOWNLOAD_RETRIES` in `settings.py`.

- `pipeline_state.json` is missing or empty
  - Verify the pipeline completed successfully; re-run `init` or `update` as appropriate. Check `logs/` for errors.

- `status` reports missing processed or catalog files
  - Run the processing steps (if present) or inspect the `processed/` and `raw/` directories for partial data. Re-run ingestion for affected date ranges.

- Files that use `CNPJ_FUNDO_CLASSE` instead of `CNPJ_FUNDO`
  - Some monthly files use a different column name (`CNPJ_FUNDO_CLASSE`). The processor detects and normalizes CNPJ column names automatically; if you still see issues, open the CSV header to inspect column names and report the file.

- Unexpected exceptions during runs
  - Inspect the most recent log file in `logs/` for stack traces and timestamps. Consider running the command with a higher log level (adjust `--log-level` or logging settings) and re-running.

## Logs and diagnostics

Log files are placed in the project `logs/` directory and follow the naming convention defined in `config/settings.py` (see `LOG_FILE_FORMAT`). Use the most recent log file to debug failures and to track long-running downloads.

---

This README focuses exclusively on data ingestion for the project. For other components (processing, pipelines, or deployment), see other modules in the repository.