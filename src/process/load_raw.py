from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import re
import warnings

import pandas as pd
import numpy as np

from src.config.settings import DATA_RAW_UNZIP_PATH, DATA_PROCESSED_PATH, PROJECT_ROOT, DATA_INTERIM_PATH
from src.utils.custom_exception import raise_from_exception, CustomException
from src.utils.custom_logger import get_logger
from src.process.validate_data import validate_required_columns, generate_source_report

logger = get_logger(__name__)


class ProcessRaw:
    """Load and concatenate raw CSV files, then save the processed DataFrame.

    It walks `DATA_RAW_UNZIP_PATH`, reads CSV files (default sep=';'),
    concatenates them into a single DataFrame, and saves the result to
    `DATA_PROCESSED_PATH` as `processed.csv` (unless a different name is
    provided).
    """

    def __init__(self, raw_path: Optional[Path] = None, 
                 processed_path: Optional[Path] = None, 
                 interim_path: Optional[Path] = None) -> None:
        self.path_raw_data: Path = Path(raw_path) if raw_path else DATA_RAW_UNZIP_PATH
        self.path_processed_path: Path = Path(processed_path) if processed_path else DATA_PROCESSED_PATH
        self.path_interim_path: Path = Path(interim_path) if interim_path else DATA_INTERIM_PATH

        # Ensure directories exist
        self.path_processed_path.mkdir(parents=True, exist_ok=True)
        self.path_interim_path.mkdir(parents=True, exist_ok=True)
        
        # Track statistics for reporting
        self.stats: Dict[str, Any] = {}
        
        # Define known CNPJ column patterns
        self.CNPJ_COLUMN_PATTERNS = [
            r'cnpj[_\s]*fundo[_\s]*classe',
            r'cnpj[_\s]*fundo',
            r'codigo[_\s]*fundo',
            r'id[_\s]*fundo',
            r'fundo[_\s]*cnpj'
        ]

    def concat(self, sep: str = ";", 
               sample_size: Optional[int] = None,
               random_state: int = 42) -> pd.DataFrame:
        """Read all CSV files under `path_raw_data` and concatenate into a DataFrame.

        Parameters
        ----------
        sep : str
            Delimiter to use when reading CSV files.
        sample_size : Optional[int]
            If provided, only read a sample of rows from each file (for testing).
        random_state : int
            Random state for sampling.

        Returns
        -------
        pd.DataFrame
            Concatenated dataframe of all CSVs found.
        """
        try:
            df_list = []
            provenance = []
            files_found = 0
            failed_files = []
            total_rows = 0
            
            # Reset stats
            self.stats = {
                'files_found': 0,
                'files_read': 0,
                'failed_files': [],
                'total_rows': 0,
                'columns_before': 0,
                'columns_after': 0,
                'cnpj_column_found': False,
                'missing_cnpj_rows': 0,
                'date_columns_found': []
            }
            
            # Get list of all CSV files first
            csv_files = []
            for root, _dirs, files in os.walk(self.path_raw_data):
                for file in files:
                    if file.lower().endswith(".csv"):
                        csv_files.append(Path(root) / file)
            
            self.stats['files_found'] = len(csv_files)
            
            if not csv_files:
                raise CustomException(f"No CSV files found in {self.path_raw_data}")
            
            logger.info(f"Found {len(csv_files)} CSV files to process")
            
            # Process files in batches for better memory management
            batch_size = 30
            for i in range(0, len(csv_files), batch_size):
                batch = csv_files[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(csv_files)-1)//batch_size + 1}")
                
                for file_path in batch:
                    files_found += 1
                    
                    try:
                        rel_path = str(file_path.relative_to(PROJECT_ROOT))
                    except Exception:
                        rel_path = str(file_path.name)

                    try:
                        # Read the file
                        df_read = self._read_csv_file(file_path, sep, sample_size, random_state)
                        
                        if df_read is None or df_read.empty:
                            logger.warning(f"Empty file: {rel_path}")
                            continue
                            

                        
                        # Process the file
                        df_read_processed = self._process_single_file(df_read, rel_path)
                        
                        if df_read_processed is not None:
                            provenance.append((rel_path, df_read_processed))
                            df_list.append(df_read_processed)
                            total_rows += len(df_read_processed)
                            
                            logger.debug("Successfully read %s: %d rows", rel_path, len(df_read_processed))
                            
                    except Exception as read_error:
                        failed_files.append((rel_path, str(read_error)))
                        logger.error("Failed to read %s: %s", rel_path, read_error, exc_info=True)
                        continue

            # Log summary
            self.stats['files_read'] = len(df_list)
            self.stats['failed_files'] = failed_files
            
            if not df_list:
                raise CustomException("No CSV files could be successfully read")

            # Concatenate all dataframes
            logger.info("Concatenating %d dataframes...", len(df_list))
            df = pd.concat(df_list, ignore_index=True, sort=False)
            
            self.stats['total_rows'] = len(df)
            self.stats['columns_before'] = len(df.columns)
            
            logger.info("Concatenated %d files into dataframe with %d rows and %d columns",
                        len(df_list), len(df), len(df.columns))

            # Final processing on concatenated dataframe
            df = self._final_processing(df)
            
            # Validate and check for missing CNPJ
            self._validate_and_report(df, provenance)
            
            return df

        except Exception as e:
            raise_from_exception("Failed to concatenate raw CSV files", e)

    def _read_csv_file(self, file_path: Path, sep: str, 
                       sample_size: Optional[int], 
                       random_state: int) -> Optional[pd.DataFrame]:
        """Read a single CSV file with multiple encoding fallbacks."""
        # Try multiple encoding strategies
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                # For large files, use chunking or sampling
                if sample_size:
                    # Get total rows first
                    total_rows = sum(1 for _ in open(file_path, 'r', encoding=encoding)) - 1
                    if total_rows > sample_size:
                        # Use skiprows to sample
                        skip = lambda x: x > 0 and np.random.rand() > (sample_size / total_rows)
                        df = pd.read_csv(
                            file_path, 
                            sep=sep, 
                            encoding=encoding,
                            low_memory=False,
                            on_bad_lines='skip',
                            skiprows=skip
                        )
                    else:
                        df = pd.read_csv(
                            file_path, 
                            sep=sep, 
                            encoding=encoding,
                            low_memory=False,
                            on_bad_lines='skip'
                        )
                else:
                    df = pd.read_csv(
                        file_path, 
                        sep=sep, 
                        encoding=encoding,
                        low_memory=False,
                        on_bad_lines='skip'
                    )

                
                
                logger.debug(f"Successfully read {file_path.name} with {encoding} encoding")
                return df
                
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.debug(f"Failed to read {file_path.name} with {encoding}: {e}")
                continue
        
        logger.error(f"Failed to read {file_path.name} with any encoding")
        return None

    def _process_single_file(self, df: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
        """Process a single file's dataframe."""
        try:
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            
            # Remove completely empty columns
            df = df.dropna(axis=1, how='all')
            
            if df.empty:
                return None
            
            # Apply standard column processing
            df_processed = self._standardize_columns(df.copy())
            
            # Validate required columns exist
            required_cols = self._identify_required_columns(df_processed)
            missing_req = [col for col in required_cols if col not in df_processed.columns]
            
            if missing_req:
                logger.warning(f"Missing required columns in {source_name}: {missing_req}")
                # Try to find alternatives
                df_processed = self._find_alternative_columns(df_processed)
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error processing {source_name}: {e}")
            return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and remove unnecessary columns."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # 1. Convert all column names to snake_case first
        df.columns = self._to_snake_case(df.columns)
        
        # 2. Drop clearly unnecessary columns
        columns_to_drop = [
            "tp_fundo_classe", "id_subclase", "id_subclasse", "tp_fundo",
            "unamed", "unnamed", "index", "row_id", "id"
        ]
        
        # Also drop columns that are all NaN
        nan_cols = df.columns[df.isna().all()]
        columns_to_drop.extend(nan_cols)
        
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        if columns_to_drop:
            logger.debug(f"Dropping columns: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        # 3. Find and standardize CNPJ column
        df = self._find_and_standardize_cnpj(df)
        
        # 4. Standardize other columns with flexible matching
        column_mappings = {
            'report_date': ['dt_comptc', 'data_competencia', 'data', 'date', 'dt'],
            'quota_value': ['vl_quota', 'valor_quota', 'quota', 'cotacao'],
            'total_value': ['vl_total', 'valor_total', 'total', 'montante'],
            'net_asset_value': ['vl_patrim_liq', 'patrimonio_liquido', 'pl', 'nav'],
            'daily_inflow': ['captc_dia', 'captacao_diaria', 'inflow', 'entrada'],
            'daily_redemptions': ['resg_dia', 'resgate_diario', 'redemption', 'saida'],
            'num_shareholders': ['nr_cotst', 'num_cotistas', 'cotistas', 'investidores']
        }
        
        for target_col, possible_names in column_mappings.items():
            if target_col not in df.columns:
                for possible in possible_names:
                    if possible in df.columns:
                        df = df.rename(columns={possible: target_col})
                        logger.debug(f"Renamed {possible} to {target_col}")
                        break
        
        return df

    def _find_and_standardize_cnpj(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find CNPJ column and standardize it to 'fund_cnpj'."""
        # Look for CNPJ column using patterns
        cnpj_candidates = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Direct matches
            if 'cnpj' in col_lower and 'fundo' in col_lower:
                cnpj_candidates.append((col, 1.0))  # High confidence
            elif 'cnpj' in col_lower:
                cnpj_candidates.append((col, 0.8))  # Medium confidence
            elif 'codigo' in col_lower and 'fundo' in col_lower:
                cnpj_candidates.append((col, 0.6))  # Lower confidence
        
        # Sort by confidence
        cnpj_candidates.sort(key=lambda x: x[1], reverse=True)
        
        if cnpj_candidates:
            best_cnpj_col = cnpj_candidates[0][0]
            logger.debug(f"Found CNPJ column: {best_cnpj_col} (confidence: {cnpj_candidates[0][1]})")
            
            # Clean the CNPJ values
            df['fund_cnpj'] = df[best_cnpj_col].astype(str).str.strip()
            
            # Remove non-numeric characters but preserve the CNPJ structure
            df['fund_cnpj'] = df['fund_cnpj'].str.replace(r'\D', '', regex=True)
            
            # Validate CNPJ format (should be 14 digits for Brazilian CNPJ)
            invalid_cnpj = df['fund_cnpj'].apply(
                lambda x: not (isinstance(x, str) and x.isdigit() and len(x) == 14)
            )
            
            if invalid_cnpj.any():
                invalid_count = invalid_cnpj.sum()
                logger.warning(f"Found {invalid_count} rows with invalid CNPJ format")
                
                # Show some examples
                invalid_samples = df.loc[invalid_cnpj, 'fund_cnpj'].head(5).tolist()
                logger.warning(f"Invalid CNPJ examples: {invalid_samples}")
            
            # Drop the original column if it's not already 'fund_cnpj'
            if best_cnpj_col != 'fund_cnpj' and best_cnpj_col in df.columns:
                df = df.drop(columns=[best_cnpj_col])
                
            self.stats['cnpj_column_found'] = True
            
        else:
            logger.warning("No CNPJ column found in this file")
            # Check what columns we have for debugging
            logger.debug(f"Available columns: {list(df.columns)}")
        
        return df

    def _identify_required_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify which required columns are present."""
        required = []
        
        # Always require CNPJ and date
        if 'fund_cnpj' in df.columns:
            required.append('fund_cnpj')
        
        # Check for date columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'data' in col.lower()]
        if date_cols:
            required.append(date_cols[0])
            
        return required

    def _find_alternative_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find alternative columns for missing required columns."""
        # Look for CNPJ in other column names
        if 'fund_cnpj' not in df.columns:
            for col in df.columns:
                if any(pattern in col.lower() for pattern in ['id', 'cod', 'num']):
                    # Check if this looks like an ID column
                    unique_vals = df[col].nunique()
                    total_vals = len(df[col])
                    
                    if unique_vals > 0.9 * total_vals:  # High cardinality, could be an ID
                        logger.info(f"Using {col} as fund_cnpj alternative")
                        df = df.rename(columns={col: 'fund_cnpj'})
                        break
        
        return df

    def _final_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final processing to concatenated dataframe."""
        # Remove mis-parsed combined-header columns
        comma_cols = [c for c in df.columns if "," in c]
        if comma_cols:
            logger.warning(f"Found mis-parsed combined columns, dropping: {comma_cols}")
            df = df.drop(columns=comma_cols)
        
        # Convert date columns
        date_cols_found = []
        for date_pattern in ["report_date", "dt_comptc", "data_competencia", "date", "data"]:
            matching_cols = [col for col in df.columns if date_pattern in col.lower()]
            for col in matching_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    date_cols_found.append(col)
                    nulls = df[col].isna().sum()
                    if nulls > 0:
                        logger.warning(f"Column {col}: {nulls} null values after datetime conversion")
                except Exception as e:
                    logger.warning(f"Failed to convert {col} to datetime: {e}")
        
        self.stats['date_columns_found'] = date_cols_found
        
        # Ensure consistent column order
        preferred_order = [
            'fund_cnpj', 'report_date', 'quota_value', 'total_value', 
            'net_asset_value', 'daily_inflow', 'daily_redemptions', 
            'num_shareholders'
        ]
        
        # Reorder columns if possible
        existing_cols = [col for col in preferred_order if col in df.columns]
        other_cols = [col for col in df.columns if col not in existing_cols]
        df = df[existing_cols + other_cols]
        
        self.stats['columns_after'] = len(df.columns)
        
        return df

    def _validate_and_report(self, df: pd.DataFrame, provenance: List[Tuple]) -> None:
        """Validate the final dataframe and generate reports."""
        # Check for missing CNPJ
        if 'fund_cnpj' in df.columns:
            missing_cnpj = df['fund_cnpj'].isna() | (df['fund_cnpj'] == '')
            self.stats['missing_cnpj_rows'] = missing_cnpj.sum()
            
            if missing_cnpj.any():
                logger.warning(f"Found {missing_cnpj.sum()} rows with missing or empty CNPJ")
                
                # Show some examples of rows with missing CNPJ
                missing_samples = df[missing_cnpj].head(3)
                logger.warning(f"Sample rows with missing CNPJ:\n{missing_samples}")
                
                # Try to find patterns in rows with missing CNPJ
                if 'report_date' in df.columns:
                    date_distribution = df.loc[missing_cnpj, 'report_date'].value_counts().head(5)
                    logger.warning(f"Missing CNPJ by date (top 5):\n{date_distribution}")
        else:
            logger.error("No 'fund_cnpj' column found in final dataframe!")
            logger.error(f"Available columns: {list(df.columns)}")
        
        # Validate required columns
        try:
            validate_required_columns(df, ["fund_cnpj", "report_date"])
            logger.info("Validation passed for required columns")
        except CustomException as e:
            logger.error(f"Schema validation failed: {e}")
        
        # Generate source report
        try:
            if 'fund_cnpj' in df.columns:
                report_df = generate_source_report(
                    df, 
                    out_dir=self.path_interim_path, 
                    required_columns=["fund_cnpj", "report_date"], 
                    name="concat", 
                    write_csv=True, 
                    per_source=provenance
                )
                
                if report_df is not None and not report_df.empty:
                    # Log summary
                    missing_summary = report_df[['source_file', 'rows_with_any_missing_required', 'missing_fraction_pct']]
                    logger.info("Missing data summary:\n%s", missing_summary.to_string())
                    
                    # Save detailed report
                    report_path = self.path_interim_path / "concat_detailed_report.csv"
                    report_df.to_csv(report_path, index=False)
                    logger.info(f"Detailed report saved to {report_path}")
            else:
                logger.warning("Skipping source report: 'fund_cnpj' column not found")
                
        except Exception as e:
            logger.warning(f"Failed to generate source report: {e}", exc_info=True)
        
        # Log final statistics
        self._log_final_statistics(df)

    def _log_final_statistics(self, df: pd.DataFrame) -> None:
        """Log final statistics about the processed data."""
        logger.info("=" * 50)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total files found: {self.stats['files_found']}")
        logger.info(f"Files successfully read: {self.stats['files_read']}")
        logger.info(f"Failed files: {len(self.stats['failed_files'])}")
        logger.info(f"Total rows: {self.stats['total_rows']:,}")
        logger.info(f"Total columns: {self.stats['columns_before']} -> {self.stats['columns_after']}")
        
        if 'fund_cnpj' in df.columns:
            unique_funds = df['fund_cnpj'].nunique()
            null_cnpj = df['fund_cnpj'].isna().sum()
            empty_cnpj = (df['fund_cnpj'] == '').sum()
            
            logger.info(f"Unique funds (CNPJ): {unique_funds:,}")
            logger.info(f"Rows with null CNPJ: {null_cnpj:,} ({null_cnpj/len(df)*100:.2f}%)")
            logger.info(f"Rows with empty CNPJ: {empty_cnpj:,} ({empty_cnpj/len(df)*100:.2f}%)")
            
            # Sample some CNPJ values for verification
            sample_cnpjs = df['fund_cnpj'].dropna().head(3).tolist()
            logger.info(f"Sample CNPJ values: {sample_cnpjs}")
        
        if 'report_date' in df.columns:
            date_range = f"{df['report_date'].min()} to {df['report_date'].max()}"
            logger.info(f"Date range: {date_range}")
        
        logger.info("=" * 50)

    def _to_snake_case(self, columns: pd.Index) -> List[str]:
        """Convert column names to consistent snake_case format."""
        def clean_column_name(col: str) -> str:
            # Convert to string if not already
            col = str(col)
            
            # Remove leading/trailing whitespace
            col = col.strip()
            
            # Replace multiple spaces, dots, hyphens with underscore
            col = re.sub(r'[\s\-\.]+', '_', col)
            
            # Convert to lowercase
            col = col.lower()
            
            # Remove special characters except underscore
            col = re.sub(r'[^a-z0-9_]', '', col)
            
            # Remove consecutive underscores
            col = re.sub(r'_+', '_', col)
            
            # Remove leading/trailing underscores
            col = col.strip('_')
            
            return col
        
        return [clean_column_name(col) for col in columns]

    def save(
        self,
        df: pd.DataFrame,
        filename: str = "interim.parquet",
        fmt: str = "parquet",
        sample_csv_lines: int = 10,
        sep: str = ";",
        allow_full_csv: bool = False,
        target: str = "interim",
        compression: str = "snappy"
    ) -> Path:
        """Save dataframe in the chosen format and also write a small CSV sample."""
        try:
            if target not in {"processed", "interim"}:
                raise CustomException(f"Unsupported target: {target}; use 'processed' or 'interim'")

            base_dir = self.path_processed_path if target == "processed" else self.path_interim_path
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / filename
            nrows = len(df)

            # Save main file
            if fmt == "parquet":
                try:
                    # Try with pyarrow
                    df.to_parquet(
                        out_path, 
                        index=False, 
                        compression=compression,
                        engine='pyarrow'
                    )
                    logger.info(f"Saved dataframe as parquet to {out_path} (pyarrow)")
                except Exception as e1:
                    logger.warning(f"PyArrow failed: {e1}, trying fastparquet")
                    try:
                        df.to_parquet(
                            out_path, 
                            index=False, 
                            compression=compression,
                            engine='fastparquet'
                        )
                        logger.info(f"Saved dataframe as parquet to {out_path} (fastparquet)")
                    except Exception as e2:
                        raise CustomException(f"Failed to write parquet file: {e2}")

            elif fmt == "csv":
                if nrows > sample_csv_lines and not allow_full_csv:
                    raise CustomException(
                        f"Refusing to write full CSV ({nrows} rows). "
                        "Set `allow_full_csv=True` to override (not recommended)."
                    )
                df.to_csv(out_path, index=False, sep=sep, encoding="utf-8")
                logger.info(f"Saved dataframe as CSV to {out_path}")
                
            elif fmt == "feather":
                try:
                    df.to_feather(out_path, compression=compression)
                    logger.info(f"Saved dataframe as feather to {out_path}")
                except Exception as e:
                    raise CustomException(f"Failed to write feather file: {e}")
                    
            else:
                raise CustomException(f"Unsupported save format: {fmt}")

            # Write sample CSV for inspection (if main file is not CSV)
            if fmt != "csv":
                sample_path = self.path_interim_path / f"{out_path.stem}_sample.csv"
                df.head(sample_csv_lines).to_csv(
                    sample_path, 
                    index=False, 
                    sep=sep, 
                    encoding="utf-8"
                )
                logger.info(f"Saved sample CSV ({sample_csv_lines} rows) to {sample_path}")

            # Also save a summary statistics file
            self._save_summary_stats(df, out_path)
            
            return out_path
            
        except CustomException:
            raise
        except Exception as e:
            raise_from_exception(f"Failed to save dataframe to {filename}", e)

    def _save_summary_stats(self, df: pd.DataFrame, out_path: Path) -> None:
        """Save summary statistics about the dataframe."""
        try:
            stats_path = out_path.parent / f"{out_path.stem}_summary.txt"
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("DATA SUMMARY REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"File saved: {out_path.name}\n")
                f.write(f"Total rows: {len(df):,}\n")
                f.write(f"Total columns: {len(df.columns)}\n\n")
                
                f.write("COLUMNS:\n")
                for i, col in enumerate(df.columns, 1):
                    f.write(f"  {i:2d}. {col}\n")
                f.write("\n")
                
                f.write("DATA TYPES:\n")
                dtypes = df.dtypes.astype(str).value_counts()
                for dtype, count in dtypes.items():
                    f.write(f"  {dtype}: {count} columns\n")
                f.write("\n")
                
                f.write("MISSING VALUES:\n")
                missing = df.isnull().sum()
                missing_pct = (missing / len(df) * 100).round(2)
                
                for col in df.columns:
                    if missing[col] > 0:
                        f.write(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)\n")
                
                if 'fund_cnpj' in df.columns:
                    f.write("\nCNPJ STATISTICS:\n")
                    unique_funds = df['fund_cnpj'].nunique()
                    missing_cnpj = df['fund_cnpj'].isna().sum()
                    f.write(f"  Unique funds: {unique_funds:,}\n")
                    f.write(f"  Missing CNPJ: {missing_cnpj:,}\n")
                    
                    if unique_funds > 0:
                        fund_counts = df['fund_cnpj'].value_counts().head(10)
                        f.write("\n  Top 10 funds by row count:\n")
                        for cnpj, count in fund_counts.items():
                            f.write(f"    {cnpj}: {count:,}\n")
                
                if 'report_date' in df.columns:
                    f.write("\nDATE RANGE:\n")
                    f.write(f"  From: {df['report_date'].min()}\n")
                    f.write(f"  To:   {df['report_date'].max()}\n")
            
            logger.info(f"Summary statistics saved to {stats_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save summary statistics: {e}")

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()

    def validate_cnpj_column(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate the CNPJ column.
        
        Returns:
            Tuple of (is_valid, message)
        """
        if 'fund_cnpj' not in df.columns:
            return False, "Column 'fund_cnpj' not found"
        
        # Check for missing values
        missing = df['fund_cnpj'].isna().sum()
        empty = (df['fund_cnpj'] == '').sum()
        
        if missing > 0 or empty > 0:
            return False, f"Found {missing} null and {empty} empty CNPJ values"
        
        # Check CNPJ format (Brazilian CNPJ should be 14 digits)
        invalid_format = 0
        for cnpj in df['fund_cnpj'].unique():
            if not isinstance(cnpj, str):
                invalid_format += 1
            elif not cnpj.isdigit() or len(cnpj) != 14:
                invalid_format += 1
        
        if invalid_format > 0:
            return False, f"Found {invalid_format} CNPJ values with invalid format"
        
        return True, f"CNPJ column valid: {df['fund_cnpj'].nunique():,} unique values"

    def fix_missing_cnpj(self, df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
        """
        Fix missing CNPJ values.
        
        Args:
            df: Input DataFrame
            method: One of 'drop', 'fill_forward', or 'fill_backward'
            
        Returns:
            DataFrame with fixed CNPJ
        """
        if 'fund_cnpj' not in df.columns:
            logger.error("No 'fund_cnpj' column to fix")
            return df
        
        original_len = len(df)
        
        if method == 'drop':
            # Drop rows with missing CNPJ
            df = df.dropna(subset=['fund_cnpj'])
            df = df[df['fund_cnpj'] != '']
            
        elif method in ['fill_forward', 'fill_backward']:
            # Sort by fund and date for fill operations
            if 'report_date' in df.columns:
                df = df.sort_values(['fund_cnpj', 'report_date'])
            
            if method == 'fill_forward':
                df['fund_cnpj'] = df.groupby('fund_cnpj')['fund_cnpj'].ffill()
            else:
                df['fund_cnpj'] = df.groupby('fund_cnpj')['fund_cnpj'].bfill()
        
        else:
            logger.warning(f"Unknown method: {method}. No changes made.")
            return df
        
        new_len = len(df)
        removed = original_len - new_len
        
        if removed > 0:
            logger.info(f"Removed {removed} rows with missing CNPJ ({removed/original_len*100:.1f}%)")
        
        return df