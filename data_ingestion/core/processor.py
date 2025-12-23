"""
Data processing for CVM monthly files.
"""

import pandas as pd
import zipfile
import logging
from pathlib import Path
from typing import Optional, List, Set
import warnings
warnings.filterwarnings('ignore')

from config import settings, constants

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and consolidate CVM monthly data."""
    
    def __init__(self):
        self.processed_dir = settings.PROCESSED_DATA_DIR
        self.chunk_size = settings.CHUNK_SIZE
        self.logger = logging.getLogger(__name__)
    
    def process_monthly_file(self, file_path: Path, target_cnpjs: Set[str]) -> Optional[pd.DataFrame]:
        """
        Process a single monthly ZIP file.
        
        Args:
            file_path: Path to ZIP file
            target_cnpjs: Set of CNPJs to filter for
            
        Returns:
            Filtered DataFrame or None if processing fails
        """
        try:
            self.logger.debug(f"Processing {file_path.name}")
            
            with zipfile.ZipFile(file_path, 'r') as zf:
                # Get CSV filename
                csv_name = file_path.stem + '.csv'
                
                if csv_name not in zf.namelist():
                    self.logger.error(f"CSV not found in {file_path.name}")
                    return None
                
                # Read CSV header to detect a CNPJ column name
                with zf.open(csv_name) as head_file:
                    header_line = head_file.readline().decode('utf-8', errors='replace').strip()
                    cols = [c.strip() for c in header_line.split(';')]
                    cnpj_col = None
                    for c in cols:
                        if 'cnpj' in c.lower():
                            cnpj_col = c
                            break

                    if not cnpj_col:
                        # Don't treat missing CNPJ column as an error - skip file quietly
                        self.logger.debug(f"No CNPJ-like column found in {file_path.name}; skipping file")
                        return None

                # Re-open the CSV and delegate to small/large processors with detected cnpj_col
                with zf.open(csv_name) as csv_file:
                    # Read in chunks if file is large
                    file_size = zf.getinfo(csv_name).file_size
                    try:
                        if file_size > 50 * 1024 * 1024:  # > 50 MB
                            return self._process_large_file(csv_file, target_cnpjs, cnpj_col)
                        else:
                            return self._process_small_file(csv_file, target_cnpjs, cnpj_col)
                    except ValueError as ve:
                        msg = str(ve)
                        if 'Usecols do not match columns' in msg or 'usecols' in msg.lower():
                            # Treat usecols mismatch as non-fatal and skip the file
                            self.logger.debug(f"Usecols mismatch when reading {file_path.name}: {msg}; skipping file")
                            return None
                        raise
                    except pd.errors.ParserError as pe:
                        self.logger.debug(f"ParserError when reading {file_path.name}: {pe}; skipping file")
                        return None
        except zipfile.BadZipFile:
            self.logger.error(f"Bad ZIP file: {file_path.name}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}")
            return None
    
    def _process_small_file(self, csv_file, target_cnpjs: Set[str], cnpj_col: str = 'CNPJ_FUNDO') -> pd.DataFrame:
        """Process small files by reading all at once.

        This function attempts to avoid modifying original data values. It will:
        - Read only the necessary columns (detected `cnpj_col`, `DT_COMPTC`, `VL_QUOTA`) to reduce memory usage
        - NOT perform numeric casting on `VL_QUOTA` (leave as read)
        - NOT coerce `DT_COMPTC` to datetime here (preserve original string)
        - Add a `CNPJ_FUNDO` column (string) if the detected column has a different name, for downstream compatibility
        """
        usecols = [cnpj_col, 'DT_COMPTC', 'VL_QUOTA']

        df = pd.read_csv(
            csv_file,
            sep=';',
            usecols=usecols,
            encoding='utf-8'
        )

        # Normalize a copy of the CNPJ column for downstream compatibility, without altering original columns
        if cnpj_col != 'CNPJ_FUNDO':
            df['CNPJ_FUNDO'] = df[cnpj_col].astype(str)
        else:
            df['CNPJ_FUNDO'] = df[cnpj_col].astype(str)

        # Filter for target CNPJs (matching as strings)
        filtered = df[df['CNPJ_FUNDO'].isin(set(map(str, target_cnpjs)))].copy()

        return filtered
    
    def _process_large_file(self, csv_file, target_cnpjs: Set[str], cnpj_col: str = 'CNPJ_FUNDO') -> pd.DataFrame:
        """Process large files in chunks.

        Reads chunks and preserves original data types/values as much as possible. Adds a `CNPJ_FUNDO` column for compatibility.
        """
        chunks = []
        usecols = [cnpj_col, 'DT_COMPTC', 'VL_QUOTA']

        chunk_iterator = pd.read_csv(
            csv_file,
            sep=';',
            usecols=usecols,
            encoding='utf-8',
            chunksize=self.chunk_size
        )
        
        for chunk in chunk_iterator:
            # Normalize CNPJ column to string copy
            chunk['CNPJ_FUNDO'] = chunk[cnpj_col].astype(str)

            # Filter chunk (string matching)
            filtered_chunk = chunk[chunk['CNPJ_FUNDO'].isin(set(map(str, target_cnpjs)))].copy()
            if not filtered_chunk.empty:
                chunks.append(filtered_chunk)
        
        if chunks:
            return pd.concat(chunks, ignore_index=True)
        return pd.DataFrame()
    
    def consolidate_data(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Consolidate multiple DataFrames into one.
        
        Args:
            dataframes: List of DataFrames to consolidate
            
        Returns:
            Consolidated DataFrame
        """
        if not dataframes:
            return pd.DataFrame()
        
        self.logger.info(f"Consolidating {len(dataframes)} DataFrames")
        
        # Concatenate all DataFrames
        consolidated = pd.concat(dataframes, ignore_index=True)
        
        # Sort and deduplicate
        consolidated.sort_values(['CNPJ_FUNDO', 'DT_COMPTC'], inplace=True)
        consolidated.drop_duplicates(subset=['CNPJ_FUNDO', 'DT_COMPTC'], keep='last', inplace=True)
        
        self.logger.info(f"Consolidated to {len(consolidated):,} records")
        return consolidated
    
    def save_consolidated(self, df: pd.DataFrame, filename: str = "consolidated_quotas.parquet") -> Path:
        """
        Save consolidated data to file.
        
        Args:
            df: DataFrame to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.processed_dir / filename
        
        # Save to Parquet (efficient for large datasets)
        df.to_parquet(output_path, index=False)
        
        # Also save a sample CSV for quick inspection
        sample_path = self.processed_dir / "latest_sample.csv"
        df.head(1000).to_csv(sample_path, index=False)
        
        self.logger.info(f"Saved {len(df):,} records to {output_path}")
        return output_path
    
    def load_consolidated(self, filename: str = "consolidated_quotas.parquet") -> Optional[pd.DataFrame]:
        """
        Load consolidated data from file.
        
        Args:
            filename: Input filename
            
        Returns:
            DataFrame or None if file doesn't exist
        """
        file_path = self.processed_dir / filename
        
        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return None
        
        self.logger.info(f"Loading consolidated data from {file_path}")
        return pd.read_parquet(file_path)