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
                
                # Read CSV from ZIP
                with zf.open(csv_name) as csv_file:
                    # Read in chunks if file is large
                    file_size = zf.getinfo(csv_name).file_size
                    
                    if file_size > 50 * 1024 * 1024:  # > 50 MB
                        return self._process_large_file(csv_file, target_cnpjs)
                    else:
                        return self._process_small_file(csv_file, target_cnpjs)
                        
        except zipfile.BadZipFile:
            self.logger.error(f"Bad ZIP file: {file_path.name}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}")
            return None
    
    def _process_small_file(self, csv_file, target_cnpjs: Set[str]) -> pd.DataFrame:
        """Process small files by reading all at once."""
        df = pd.read_csv(
            csv_file,
            sep=';',
            decimal=',',
            dtype={'CNPJ_FUNDO': str, 'VL_QUOTA': float},
            usecols=['CNPJ_FUNDO', 'DT_COMPTC', 'VL_QUOTA'],
            encoding='utf-8'
        )
        
        # Filter for target CNPJs
        filtered = df[df['CNPJ_FUNDO'].isin(target_cnpjs)].copy()
        filtered['DT_COMPTC'] = pd.to_datetime(filtered['DT_COMPTC'])
        
        return filtered
    
    def _process_large_file(self, csv_file, target_cnpjs: Set[str]) -> pd.DataFrame:
        """Process large files in chunks."""
        chunks = []
        chunk_iterator = pd.read_csv(
            csv_file,
            sep=';',
            decimal=',',
            dtype={'CNPJ_FUNDO': str, 'VL_QUOTA': float},
            usecols=['CNPJ_FUNDO', 'DT_COMPTC', 'VL_QUOTA'],
            encoding='utf-8',
            chunksize=self.chunk_size
        )
        
        for chunk in chunk_iterator:
            # Filter chunk
            filtered_chunk = chunk[chunk['CNPJ_FUNDO'].isin(target_cnpjs)].copy()
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