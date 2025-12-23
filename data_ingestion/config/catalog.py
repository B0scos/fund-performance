"""
Fund catalog management for CVM pipeline.
"""

import pandas as pd
import logging
from typing import Optional
from config import settings, constants
from utils.state_manager import PipelineState
from datetime import datetime

logger = logging.getLogger(__name__)

class FundCatalog:
    """Manage the fund catalog download and updates."""
    
    def __init__(self, state_manager: Optional[PipelineState] = None):
        self.catalog_path = settings.FUND_CATALOG_PATH
        self.state = state_manager or PipelineState()
        self.logger = logging.getLogger(__name__)
    
    def fetch(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Download and save the latest fund catalog.
        
        Args:
            force_refresh: Whether to force download even if cache exists
            
        Returns:
            DataFrame with fund catalog
        """
        self.logger.info("Starting fund catalog update")
        
        # Check cache age
        if self.catalog_path.exists() and not force_refresh:
            cache_age = self._get_cache_age()
            if cache_age.days < 7:
                self.logger.info("Loading fund catalog from cache (less than 7 days old)")
                return self._load_cached_catalog()
        
        # Download fresh catalog
        return self._download_catalog()
    
    def _get_cache_age(self) -> pd.Timedelta:
        """Calculate how old the cache is."""

        
        cache_time = datetime.fromtimestamp(self.catalog_path.stat().st_mtime)
        return pd.Timestamp.now() - pd.Timestamp(cache_time)
    
    def _load_cached_catalog(self) -> pd.DataFrame:
        """Load catalog from cache."""
        try:
            df = pd.read_parquet(self.catalog_path)
            self.state.update(total_funds=len(df))
            return df
        except Exception as e:
            self.logger.error(f"Failed to load cached catalog: {e}")
            raise
    
    def _download_catalog(self) -> pd.DataFrame:
        """Download fresh catalog from CVM."""
        import time
        
        self.logger.info("Downloading latest fund catalog from CVM...")
        start_time = time.time()
        
        try:
            # Download CSV
            df = pd.read_csv(
                constants.FUND_CATALOG_URL,
                sep=';',
                encoding='ISO-8859-1',
                dtype=str,
                low_memory=False
            )
            
            # Normalize CNPJ
            df['CNPJ_FUNDO'] = df['CNPJ_FUNDO'].astype(str).str.replace(r'\D', '', regex=True)
            
            # Keep essential columns
            available_cols = [c for c in constants.FUND_CATALOG_COLUMNS if c in df.columns]
            df = df[available_cols]
            
            # Save
            df.to_parquet(self.catalog_path)
            
            # Update state
            self.state.update(total_funds=len(df))
            
            elapsed = time.time() - start_time
            self.logger.info(f"Fund catalog saved. Total funds: {len(df):,}")
            self.logger.info(f"Download completed in {elapsed:.2f} seconds")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download fund catalog: {e}")
            raise
    
    def get_fund_cnpjs(self, catalog_df: Optional[pd.DataFrame] = None) -> set:
        """
        Extract CNPJ numbers from catalog.
        
        Args:
            catalog_df: Optional catalog DataFrame
            
        Returns:
            Set of CNPJ numbers
        """
        if catalog_df is None:
            catalog_df = self.fetch()
        
        cnpjs = set(catalog_df['CNPJ_FUNDO'].unique())
        self.logger.info(f"Extracted {len(cnpjs):,} unique CNPJs")
        return cnpjs