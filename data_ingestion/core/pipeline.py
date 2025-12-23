"""
Main pipeline orchestrator for CVM data processing.
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

from config import settings
from core.catalog import FundCatalog
from core.downloader import DownloadManager
from core.processor import DataProcessor
from utils.state_manager import PipelineState
from utils.helpers import generate_month_range, get_last_complete_month

logger = logging.getLogger(__name__)

class CVMPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, data_dir: Optional[str] = None):
        # Use custom data dir if provided
        if data_dir:
            settings.DATA_DIR = Path(data_dir)
            # Recreate directory structure
            for subdir in ["raw", "processed", "cache"]:
                (settings.DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.state = PipelineState()
        self.catalog = FundCatalog(self.state)
        self.downloader = DownloadManager()
        self.processor = DataProcessor()
        self.logger = logging.getLogger(__name__)
    
    def run_initial_ingestion(self, start_date: str, end_date: str, max_workers: int = None,
                              raw_only: bool = False, extract: bool = False, delete_zip: bool = False, overwrite: bool = False) -> bool:
        """
        Run initial historical data ingestion.

        Behavior options added:
          - raw_only: If True, skip any processing and only download (and optionally extract/delete zips).
          - extract: If True, unzip downloaded files into `data/raw_unzip/<zip_stem>/` without modifying contents.
          - delete_zip: If True and `extract` is True, delete the zip after a successful extraction.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_workers: Maximum parallel downloads
            raw_only: Download only; skip processing and consolidation
            extract: Extract zip files after download
            delete_zip: Delete zip files after successful extraction

        Returns:
            True if successful, False otherwise
        """        
        self.logger.info("=" * 60)
        self.logger.info("STARTING INITIAL DATA INGESTION")
        self.logger.info(f"Date Range: {start_date} to {end_date}")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Get fund catalog (skip when raw-only)
            if not raw_only:
                self.logger.info("Step 1: Fetching fund catalog...")
                catalog_df = self.catalog.fetch()
                target_cnpjs = self.catalog.get_fund_cnpjs(catalog_df)
            else:
                target_cnpjs = set()

            # Step 2: Download historical data
            self.logger.info("Step 2: Downloading historical data...")
            downloaded_files = self.downloader.download_range(
                start_date, end_date, max_workers, force=overwrite
            )
            
            # In raw-only mode do not treat missing downloads as fatal; just attempt extraction if requested and return
            if raw_only:
                self.logger.info("Raw-only mode: skipping processing and only handling downloads/unzip")
                if extract:
                    for file_path in downloaded_files:
                        ok = self.downloader.extract_zip(file_path, delete_zip_after=delete_zip)
                        if not ok:
                            self.logger.warning(f"Extraction failed for {file_path.name}")
                return True

            if not downloaded_files:
                self.logger.error("No files were downloaded!")
                return False
            
            # Step 3: Process downloaded files
            self.logger.info("Step 3: Processing downloaded files...")
            processed_dfs = []
            
            for file_path in downloaded_files:
                df = self.processor.process_monthly_file(file_path, target_cnpjs)
                if df is not None and not df.empty:
                    processed_dfs.append(df)
            
            if not processed_dfs:
                self.logger.error("No data was processed!")
                return False
            
            # Step 4: Consolidate and save
            self.logger.info("Step 4: Consolidating data...")
            consolidated_df = self.processor.consolidate_data(processed_dfs)
            
            if consolidated_df.empty:
                self.logger.error("Consolidated DataFrame is empty!")
                return False
            
            # Save consolidated data
            output_path = self.processor.save_consolidated(consolidated_df)
            
            # Update state
            self.state.update(
                total_records=len(consolidated_df),
                last_processed_month=consolidated_df['DT_COMPTC'].max().strftime('%Y-%m-%d'),
                last_successful_run=datetime.now().isoformat()
            )
            
            self.logger.info("=" * 60)
            self.logger.info("INITIAL INGESTION COMPLETE")
            self.logger.info(f"Total records: {len(consolidated_df):,}")
            self.logger.info(f"Output file: {output_path}")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initial ingestion failed: {e}")
            self.logger.exception("Error details:")
            return False
    
    def run_monthly_update(self, include_current_month: bool = False, raw_only: bool = False,
                           extract: bool = False, delete_zip: bool = False, overwrite: bool = False) -> bool:
        """
        Run monthly update to get new data.

        Behavior options added:
          - raw_only: If True, skip any processing and only download (and optionally extract/delete zips).
          - extract: If True, unzip downloaded files into `data/raw_unzip/<zip_stem>/` without modifying contents.
          - delete_zip: If True and `extract` is True, delete the zip after a successful extraction.

        Args:
            include_current_month: Whether to include current (incomplete) month
            raw_only: Download only; skip processing and consolidation
            extract: Extract zip files after download
            delete_zip: Delete zip files after successful extraction

        Returns:
            True if successful, False otherwise
        """        
        self.logger.info("=" * 60)
        self.logger.info("STARTING MONTHLY UPDATE")
        self.logger.info(f"Include current month: {include_current_month}")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Get fund catalog
            self.logger.info("Step 1: Updating fund catalog...")
            catalog_df = self.catalog.fetch()
            target_cnpjs = self.catalog.get_fund_cnpjs(catalog_df)
            
            # Step 2: Determine which months to download
            self.logger.info("Step 2: Determining update range...")

            if raw_only:
                # In raw-only mode, do not load consolidated data or validate anything.
                # Download the last complete month (and current month if requested).
                today = datetime.now()
                if include_current_month:
                    months_to_download = [today.replace(day=1), get_last_complete_month()]
                else:
                    months_to_download = [get_last_complete_month()]
            else:
                # Load existing data to find last date
                existing_df = self.processor.load_consolidated()
                
                if existing_df is None or existing_df.empty:
                    self.logger.error("No existing data found. Run initial ingestion first.")
                    return False
                
                last_date = existing_df['DT_COMPTC'].max()
                self.logger.info(f"Last date in database: {last_date.date()}")
                
                # Calculate start month (next month after last date)
                start_month = last_date.replace(day=1) + timedelta(days=32)
                start_month = start_month.replace(day=1)
                
                # Calculate end month
                today = datetime.now()
                if include_current_month:
                    end_month = today.replace(day=1)
                else:
                    end_month = get_last_complete_month()
                
                self.logger.info(f"Update range: {start_month.strftime('%Y-%m')} to {end_month.strftime('%Y-%m')}")
                
                # Check if update is needed
                if start_month > end_month:
                    self.logger.info("Data is already up to date!")
                    return True
                
                months_to_download = generate_month_range(
                    start_month.strftime('%Y-%m-01'),
                    end_month.strftime('%Y-%m-01')
                )

            # Step 3: Download new months
            self.logger.info("Step 3: Downloading new data...")
            downloaded_files = []
            for month in months_to_download:
                file_path = self.downloader.download_single_month(month, force=overwrite)
                if file_path:
                    downloaded_files.append(file_path)
            
            if not downloaded_files:
                self.logger.warning("No new files to process!")
                return True

            # If raw-only mode requested, optionally extract and delete zip files then exit
            if raw_only:
                self.logger.info("Raw-only mode: skipping processing and only handling downloads/unzip")
                if extract:
                    for file_path in downloaded_files:
                        ok = self.downloader.extract_zip(file_path, delete_zip_after=delete_zip)
                        if not ok:
                            self.logger.warning(f"Extraction failed for {file_path.name}")
                return True

            # Step 4: Process and append new data
            self.logger.info("Step 4: Processing new data...")
            new_records = []
            
            for file_path in downloaded_files:
                df = self.processor.process_monthly_file(file_path, target_cnpjs)
                if df is not None and not df.empty:
                    new_records.append(df)
            
            if not new_records:
                self.logger.warning("No new records found!")
                return True
            
            # Append to existing data
            new_df = self.processor.consolidate_data(new_records)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            
            # Remove duplicates
            updated_df.sort_values(['CNPJ_FUNDO', 'DT_COMPTC'], inplace=True)
            updated_df.drop_duplicates(subset=['CNPJ_FUNDO', 'DT_COMPTC'], keep='last', inplace=True)
            
            # Step 5: Save updated data
            self.logger.info("Step 5: Saving updated data...")
            output_path = self.processor.save_consolidated(updated_df)
            
            # Update state
            new_records_count = len(updated_df) - len(existing_df)
            self.state.update(
                total_records=len(updated_df),
                last_processed_month=updated_df['DT_COMPTC'].max().strftime('%Y-%m-%d'),
                last_update=datetime.now().isoformat()
            )
            
            self.logger.info("=" * 60)
            self.logger.info("MONTHLY UPDATE COMPLETE")
            self.logger.info(f"New records added: {new_records_count:,}")
            self.logger.info(f"Total records now: {len(updated_df):,}")
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Monthly update failed: {e}")
            self.logger.exception("Error details:")
            return False
    
    def get_status(self) -> dict:
        """Get current pipeline status."""
        return self.state.get_full_status()