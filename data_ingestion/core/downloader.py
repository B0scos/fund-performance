"""
Download functionality for CVM monthly files.
"""

import requests
import time
import logging
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from config import settings, constants
from utils.helpers import retry
from utils.helpers import generate_month_range

logger = logging.getLogger(__name__)

class DownloadManager:
    """Manage downloading of CVM monthly files."""
    
    def __init__(self):
        self.raw_dir = settings.RAW_DATA_DIR
        self.raw_unzip_dir = settings.RAW_UNZIP_DIR
        self.timeout = settings.DOWNLOAD_TIMEOUT
        self.retries = settings.DOWNLOAD_RETRIES
        self.logger = logging.getLogger(__name__)

    def extract_zip(self, zip_path: Path, dest_dir: Optional[Path] = None, delete_zip_after: bool = False) -> bool:
        """Extract a downloaded ZIP to the raw_unzip directory (or `dest_dir`).

        The extraction preserves the CSVs exactly (no parsing or modification).
        It creates a subdirectory named after the zip stem to avoid collisions.

        Args:
            zip_path: Path to the ZIP file
            dest_dir: Optional destination directory; defaults to settings.RAW_UNZIP_DIR
            delete_zip_after: Whether to delete the ZIP file after successful extraction

        Returns:
            True if extraction succeeded, False otherwise
        """
        import zipfile

        dest_dir = dest_dir or self.raw_unzip_dir
        target_dir = dest_dir / zip_path.stem
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(path=target_dir)
            self.logger.info(f"✓ Extracted {zip_path.name} -> {target_dir}")

            if delete_zip_after:
                try:
                    zip_path.unlink()
                    self.logger.info(f"✓ Deleted zip file {zip_path.name}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete zip file {zip_path.name}: {e}")

            return True
        except zipfile.BadZipFile:
            self.logger.error(f"Bad ZIP file: {zip_path.name}")
            return False
        except Exception as e:
            self.logger.error(f"Error extracting {zip_path.name}: {e}")
            return False    
    def generate_monthly_url(self, year_month: datetime) -> str:
        """
        Generate URL for monthly file based on date.
        
        Args:
            year_month: Datetime for the month
            
        Returns:
            URL string
        """
        ym_str = year_month.strftime("%Y%m")
        
        if year_month.year < 2021:
            # Historical files are in annual ZIPs
            return f"{constants.HIST_BASE_URL}inf_diario_fi_{year_month.year}.zip"
        else:
            # Recent files are monthly
            return f"{constants.BASE_URL}inf_diario_fi_{ym_str}.zip"
    
    def get_local_path(self, year_month: datetime) -> Path:
        """
        Get local path for monthly file.
        
        Args:
            year_month: Datetime for the month
            
        Returns:
            Path object
        """
        ym_str = year_month.strftime("%Y%m")
        filename = constants.MONTHLY_FILE_PATTERN.format(year_month=ym_str)
        return self.raw_dir / filename
    
    @retry(max_attempts=3, delay=2)
    def download_single_month(self, year_month: datetime, force: bool = False) -> Optional[Path]:
        """
        Download a single monthly file.
        
        Args:
            year_month: Month to download
            force: If True, re-download even when the file exists
            
        Returns:
            Path to downloaded file, or None if failed
        """
        url = self.generate_monthly_url(year_month)
        local_path = self.get_local_path(year_month)
        
        # Skip if already exists unless force requested
        if local_path.exists() and not force:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            self.logger.debug(f"File exists: {local_path.name} ({size_mb:.1f} MB)")
            return local_path
        
        self.logger.info(f"Downloading {local_path.name}...")
        start_time = time.time()
        
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Save file
            with open(local_path, 'wb') as f:
                f.write(response.content)
            
            # Verify download
            if local_path.exists():
                size_mb = local_path.stat().st_size / (1024 * 1024)
                elapsed = time.time() - start_time
                self.logger.info(f"✓ Downloaded {local_path.name} ({size_mb:.1f} MB in {elapsed:.1f}s)")
                return local_path
            else:
                self.logger.error(f"File not saved: {local_path.name}")
                return None
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self.logger.error(f"File not found: {url}")
            else:
                self.logger.error(f"HTTP error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Download error: {e}")
            return None
    
    def download_range(self, start_date: str, end_date: str, max_workers: int = None, force: bool = False) -> List[Path]:
        """
        Download multiple months in parallel.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_workers: Maximum parallel downloads
            force: If True, re-download files even when they exist
            
        Returns:
            List of downloaded file paths
        """
        
        max_workers = max_workers or settings.MAX_WORKERS
        months = generate_month_range(start_date, end_date)
        
        self.logger.info(f"Downloading {len(months)} months with {max_workers} workers")
        
        downloaded = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit download tasks
            future_to_month = {
                executor.submit(self.download_single_month, month, force): month 
                for month in months
            }
            
            # Process results
            for future in as_completed(future_to_month):
                month = future_to_month[future]
                try:
                    result = future.result()
                    if result:
                        downloaded.append(result)
                except Exception as e:
                    self.logger.error(f"Failed to download {month.strftime('%Y-%m')}: {e}")
        
        self.logger.info(f"Downloaded {len(downloaded)}/{len(months)} files")
        return downloaded