"""
Command-line interface for the CVM pipeline.
"""

import argparse
import sys
from pathlib import Path

from core.pipeline import CVMPipeline

from config import settings
import shutil

from utils.state_manager import PipelineState
from config import settings

from core.downloader import DownloadManager
from config import settings

def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="CVM Fund Quota Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initial historical download
  python main.py init --start 2020-01-01 --end 2025-12-31
  
  # Download only and extract zips (do not process or change data)
  python main.py init --start 2025-10-01 --end 2025-12-31 --raw-only --extract --delete-zip
  
  # Monthly update (schedule this)
  python main.py update
  
  # Check status
  python main.py status
  
  # Update with current month
  python main.py update --include-current
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Init command
    init_parser = subparsers.add_parser('init', help='Initial historical download')
    init_parser.add_argument('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    init_parser.add_argument('--end', default='2025-12-31', help='End date (YYYY-MM-DD)')
    init_parser.add_argument('--workers', type=int, default=4, help='Parallel downloads')
    init_parser.add_argument('--data-dir', help='Custom data directory')
    init_parser.add_argument('--raw-only', action='store_true', help='Only download (no processing)')
    init_parser.add_argument('--extract', action='store_true', help='Extract ZIP files after download')
    init_parser.add_argument('--delete-zip', action='store_true', help='Delete ZIP files after extraction')
    init_parser.add_argument('--overwrite', action='store_true', help='Re-download files even if present')
    init_parser.add_argument('--log-level', default='INFO', 
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Monthly update')
    update_parser.add_argument('--include-current', action='store_true',
                             help='Include current month')
    update_parser.add_argument('--data-dir', help='Custom data directory')
    update_parser.add_argument('--raw-only', action='store_true', help='Only download (no processing)')
    update_parser.add_argument('--extract', action='store_true', help='Extract ZIP files after download')
    update_parser.add_argument('--delete-zip', action='store_true', help='Delete ZIP files after extraction')
    update_parser.add_argument('--overwrite', action='store_true', help='Re-download files even if present')
    update_parser.add_argument('--log-level', default='INFO',
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check pipeline status')
    status_parser.add_argument('--data-dir', help='Custom data directory')
    
    # Fetch command (download only, no processing)
    fetch_parser = subparsers.add_parser('fetch', help='Download monthly ZIPs without processing')
    fetch_parser.add_argument('--start', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    fetch_parser.add_argument('--end', default='2025-12-31', help='End date (YYYY-MM-DD)')
    fetch_parser.add_argument('--workers', type=int, default=4, help='Parallel downloads')
    fetch_parser.add_argument('--data-dir', help='Custom data directory')
    fetch_parser.add_argument('--extract', action='store_true', help='Extract ZIP files after download')
    fetch_parser.add_argument('--delete-zip', action='store_true', help='Delete ZIP files after extraction')
    fetch_parser.add_argument('--overwrite', action='store_true', help='Re-download files even if present')
    fetch_parser.add_argument('--log-level', default='INFO', 
                              choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean data directory')
    clean_parser.add_argument('--data-dir', help='Custom data directory')
    clean_parser.add_argument('--keep-raw', action='store_true', help='Keep raw files')
    clean_parser.add_argument('--keep-processed', action='store_true', help='Keep processed data')
    
    return parser

def handle_init(args):
    """Handle init command."""
    print("\n" + "="*60)
    print("INITIAL HISTORICAL DOWNLOAD")
    print("="*60)
    
    if args.start == '2020-01-01' and args.end == '2025-12-31':
        print("WARNING: This will download ~15-20 GB of data.")
        print("The process may take several hours.")
        print("\nDo you want to continue? (yes/no): ", end='')
        response = input().strip().lower()
        if response not in ['yes', 'y']:
            print("Download cancelled.")
            return
    

    
    # Run pipeline
    pipeline = CVMPipeline(data_dir=args.data_dir)
    success = pipeline.run_initial_ingestion(
        start_date=args.start,
        end_date=args.end,
        max_workers=args.workers,
        raw_only=bool(getattr(args, 'raw_only', False)),
        extract=bool(getattr(args, 'extract', False)),
        delete_zip=bool(getattr(args, 'delete_zip', False))
    )
    
    if not success:
        sys.exit(1)

def handle_update(args):
    """Handle update command."""
    print("\n" + "="*60)
    print("MONTHLY UPDATE")
    print("="*60)
    
    
    # Run pipeline
    pipeline = CVMPipeline(data_dir=args.data_dir)
    success = pipeline.run_monthly_update(
        include_current_month=args.include_current,
        raw_only=bool(getattr(args, 'raw_only', False)),
        extract=bool(getattr(args, 'extract', False)),
        delete_zip=bool(getattr(args, 'delete_zip', False))
    )
    
    if not success:
        sys.exit(1)

def handle_status(args):
    """Handle status command."""

    
    # Load state
    if args.data_dir:
        settings.DATA_DIR = Path(args.data_dir)
    
    state = PipelineState()
    status = state.get_full_status()
    
    print("\n" + "="*60)
    print("PIPELINE STATUS")
    print("="*60)
    
    # Pipeline info
    print(f"\nPipeline:")
    print(f"  Version: {status['pipeline'].get('version', 'N/A')}")
    print(f"  Created: {status['pipeline'].get('created', 'N/A')}")
    
    # Data info
    print(f"\nData:")
    print(f"  Total funds: {status['data'].get('total_funds', 0):,}")
    print(f"  Total records: {status['data'].get('total_records', 0):,}")
    print(f"  Last processed: {status['data'].get('last_processed_month', 'Never')}")
    print(f"  Last update: {status['data'].get('last_update', 'Never')}")
    
    # Execution info
    print(f"\nExecution:")
    print(f"  Run count: {status['execution'].get('run_count', 0)}")
    print(f"  Last successful: {status['execution'].get('last_successful_run', 'Never')}")
    
    # Storage info
    print(f"\nStorage:")
    print(f"  Raw files: {status['storage'].get('raw_files', 0)}")
    print(f"  Processed file: {'Exists' if status['storage'].get('processed_file_exists') else 'Missing'}")
    print(f"  Catalog file: {'Exists' if status['storage'].get('catalog_file_exists') else 'Missing'}")
    
    print("="*60)


def handle_fetch(args):
    """Handle fetch command: download and optionally extract zips without any processing."""

    # Apply custom data dir if provided
    if args.data_dir:
        settings.DATA_DIR = Path(args.data_dir)
        for subdir in ["raw", "raw_unzip", "processed", "cache"]:
            (settings.DATA_DIR / subdir).mkdir(parents=True, exist_ok=True)

    dm = DownloadManager()

    # Download range
    downloaded = dm.download_range(args.start, args.end, max_workers=args.workers, force=bool(getattr(args, 'overwrite', False)))

    if not downloaded:
        print("No files were downloaded.")
        return

    if args.extract:
        for p in downloaded:
            ok = dm.extract_zip(p, delete_zip_after=bool(getattr(args, 'delete_zip', False)))
            if not ok:
                print(f"Failed to extract {p.name}")

    print("Fetch complete.")

def handle_clean(args):
    """Handle clean command."""

    
    if args.data_dir:
        settings.DATA_DIR = Path(args.data_dir)
    
    print("\n" + "="*60)
    print("CLEANING DATA DIRECTORY")
    print("="*60)
    
    # List what will be deleted
    to_delete = []
    
    if not args.keep_raw and settings.RAW_DATA_DIR.exists():
        raw_files = list(settings.RAW_DATA_DIR.glob("*"))
        to_delete.append(f"Raw files: {len(raw_files)} files")
    
    if not args.keep_processed and settings.PROCESSED_DATA_DIR.exists():
        processed_files = list(settings.PROCESSED_DATA_DIR.glob("*"))
        to_delete.append(f"Processed data: {len(processed_files)} files")
    
    if settings.CACHE_DIR.exists():
        cache_files = list(settings.CACHE_DIR.glob("*"))
        to_delete.append(f"Cache: {len(cache_files)} files")
    
    if not to_delete:
        print("Nothing to clean!")
        return
    
    print("\nWill delete:")
    for item in to_delete:
        print(f"  - {item}")
    
    print("\nAre you sure? (yes/no): ", end='')
    response = input().strip().lower()
    
    if response not in ['yes', 'y']:
        print("Clean cancelled.")
        return
    
    # Perform cleanup
    try:
        if not args.keep_raw and settings.RAW_DATA_DIR.exists():
            shutil.rmtree(settings.RAW_DATA_DIR)
            settings.RAW_DATA_DIR.mkdir()
            print("✓ Raw files cleaned")
        
        if not args.keep_processed and settings.PROCESSED_DATA_DIR.exists():
            shutil.rmtree(settings.PROCESSED_DATA_DIR)
            settings.PROCESSED_DATA_DIR.mkdir()
            print("✓ Processed data cleaned")
        
        if settings.CACHE_DIR.exists():
            shutil.rmtree(settings.CACHE_DIR)
            settings.CACHE_DIR.mkdir()
            print("✓ Cache cleaned")
        
        print("\nCleanup complete!")
        
    except Exception as e:
        print(f"\nError during cleanup: {e}")

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'init':
            handle_init(args)
        elif args.command == 'update':
            handle_update(args)
        elif args.command == 'fetch':
            handle_fetch(args)
        elif args.command == 'status':
            handle_status(args)
        elif args.command == 'clean':
            handle_clean(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)