"""
Helper functions for the CVM pipeline.
"""

import time
import functools
from datetime import datetime, timedelta
from typing import List, Callable, Any
from pathlib import Path
import re
import shutil


def retry(max_attempts: int = 3, delay: int = 1):
    """
    Decorator for retrying functions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Delay between attempts in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
            raise last_exception
        return wrapper
    return decorator

def generate_month_range(start_date: str, end_date: str) -> List[datetime]:
    """
    Generate list of months between two dates.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        List of datetime objects (first day of each month)
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    months = []
    current = start.replace(day=1)
    
    while current <= end:
        months.append(current)
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    
    return months

def get_last_complete_month() -> datetime:
    """
    Get the first day of the last complete month.
    
    Returns:
        Datetime for first day of last complete month
    """
    today = datetime.now()
    # Go to first day of current month, then back one month
    last_month = today.replace(day=1) - timedelta(days=1)
    return last_month.replace(day=1)

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

def normalize_cnpj(cnpj: str) -> str:
    """
    Normalize CNPJ by removing non-digit characters.
    
    Args:
        cnpj: CNPJ string (with or without punctuation)
        
    Returns:
        CNPJ with only digits
    """
    return re.sub(r'\D', '', cnpj)

def check_disk_space(path: Path, required_gb: float = 1.0) -> bool:
    """
    Check if there's enough disk space.
    
    Args:
        path: Path to check
        required_gb: Required space in GB
        
    Returns:
        True if enough space, False otherwise
    """
    try:
        free_bytes = shutil.disk_usage(path).free
        free_gb = free_bytes / (1024 ** 3)
        return free_gb >= required_gb
    except:
        return True  # If we can't check, assume OK