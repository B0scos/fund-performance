"""
State management for the pipeline.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from config import settings

class PipelineState:
    """Manage pipeline state persistence."""
    
    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or settings.STATE_FILE
        self._state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or create default."""
        default_state = {
            "pipeline": {
                "version": "2.0",
                "created": datetime.now().isoformat()
            },
            "data": {
                "total_funds": 0,
                "total_records": 0,
                "last_processed_month": None,
                "last_update": None
            },
            "execution": {
                "last_successful_run": None,
                "last_error": None,
                "run_count": 0
            }
        }
        
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    return self._merge_dicts(default_state, loaded)
            except Exception:
                return default_state
        else:
            return default_state
    
    def _merge_dicts(self, d1: Dict, d2: Dict) -> Dict:
        """Recursively merge two dictionaries."""
        result = d1.copy()
        for key, value in d2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_dicts(result[key], value)
            else:
                result[key] = value
        return result
    
    def save(self):
        """Save state to file."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save state: {e}")
    
    def update(self, **kwargs):
        """
        Update state values.
        
        Example:
            state.update(total_records=1000, last_update="2024-01-01")
        """
        for key, value in kwargs.items():
            # Try to find the key in nested structure
            if key in ["total_funds", "total_records", "last_processed_month", "last_update"]:
                self._state["data"][key] = value
            elif key in ["last_successful_run", "last_error"]:
                self._state["execution"][key] = value
            elif key == "run_count":
                self._state["execution"]["run_count"] = self._state["execution"].get("run_count", 0) + 1
            else:
                self._state[key] = value
        
        self.save()
    
    def get(self, key: str, default=None):
        """Get state value."""
        # Check nested structures
        if key in self._state["data"]:
            return self._state["data"][key]
        elif key in self._state["execution"]:
            return self._state["execution"][key]
        elif key in self._state["pipeline"]:
            return self._state["pipeline"][key]
        else:
            return self._state.get(key, default)
    
    def get_full_status(self) -> Dict[str, Any]:
        """Get complete status information."""
        from pathlib import Path
        
        status = self._state.copy()
        
        # Add file system info
        status["storage"] = {
            "raw_files": len(list(settings.RAW_DATA_DIR.glob("*.zip"))),
            "processed_file_exists": (settings.PROCESSED_DATA_DIR / "consolidated_quotas.parquet").exists(),
            "catalog_file_exists": settings.FUND_CATALOG_PATH.exists()
        }
        
        return status