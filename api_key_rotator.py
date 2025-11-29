"""
API Key Rotation Manager - Optimized
Skipped exhausted keys automatically.
"""

import os
from typing import List
import threading

class APIKeyRotator:
    """Manages rotation of multiple API keys to avoid rate limits."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._current_index = 0
        self._api_keys = self._load_api_keys()
        self._exhausted_keys = set()  # Track rate-limited keys
        self._all_exhausted = False
        
        if not self._api_keys:
            raise ValueError("No Google API keys found in environment")
        
        print(f"[API_ROTATOR] Loaded {len(self._api_keys)} API key(s)")
    
    def _load_api_keys(self) -> List[str]:
        keys = []
        # Primary key
        primary_key = os.getenv("GOOGLE_API_KEY")
        if primary_key:
            keys.append(primary_key)
        
        # Additional keys
        i = 2
        while True:
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if not key:
                break
            keys.append(key)
            i += 1
        return keys
    
    def get_next_key(self) -> str:
        """Get the next VALID key, skipping exhausted ones."""
        with self._lock:
            # If all keys are exhausted, just return the current one (fallback logic elsewhere handles this)
            if len(self._exhausted_keys) >= len(self._api_keys):
                self._all_exhausted = True
                return self._api_keys[self._current_index]

            # Try to find a non-exhausted key
            start_index = self._current_index
            for i in range(len(self._api_keys)):
                # Check next candidate
                candidate_index = (start_index + i) % len(self._api_keys)
                
                # If this key is NOT in the exhausted list, pick it
                if candidate_index not in self._exhausted_keys:
                    self._current_index = candidate_index
                    key = self._api_keys[self._current_index]
                    key_preview = f"...{key[-4:]}" if len(key) > 4 else "****"
                    print(f"[API_ROTATOR] Switching to key {self._current_index + 1}/{len(self._api_keys)}: {key_preview}")
                    return key

            # If we get here, everything is exhausted
            self._all_exhausted = True
            return self._api_keys[self._current_index]
    
    def get_current_key(self) -> str:
        with self._lock:
            return self._api_keys[self._current_index]
    
    def mark_key_exhausted(self, key_string: str = None):
        """Mark the CURRENT key (or specific key) as exhausted."""
        with self._lock:
            # If key string provided, find its index
            target_index = self._current_index
            if key_string:
                try:
                    target_index = self._api_keys.index(key_string)
                except ValueError:
                    pass

            if target_index not in self._exhausted_keys:
                self._exhausted_keys.add(target_index)
                print(f"[API_ROTATOR] ⚠️ Key {target_index + 1} marked as DEAD. ({len(self._exhausted_keys)}/{len(self._api_keys)} dead)")
            
            if len(self._exhausted_keys) >= len(self._api_keys):
                self._all_exhausted = True
                print(f"[API_ROTATOR] 🚨 ALL KEYS EXHAUSTED!")

    def are_all_keys_exhausted(self) -> bool:
        with self._lock:
            return len(self._exhausted_keys) >= len(self._api_keys)
    
    @property
    def key_count(self) -> int:
        return len(self._api_keys)

# Global instance
_rotator = None

def get_api_key_rotator() -> APIKeyRotator:
    global _rotator
    if _rotator is None:
        _rotator = APIKeyRotator()
    return _rotator