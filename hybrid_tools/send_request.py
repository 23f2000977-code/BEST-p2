"""
Enhanced POST request tool with Force-Skip and Credential Injection.
"""

from langchain_core.tools import tool
import requests
import time
import os
from typing import Any, Dict, Optional

# Global timer state
class RequestState:
    def __init__(self):
        self.start_time = time.time()

    def reset_timer(self):
        self.start_time = time.time()
        print(f"[TIMER] ⏱️ Timer RESET. Starting count from 0s.")

    def get_elapsed(self):
        return time.time() - self.start_time

_state = RequestState()

def reset_submission_tracking():
    _state.reset_timer()

@tool
def post_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
    """
    Send an HTTP POST request.
    Features: Auto-injects credentials for SKIP, tracks time, handles timeouts.
    """
    # 1. INJECT CREDENTIALS (Fixes the 400 Bad Request Loop)
    if "email" not in payload:
        email = os.getenv("TDS_EMAIL") or os.getenv("EMAIL")
        if email: payload["email"] = email
            
    if "secret" not in payload:
        secret = os.getenv("TDS_SECRET") or os.getenv("SECRET")
        if secret: payload["secret"] = secret

    # 2. TIME LIMIT CHECK (Force Skip if > 160s)
    elapsed = _state.get_elapsed()
    if elapsed > 160 and payload.get("answer") != "SKIP":
        print(f"[SUBMIT] 🚨 Time limit imminent ({elapsed:.1f}s). FORCING 'SKIP'.")
        payload["answer"] = "SKIP"
    
    print(f"\n[SUBMIT] Submitting answer to: {url}")
    
    headers = headers or {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        correct = data.get("correct", False)
        next_url = data.get("url")
        
        result = {"correct": correct, "message": data.get("message", ""), "reason": data.get("reason", "")}

        if correct:
            print(f"[SUBMIT] ✓ Correct answer!")
            if next_url:
                print(f"[SUBMIT] 🔗 Next URL found: {next_url}")
                result["url"] = next_url
                _state.reset_timer()
        else:
            print(f"[SUBMIT] ✗ Wrong/Skipped. Reason: {data.get('reason')}")
            if next_url:
                print(f"[SUBMIT] 🔗 Next URL provided despite failure: {next_url}")
                result["url"] = next_url
                _state.reset_timer()

        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"[SUBMIT] ✗ Exception: {error_msg}")
        return {"error": error_msg, "correct": False}