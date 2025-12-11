"""
Enhanced POST request tool with Hardcoded Credentials and Force-Skip.
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
    Send an HTTP POST request to submit an answer.
    """
    # ------------------------------------------------------------------
    # 1. HARDCODED CREDENTIALS (The "Nuclear Option")
    # ------------------------------------------------------------------
    # These match the logs you provided. This ensures 400 Bad Request never happens.
    FALLBACK_EMAIL = "23f2000977@ds.study.iitm.ac.in"
    FALLBACK_SECRET = "Saumya29june"

    # Inject if missing
    if "email" not in payload:
        payload["email"] = os.getenv("TDS_EMAIL") or FALLBACK_EMAIL
            
    if "secret" not in payload:
        payload["secret"] = os.getenv("TDS_SECRET") or FALLBACK_SECRET

    # ------------------------------------------------------------------
    # 2. TIME LIMIT CHECK
    # ------------------------------------------------------------------
    elapsed = _state.get_elapsed()
    if elapsed > 160 and payload.get("answer") != "SKIP":
        print(f"[SUBMIT] 🚨 Time limit imminent ({elapsed:.1f}s). FORCING 'SKIP'.")
        payload["answer"] = "SKIP"
    
    print(f"\n[SUBMIT] Submitting answer to: {url}")
    
    headers = headers or {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        # Handle 400 Errors gracefully -> Force Skip Success logic on client side
        if response.status_code == 400:
            print(f"[SUBMIT] ⚠️ 400 Bad Request. Server rejected payload.")
            print(f"[SUBMIT] Payload sent: {payload}")
            if payload.get("answer") == "SKIP":
                 # If we failed to skip, return a fake success to break the loop
                 return {"status": "skipped_locally", "message": "Server rejected skip, moving on."}

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
        return {"error": str(e), "correct": False}