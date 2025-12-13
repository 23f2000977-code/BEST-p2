"""
Enhanced POST request tool with Auto-Correction for 405 Errors.
"""

from langchain_core.tools import tool
import requests
import time
import os
from typing import Any, Dict, Optional

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
    
    FEATURES:
    1. Auto-injects Email/Secret.
    2. Auto-corrects 405 Method Not Allowed (Redirects to /submit).
    3. FORCE SKIPS if time > 140s.
    """
    # 1. CREDENTIAL INJECTION
    FALLBACK_EMAIL = os.getenv("TDS_EMAIL")
    FALLBACK_SECRET = os.getenv("TDS_SECRET")

    if "email" not in payload and FALLBACK_EMAIL:
        payload["email"] = FALLBACK_EMAIL
    if "secret" not in payload and FALLBACK_SECRET:
        payload["secret"] = FALLBACK_SECRET

    # 2. TIME LIMIT CHECK
    elapsed = _state.get_elapsed()
    if elapsed > 140 and payload.get("answer") != "SKIP":
        print(f"\n[SUBMIT] 🚨 CRITICAL: Time limit imminent ({elapsed:.1f}s / 180s).")
        print(f"[SUBMIT] ⏭️ FORCING 'SKIP' to get next question link.")
        payload["answer"] = "SKIP"
    
    print(f"\n[SUBMIT] Submitting answer to: {url}")
    headers = headers or {"Content-Type": "application/json"}
    
    try:
        # Initial Request
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        # 3. CRITICAL FIX: 405 AUTO-CORRECTION
        if response.status_code == 405:
            print(f"[SUBMIT] ⚠️ 405 Method Not Allowed at {url}")
            
            # If we aren't already at /submit, try the standard submit URL
            if "/submit" not in url:
                new_url = "https://tds-llm-analysis.s-anand.net/submit"
                print(f"[SUBMIT] 🔄 Auto-redirecting to: {new_url}")
                response = requests.post(new_url, json=payload, headers=headers, timeout=30)
        
        # 4. HANDLE 400 ERRORS (The Skip Loop Fix)
        if response.status_code == 400:
            print(f"[SUBMIT] ⚠️ 400 Bad Request. Server rejected payload.")
            if payload.get("answer") == "SKIP":
                 print(f"[SUBMIT] 🛑 Breaking SKIP loop manually.")
                 return {"status": "error", "message": "Server rejected skip.", "correct": False}

        response.raise_for_status()
        data = response.json()
        
        if data.get("correct") or data.get("url"):
            if data.get("url"):
                print(f"[SUBMIT] 🔗 Next URL found: {data.get('url')}")
                _state.reset_timer()
            else:
                print(f"[SUBMIT] ✓ Answer Correct (No next URL).")
        else:
            print(f"[SUBMIT] ✗ Wrong/Skipped. Reason: {data.get('reason')}")

        return data
        
    except Exception as e:
        error_msg = str(e)
        print(f"[SUBMIT] ✗ Exception: {error_msg}")
        return {"error": error_msg, "correct": False}