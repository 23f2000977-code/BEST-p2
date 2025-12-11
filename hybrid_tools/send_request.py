"""
Enhanced POST request tool with Hardcoded Credentials, Force-Skip, and Timer Logic.
"""

from langchain_core.tools import tool
import requests
import time
import os
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------
# STATE MANAGEMENT
# We use a simple class to ensure the timer state persists correctly
# --------------------------------------------------------------------------
class RequestState:
    def __init__(self):
        self.start_time = time.time()

    def reset_timer(self):
        self.start_time = time.time()
        print(f"[TIMER] ⏱️ Timer RESET. Starting count from 0s.")

    def get_elapsed(self):
        return time.time() - self.start_time

# Global instance
_state = RequestState()

def reset_submission_tracking():
    """Called by the main agent at the start of the entire quiz."""
    _state.reset_timer()

@tool
def post_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
    """
    Send an HTTP POST request to submit an answer.
    
    FEATURES:
    1. Auto-injects Email/Secret (Hardcoded Fallback) to prevent 400 Bad Request.
    2. Tracks time per question.
    3. FORCE SKIPS if time > 160s.
    """
    # ------------------------------------------------------------------
    # 1. HARDCODED CREDENTIALS (The "Nuclear Option" for Reliability)
    # ------------------------------------------------------------------
    # These ensure that even if os.getenv fails, we have the values.
    # Updated based on your logs.
    FALLBACK_EMAIL = "23f2000977@ds.study.iitm.ac.in"
    FALLBACK_SECRET = "Saumya29june"

    # Inject if missing
    if "email" not in payload:
        payload["email"] = os.getenv("TDS_EMAIL") or FALLBACK_EMAIL
            
    if "secret" not in payload:
        payload["secret"] = os.getenv("TDS_SECRET") or FALLBACK_SECRET

    # ------------------------------------------------------------------
    # 2. TIME LIMIT SAFETY CHECK
    # ------------------------------------------------------------------
    elapsed = _state.get_elapsed()
    
    # If we are nearing the 180s limit (3 mins), we FORCE a skip.
    if elapsed > 160 and payload.get("answer") != "SKIP":
        print(f"\n[SUBMIT] 🚨 CRITICAL: Time limit imminent ({elapsed:.1f}s / 180s).")
        print(f"[SUBMIT] ⏭️ FORCING 'SKIP' to get next question link.")
        payload["answer"] = "SKIP"
    
    print(f"\n[SUBMIT] Submitting answer to: {url}")
    
    headers = headers or {"Content-Type": "application/json"}
    
    try:
        # Send Request
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        # -------------------------------------------------------
        # 3. HANDLE 400 ERRORS (The Skip Loop Fix)
        # -------------------------------------------------------
        if response.status_code == 400:
            print(f"[SUBMIT] ⚠️ 400 Bad Request. Server rejected payload.")
            # If we were trying to SKIP and it failed, return a FAKE success
            # to stop the LLM from trying to skip infinitely.
            if payload.get("answer") == "SKIP":
                 print(f"[SUBMIT] 🛑 Breaking SKIP loop manually.")
                 return {
                     "status": "error", 
                     "message": "Server rejected skip, but we are moving on to avoid loop.",
                     "correct": False
                 }

        response.raise_for_status()
        
        data = response.json()
        
        # Extract fields
        correct = data.get("correct", False)
        next_url = data.get("url")
        message = data.get("message", "")
        reason = data.get("reason", "")
        
        result = {
            "correct": correct,
            "message": message,
            "reason": reason
        }

        # SCENARIO A: Correct Answer
        if correct:
            print(f"[SUBMIT] ✓ Correct answer!")
            if next_url:
                print(f"[SUBMIT] 🔗 Next URL found: {next_url}")
                result["url"] = next_url
                # CRITICAL: Reset timer for the NEW question
                _state.reset_timer()
            else:
                print(f"[SUBMIT] 🎉 Quiz chain completed!")

        # SCENARIO B: Wrong Answer (But maybe we get a link?)
        else:
            print(f"[SUBMIT] ✗ Wrong/Skipped. Reason: {reason}")
            
            # If the server gave us the next URL anyway (e.g. after a valid SKIP)
            if next_url:
                print(f"[SUBMIT] 🔗 Next URL provided despite failure: {next_url}")
                result["url"] = next_url
                # CRITICAL: Reset timer because we are moving to a NEW question
                _state.reset_timer()
            else:
                # Still on the same question
                remaining = 180 - _state.get_elapsed()
                if remaining > 0:
                    print(f"[SUBMIT] ⏳ Time remaining: {remaining:.1f}s - Retrying...")
                else:
                    print(f"[SUBMIT] 💀 Time expired. No next URL.")

        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"[SUBMIT] ✗ Exception: {error_msg}")
        return {"error": error_msg, "correct": False}