"""
Enhanced POST request tool with Force-Skip, Credential Injection, and Timer Logic.
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
    1. Auto-injects Email/Secret if missing (Prevents 400 Bad Request).
    2. Tracks time per question.
    3. FORCE SKIPS if time > 160s.
    """
    # -------------------------------------------------------
    # 1. CREDENTIAL INJECTION (Safety Net)
    # -------------------------------------------------------
    # If the LLM tries to skip but forgets credentials, we add them.
    if "email" not in payload:
        email = os.getenv("TDS_EMAIL") or os.getenv("EMAIL")
        if email:
            payload["email"] = email
            
    if "secret" not in payload:
        secret = os.getenv("TDS_SECRET") or os.getenv("SECRET")
        if secret:
            payload["secret"] = secret

    # -------------------------------------------------------
    # 2. TIME LIMIT SAFETY CHECK
    # -------------------------------------------------------
    elapsed = _state.get_elapsed()
    
    # If we are nearing the 180s limit (3 mins), we FORCE a skip.
    if elapsed > 160:
        print(f"\n[SUBMIT] 🚨 CRITICAL: Time limit imminent ({elapsed:.1f}s / 180s).")
        print(f"[SUBMIT] ⏭️ FORCING 'SKIP' to get next question link.")
        payload["answer"] = "SKIP"
    else:
        print(f"\n[SUBMIT] Submitting answer to: {url}")
        print(f"[SUBMIT] Per-Question Timer: {elapsed:.1f}s / 180s")
    
    headers = headers or {"Content-Type": "application/json"}
    
    try:
        # Send Request
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract fields
        correct = data.get("correct", False)
        next_url = data.get("url")
        message = data.get("message", "")
        reason = data.get("reason", "")
        
        # -------------------------------------------------------
        # 3. RESULT HANDLING & TIMER RESET
        # -------------------------------------------------------
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
            print(f"[SUBMIT] ✗ Wrong answer / Skipped")
            if reason: print(f"[SUBMIT] Reason: {reason}")
            
            # Did the server give us the next URL anyway (e.g. after a SKIP)?
            if next_url:
                print(f"[SUBMIT] 🔗 Next URL provided despite failure: {next_url}")
                result["url"] = next_url
                result["status"] = "moved_to_next"
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