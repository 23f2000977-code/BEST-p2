"""
Enhanced POST request tool with retry logic and time tracking.
UPDATED: Resets timer on successful answer.
"""

from langchain_core.tools import tool
import requests
import json
import time
from typing import Any, Dict, Optional

# Track submission history
_submission_history = []
_start_time = None

def reset_submission_tracking():
    """Reset submission tracking for new quiz chain."""
    global _submission_history, _start_time
    _submission_history = []
    _start_time = time.time()

@tool
def post_request(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
    """
    Send an HTTP POST request to submit an answer.
    """
    global _submission_history, _start_time
    
    # Initialize timer if not set
    if _start_time is None:
        _start_time = time.time()
    
    elapsed = time.time() - _start_time
    
    headers = headers or {"Content-Type": "application/json"}
    
    print(f"\n[SUBMIT] Submitting answer to: {url}")
    print(f"[SUBMIT] Elapsed time: {elapsed:.1f}s / 180s")
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        delay = data.get("delay", elapsed)
        correct = data.get("correct", False)
        next_url = data.get("url")
        reason = data.get("reason", "")
        
        # Track submission
        _submission_history.append({
            "url": payload.get("url"),
            "answer": payload.get("answer"),
            "correct": correct,
            "delay": delay,
            "timestamp": time.time()
        })
        
        # Build response
        result = {
            "correct": correct,
            "delay": delay,
            "reason": reason
        }
        
        if correct:
            print(f"[SUBMIT] ✓ Correct answer!")
            # ---------------------------------------------------------
            # CRITICAL FIX: Reset timer on success for the next question
            # ---------------------------------------------------------
            _start_time = time.time()
            print(f"[SUBMIT] ⏱️ Timer reset for next question")
            
            if next_url:
                result["url"] = next_url
                print(f"[SUBMIT] Next question: {next_url}")
            else:
                print(f"[SUBMIT] ✓ Quiz chain completed!")
        else:
            print(f"[SUBMIT] ✗ Wrong answer")
            if reason: print(f"[SUBMIT] Reason: {reason}")
            
            if delay < 180:
                print(f"[SUBMIT] Time remaining: {180 - delay:.1f}s - can retry")
            else:
                print(f"[SUBMIT] Time limit exceeded")
                if next_url:
                    result["url"] = next_url
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"[SUBMIT] ✗ Exception: {error_msg}")
        return {"error": error_msg, "correct": False}