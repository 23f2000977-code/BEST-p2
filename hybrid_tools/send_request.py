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
    Features: Auto-injects credentials, tracks time, handles timeouts.
    """
    # ------------------------------------------------------------------
    # 1. HARD-CODED CREDENTIAL INJECTION (The Fix for 400 Errors)
    # ------------------------------------------------------------------
    # We force these values in. The LLM cannot mess this up.
    my_email = os.getenv("TDS_EMAIL") or os.getenv("EMAIL")
    my_secret = os.getenv("TDS_SECRET") or os.getenv("SECRET")
    
    # Ensure payload is a dict
    if not isinstance(payload, dict):
        payload = {"answer": str(payload)}

    # Inject
    if my_email: payload["email"] = my_email
    if my_secret: payload["secret"] = my_secret
    
    # ------------------------------------------------------------------
    # 2. TIME LIMIT CHECK
    # ------------------------------------------------------------------
    elapsed = _state.get_elapsed()
    if elapsed > 160 and payload.get("answer") != "SKIP":
        print(f"[SUBMIT] 🚨 Time limit imminent ({elapsed:.1f}s). FORCING 'SKIP'.")
        payload["answer"] = "SKIP"
    
    print(f"\n[SUBMIT] Submitting answer to: {url}")
    # print(f"[SUBMIT] Payload keys: {list(payload.keys())}") # Debugging
    
    headers = headers or {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        # Handle 400 Errors gracefully
        if response.status_code == 400:
            print(f"[SUBMIT] ⚠️ 400 Bad Request. Retrying with raw SKIP payload...")
            # Emergency Fallback
            return {"error": "Bad Request", "correct": False, "status": "failed"}

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
            # If we skipped and got a new URL, that counts as success for the flow
            if next_url:
                print(f"[SUBMIT] 🔗 Next URL provided despite failure: {next_url}")
                result["url"] = next_url
                _state.reset_timer()

        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"[SUBMIT] ✗ Exception: {error_msg}")
        return {"error": error_msg, "correct": False}