"""
Hybrid LangGraph Agent - Loop-Proof Version
"""

from langgraph.graph import StateGraph, END, START
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from hybrid_tools import (
    get_rendered_html, run_code, post_request, download_file,
    add_dependencies, transcribe_audio, extract_context,
    analyze_image, create_visualization, create_chart_from_data
)
from typing import TypedDict, Annotated, List, Dict, Any
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langgraph.graph.message import add_messages
import os
import time
import threading
import signal
import sys
from dotenv import load_dotenv
from api_key_rotator import get_api_key_rotator

load_dotenv()

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
EMAIL = os.getenv("TDS_EMAIL") or os.getenv("EMAIL")
SECRET = os.getenv("TDS_SECRET") or os.getenv("SECRET")
RECURSION_LIMIT = 5000
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() in ("true", "1", "yes")

# -------------------------------------------------
# LOGGING (Simplified)
# -------------------------------------------------
upload_thread = None
stop_upload_thread = False

def upload_current_log(reason="Progress"):
    try:
        from remote_logger import upload_to_github_gist
        import glob
        log_files = glob.glob("hybrid_logs_*.txt")
        if log_files:
            latest_log = max(log_files, key=os.path.getctime)
            with open(latest_log, 'r') as f: content = f.read()
            upload_to_github_gist(content, f"Quiz Solver {reason}")
    except: pass

def start_periodic_uploads():
    global upload_thread, stop_upload_thread
    stop_upload_thread = False
    def worker():
        while not stop_upload_thread:
            time.sleep(300)
            if not stop_upload_thread: upload_current_log("Update")
    upload_thread = threading.Thread(target=worker, daemon=True)
    upload_thread.start()

def stop_periodic_uploads():
    global stop_upload_thread
    stop_upload_thread = True

# -------------------------------------------------
# STATE & TOOLS
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    previous_answers: Dict[str, Any]
    context: Dict[str, Any]
    start_time: float

TOOLS = [
    run_code, get_rendered_html, download_file, post_request,
    add_dependencies, transcribe_audio, extract_context,
    analyze_image, create_visualization, create_chart_from_data
]

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""You are an autonomous quiz-solving agent.

GOAL: Solve tasks and submit answers via `post_request`.

STRATEGY:
1. 🎨 HEATMAPS/COLORS: Use `analyze_image`. It has built-in math to find the hex code.
2. 🔊 AUDIO (CRITICAL):
   - If using OpenAI, **DO NOT** try to convert/transcribe audio. It fails.
   - **IMMEDIATELY** call `post_request` with `answer="SKIP"`.
   - Do not waste time trying to fix audio code.
3. 💻 UV/GIT COMMANDS:
   - NEVER use double quotes inside command strings.
   - CORRECT: "uv http get 'https://url'"
   - WRONG: "uv http get "https://url""
4. 📮 SUBMISSION URL:
   - Check `extract_context` for the submit URL (usually ends in /submit).
   - If you get "405 Method Not Allowed", submit to the `/submit` endpoint instead.

FAIL-SAFE:
- If a tool fails twice, submit "SKIP".
- If `post_request` returns "server_error_400", DO NOT RETRY. Move to the next step.

INFO:
- Email: {EMAIL}
- Secret: {SECRET}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

# -------------------------------------------------
# AGENT LOGIC
# -------------------------------------------------
api_rotator = get_api_key_rotator() if USE_GEMINI else None
rate_limiter = InMemoryRateLimiter(requests_per_second=9/60, check_every_n_seconds=1, max_bucket_size=9)

def create_gemini_llm():
    if not api_rotator: raise ValueError("No Rotator")
    return init_chat_model(model_provider="google_genai", model="gemini-2.5-flash", api_key=api_rotator.get_current_key(), rate_limiter=rate_limiter, max_retries=0).bind_tools(TOOLS)

def create_openai_llm():
    return init_chat_model(model_provider="openai", model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")).bind_tools(TOOLS)

def agent_node(state: AgentState):
    # Keep context small
    messages = state["messages"][-15:]
    while messages and isinstance(messages[0], ToolMessage): messages.pop(0)

    # 1. Try Gemini
    if USE_GEMINI and api_rotator and not api_rotator.are_all_keys_exhausted():
        try:
            print(f"[AGENT] 🧠 Thinking (Gemini)...")
            result = (prompt | create_gemini_llm()).invoke({"messages": messages})
            return {"messages": state["messages"] + [result]}
        except Exception as e:
            print(f"[AGENT] ⚠️ Gemini Error: {str(e)[:100]}")
            if "429" in str(e) or "quota" in str(e).lower() or "503" in str(e):
                api_rotator.mark_key_exhausted()

    # 2. Fallback OpenAI
    print(f"[AGENT] 🧠 Thinking (OpenAI Fallback)...")
    try:
        result = (prompt | create_openai_llm()).invoke({"messages": messages})
        return {"messages": state["messages"] + [result]}
    except Exception as e:
        print(f"[AGENT] ❌ OpenAI Error: {e}")
        raise e

def route(state):
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls: return "tools"
    if "END" in str(getattr(last, "content", "")): return END
    return "agent"

# -------------------------------------------------
# RUN AGENT
# -------------------------------------------------
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)
app = graph.compile()

def run_agent(url: str):
    print(f"--- STARTING QUIZ: {url} ---")
    from hybrid_tools.send_request import reset_submission_tracking
    reset_submission_tracking()
    start_periodic_uploads()
    
    try:
        app.invoke({
            "messages": [{"role": "user", "content": url}],
            "previous_answers": {}, "context": {}, "start_time": time.time()
        }, config={"recursion_limit": RECURSION_LIMIT})
        
        stop_periodic_uploads()
        upload_current_log("Success")
        return "success"
    except Exception as e:
        print(f"--- ERROR: {e} ---")
        stop_periodic_uploads()
        upload_current_log("Error")
        return f"error: {e}"