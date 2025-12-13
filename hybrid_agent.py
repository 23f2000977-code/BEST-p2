"""
Hybrid LangGraph Agent - The 17-Question Solver Version
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
from langchain_core.messages import ToolMessage, AIMessage
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
FALLBACK_OPENAI_MODEL = os.getenv("FALLBACK_OPENAI_MODEL", "gpt-4o-mini")
PRIMARY_OPENAI_MODEL = os.getenv("PRIMARY_OPENAI_MODEL", "gpt-4o-mini")

# -------------------------------------------------
# LOGGING INFRASTRUCTURE
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
            with open(latest_log, 'r') as f:
                log_content = f.read()
            upload_to_github_gist(
                content=log_content,
                description=f"Quiz Solver {reason} - {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
    except Exception:
        pass

def periodic_upload_worker():
    global stop_upload_thread
    while not stop_upload_thread:
        for _ in range(300):
            if stop_upload_thread: return
            time.sleep(1)
        if not stop_upload_thread: upload_current_log("Progress Update")

def start_periodic_uploads():
    global upload_thread, stop_upload_thread
    stop_upload_thread = False
    upload_thread = threading.Thread(target=periodic_upload_worker, daemon=True)
    upload_thread.start()

def stop_periodic_uploads():
    global stop_upload_thread
    stop_upload_thread = True

def signal_handler(signum, frame):
    print("\n[AGENT] ⚠️ Interrupted by user, uploading logs...")
    stop_periodic_uploads()
    upload_current_log("Interrupted")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
# LLM SETUP
# -------------------------------------------------
try:
    api_rotator = get_api_key_rotator()
except Exception as e:
    api_rotator = None

rate_limiter = InMemoryRateLimiter(requests_per_second=9/60, check_every_n_seconds=1, max_bucket_size=9)

def create_gemini_llm():
    if not api_rotator: raise ValueError("No Rotator available for Gemini")
    return init_chat_model(
        model_provider="google_genai",
        model="gemini-2.5-flash",
        api_key=api_rotator.get_current_key(),
        rate_limiter=rate_limiter,
        max_retries=0
    ).bind_tools(TOOLS)

def create_openai_llm(use_fallback=False):
    model = FALLBACK_OPENAI_MODEL if use_fallback else PRIMARY_OPENAI_MODEL
    return init_chat_model(
        model_provider="openai",
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    ).bind_tools(TOOLS)

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""You are an autonomous quiz-solving agent.

Your goal is to solve data science tasks and submit answers to the `post_request` tool.

STRATEGY FOR SPECIFIC TASK TYPES:

1. 🎨 HEATMAPS / COLORS:
   - If asked for "most frequent color" or "heatmap color":
   - CALL `analyze_image` immediately. The tool has built-in math to calculate the Hex code.
   - DO NOT write your own Python code for this.

2. 💻 UV / GIT COMMANDS (JSON FORMATTING RULES):
   - You will often be asked to submit a command string like: `uv http get ...`
   - CRITICAL: You must ensure valid JSON syntax.
   - NEVER use double quotes (") inside the command string.
   - ALWAYS use SINGLE QUOTES (') for inner arguments.
   - CORRECT: "answer": "uv http get 'https://url' -H 'Accept: json'"

3. 📁 FILE PATHS (CRITICAL FIX):
   - All files are downloaded to `hybrid_llm_files/`
   - **IMPORTANT**: When using `run_code`, the script executes INSIDE that folder.
   - **DO NOT** prepend `hybrid_llm_files/` to file paths in your Python code.
   - **CORRECT**: `Image.open('heatmap.png')`, `pd.read_csv('data.csv')`

4. 🛑 SKIPPING LOGIC:
   - If you fail a task 3 times, or if the `post_request` tool tells you "Time limit imminent",
   - SUBMIT "SKIP" as the answer.

5. 📮 SUBMISSION URL RULE:
   - Always prefer `https://tds-llm-analysis.s-anand.net/submit`.
   - If `extract_context` finds a different one, use it, but be careful of 405 errors.

INFO:
- Email: {EMAIL}
- Secret: {SECRET}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

# -------------------------------------------------
# UTILITIES & AGENT NODE
# -------------------------------------------------
def filter_messages(messages: List, max_keep=20) -> List:
    if len(messages) <= max_keep: return messages
    recent = messages[-max_keep:]
    while recent and isinstance(recent[0], ToolMessage): recent.pop(0)
    return [messages[0]] + recent

def log_llm_decision(result, llm_type="LLM"):
    if hasattr(result, "tool_calls") and result.tool_calls:
        print(f"[AGENT] 🔧 {llm_type} decided to call {len(result.tool_calls)} tool(s):")
        for i, tool_call in enumerate(result.tool_calls, 1):
            print(f"[AGENT]   {i}. {tool_call.get('name', 'unknown')}")

def agent_node(state: AgentState):
    trimmed_messages = filter_messages(state["messages"])
    
    # 1. Try Gemini
    if USE_GEMINI:
        while api_rotator and not api_rotator.are_all_keys_exhausted():
            try:
                llm = create_gemini_llm()
                print(f"[AGENT] 🧠 Thinking (Gemini)...")
                result = (prompt | llm).invoke({"messages": trimmed_messages})
                log_llm_decision(result, "Gemini")
                return {"messages": state["messages"] + [result]}
            except Exception as e:
                error_msg = str(e)
                print(f"[AGENT] ⚠️ Gemini Error: {error_msg[:100]}")
                if "429" in error_msg or "quota" in error_msg.lower() or "503" in error_msg:
                    api_rotator.mark_key_exhausted()
                else:
                    break

    # 2. Fallback OpenAI
    print(f"[AGENT] 🧠 Thinking (OpenAI Fallback)...")
    try:
        llm = create_openai_llm()
        result = (prompt | llm).invoke({"messages": trimmed_messages})
        log_llm_decision(result, "OpenAI")
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

def run_agent(url: str) -> str:
    print(f"\n{'='*60}\n[AGENT] Starting quiz chain\n[AGENT] URL: {url}\n{'='*60}\n")
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
        print(f"\n[AGENT] ✗ Error: {e}")
        stop_periodic_uploads()
        upload_current_log("Error")
        return f"error: {e}"