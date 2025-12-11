"""
Hybrid LangGraph Agent - The "Best of Both Worlds" Final Version

Features:
- 🧠 Dual-Brain: Gemini (Primary) + OpenAI (Fallback)
- 🔄 Smart Rotation: Exhausts all Gemini keys before spending OpenAI credits
- 🛡️ Safe-Fail: "Rage Quit" logic to skip impossible questions
- 📝 Full Logging: Remote Gist uploads + Detailed console output
- 🔧 Fixes: 405 Errors, JSON quoting, Image Math, CSV Dates
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

# Simple configuration: Use Gemini or OpenAI
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() in ("true", "1", "yes")
FALLBACK_OPENAI_MODEL = os.getenv("FALLBACK_OPENAI_MODEL", "gpt-4o-mini")
PRIMARY_OPENAI_MODEL = os.getenv("PRIMARY_OPENAI_MODEL", "gpt-4o-mini")

# -------------------------------------------------
# LOGGING INFRASTRUCTURE (CRITICAL)
# -------------------------------------------------
upload_thread = None
stop_upload_thread = False

def upload_current_log(reason="Progress"):
    """Upload current log file to GitHub Gist."""
    try:
        from remote_logger import upload_to_github_gist
        import glob
        
        # Find the most recent log file
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
        pass  # Silent fail

def periodic_upload_worker():
    """Background worker that uploads logs every 5 minutes."""
    global stop_upload_thread
    
    while not stop_upload_thread:
        # Wait 5 minutes (300 seconds)
        for _ in range(300):
            if stop_upload_thread:
                return
            time.sleep(1)
        
        # Upload if still running
        if not stop_upload_thread:
            upload_current_log("Progress Update")

def start_periodic_uploads():
    """Start background thread for periodic uploads."""
    global upload_thread, stop_upload_thread
    stop_upload_thread = False
    upload_thread = threading.Thread(target=periodic_upload_worker, daemon=True)
    upload_thread.start()

def stop_periodic_uploads():
    """Stop background upload thread."""
    global stop_upload_thread
    stop_upload_thread = True

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully by uploading logs."""
    print("\n[AGENT] ⚠️ Interrupted by user, uploading logs...")
    stop_periodic_uploads()
    upload_current_log("Interrupted")
    print("[AGENT] ✓ Logs uploaded, exiting...")
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# -------------------------------------------------
# STATE & TOOLS
# -------------------------------------------------
class AgentState(TypedDict):
    """Enhanced state with context tracking."""
    messages: Annotated[List, add_messages]
    previous_answers: Dict[str, Any]
    context: Dict[str, Any]
    start_time: float

TOOLS = [
    run_code,
    get_rendered_html,
    download_file,
    post_request,
    add_dependencies,
    transcribe_audio,
    extract_context,
    analyze_image,
    create_visualization,
    create_chart_from_data
]

# -------------------------------------------------
# LLM SETUP
# -------------------------------------------------
# Initialize API key rotator (for Gemini)
try:
    api_rotator = get_api_key_rotator()
    print(f"[AGENT] API Key Rotation: {api_rotator.key_count} key(s) available")
except Exception as e:
    print(f"[AGENT] Warning: API key rotation failed: {e}")
    print(f"[AGENT] Falling back to single API key")
    api_rotator = None

rate_limiter = InMemoryRateLimiter(
    requests_per_second=9/60,
    check_every_n_seconds=1,
    max_bucket_size=9
)

def create_gemini_llm():
    """Create Gemini LLM with current rotated key."""
    if not api_rotator:
        raise ValueError("No Rotator available for Gemini")
    
    # Get current VALID key
    api_key = api_rotator.get_current_key()
    
    return init_chat_model(
        model_provider="google_genai",
        model="gemini-2.5-flash",
        api_key=api_key,
        rate_limiter=rate_limiter,
        max_retries=0  # Fail fast to allow manual rotation
    ).bind_tools(TOOLS)

def create_openai_llm(use_fallback=False):
    """Create OpenAI LLM."""
    model = FALLBACK_OPENAI_MODEL if use_fallback else PRIMARY_OPENAI_MODEL
    return init_chat_model(
        model_provider="openai",
        model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    ).bind_tools(TOOLS)

if USE_GEMINI:
    print(f"[AGENT] Primary LLM: Gemini (gemini-2.5-flash)")
    print(f"[AGENT] Fallback LLM: OpenAI ({FALLBACK_OPENAI_MODEL})")
else:
    print(f"[AGENT] Primary LLM: OpenAI ({PRIMARY_OPENAI_MODEL})")


# -------------------------------------------------
# SYSTEM PROMPT (THE BRAIN)
# -------------------------------------------------
SYSTEM_PROMPT = f"""You are an autonomous quiz-solving agent.

Your goal is to solve data science tasks and submit answers to the `post_request` tool.

STRATEGY FOR SPECIFIC TASK TYPES:

1. 🎨 HEATMAPS / COLORS:
   - If asked for "most frequent color" or "heatmap color":
   - CALL `analyze_image` immediately. The tool has built-in math to calculate the Hex code.
   - Submit the exact Hex code returned by the tool.
   - DO NOT write your own Python code for this.

2. 💻 UV / GIT COMMANDS (JSON FORMATTING RULES):
   - You will often be asked to submit a command string like: `uv http get ...`
   - CRITICAL: You must ensure valid JSON syntax.
   - NEVER use double quotes (") inside the command string.
   - ALWAYS use SINGLE QUOTES (') for inner arguments.
   - WRONG: "answer": "uv http get "https://url" -H "Accept: json"" (This breaks JSON)
   - CORRECT: "answer": "uv http get 'https://url' -H 'Accept: json'"
   - If the server rejects your answer due to format, try removing the quotes around the URL entirely.

3. 📁 FILE PATHS:
   - All files are downloaded to `hybrid_llm_files/`
   - When using `run_code`, you are ALREADY inside that folder.
   - Use `pd.read_csv("filename.csv")`, NOT `pd.read_csv("hybrid_llm_files/filename.csv")`.

4. 🛑 SKIPPING LOGIC:
   - If you fail a task 3 times, or if the `post_request` tool tells you "Time limit imminent",
   - SUBMIT "SKIP" as the answer.
   - This ensures we receive the next URL instead of timing out.

5. 📮 SUBMISSION URL RULE (CRITICAL):
   - You must find the correct submission URL in the HTML.
   - It is usually `https://tds-llm-analysis.s-anand.net/submit` or ends in `/submit`.
   - DO NOT POST to the question URL (e.g., do not post to `.../project2-uv`).
   - If you get a "405 Method Not Allowed" error, you are posting to the wrong URL. Check the context or default to `/submit`.

6. 🐍 CSV & DATES (ROBUSTNESS):
   - When parsing dates in CSVs using pandas, ALWAYS use `errors='coerce'`.
   - Example: `pd.to_datetime(df['date_col'], errors='coerce')`
   - This prevents crashes when the data is messy or mixed format.

GENERAL PROCESS:
1. `get_rendered_html(url)`
2. `extract_context(html)` -> Find the `/submit` URL.
3. Solve task (use `transcribe_audio`, `analyze_image`, or `run_code`).
4. `post_request(url, payload)`

INFO:
- Email: {EMAIL}
- Secret: {SECRET}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------
def filter_messages(messages: List, max_keep=20) -> List:
    """OpenAI-Safe Memory Pruning."""
    if len(messages) <= max_keep:
        return messages
    
    system_prompt = messages[0]
    recent = messages[-max_keep:]
    
    # Remove orphan ToolMessages (prevents OpenAI 400 errors)
    while recent and isinstance(recent[0], ToolMessage):
        recent.pop(0)
    
    return [system_prompt] + recent

def log_llm_decision(result, llm_type="LLM"):
    """Log what the LLM decided to do (Restored Feature)."""
    if hasattr(result, "tool_calls") and result.tool_calls:
        print(f"[AGENT] 🔧 {llm_type} decided to call {len(result.tool_calls)} tool(s):")
        for i, tool_call in enumerate(result.tool_calls, 1):
            tool_name = tool_call.get("name", "unknown")
            print(f"[AGENT]   {i}. {tool_name}")
    elif hasattr(result, "content"):
        content = result.content
        if isinstance(content, str):
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"[AGENT] 💬 {llm_type} response: {preview}")

# -------------------------------------------------
# AGENT NODE (ROBUST FALLBACK LOGIC)
# -------------------------------------------------
def agent_node(state: AgentState):
    """
    Agent decision node.
    Logic: Try Gemini Key 1..4 -> If All Fail, IMMEDIATELY call OpenAI.
    """
    trimmed_messages = filter_messages(state["messages"])
    
    # 1. Try Gemini Loop
    if USE_GEMINI:
        # Loop while we still have valid keys to try
        while api_rotator and not api_rotator.are_all_keys_exhausted():
            try:
                # Use current key
                llm = create_gemini_llm()
                print(f"[AGENT] 🧠 Thinking (Gemini)...")
                
                result = (prompt | llm).invoke({"messages": trimmed_messages})
                
                # Success! Log and Return.
                log_llm_decision(result, "Gemini")
                return {"messages": state["messages"] + [result]}
                
            except Exception as e:
                error_msg = str(e)
                print(f"[AGENT] ⚠️ Gemini Error: {error_msg[:100]}")
                
                # Check for Quota (429) or Server Overload (503)
                if "429" in error_msg or "quota" in error_msg.lower() or "503" in error_msg:
                    print(f"[AGENT] 🔄 Key failed (429/503). Marking as dead and trying next...")
                    api_rotator.mark_key_exhausted()
                    # Loop continues to next key automatically via api_rotator
                else:
                    # Logic error? Break loop and let OpenAI handle it.
                    print(f"[AGENT] 🛑 Non-quota error. Switching to OpenAI.")
                    break

    # 2. Fallback to OpenAI
    # We reach here ONLY if Gemini is disabled, exhausted, or failed.
    print(f"[AGENT] 🧠 Thinking (OpenAI Fallback)...")
    try:
        llm = create_openai_llm()
        result = (prompt | llm).invoke({"messages": trimmed_messages})
        log_llm_decision(result, "OpenAI")
        return {"messages": state["messages"] + [result]}
    except Exception as e:
        print(f"[AGENT] ❌ OpenAI Error: {e}")
        # If OpenAI fails, we crash.
        raise e

def route(state):
    """Route based on tool calls."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    if "END" in str(getattr(last, "content", "")):
        return END
    return "agent"

# -------------------------------------------------
# GRAPH SETUP
# -------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges(
    "agent",
    route
)

app = graph.compile()

# -------------------------------------------------
# RUN AGENT ENTRY POINT
# -------------------------------------------------
def run_agent(url: str) -> str:
    """Run the agent on a quiz URL."""
    print(f"\n{'='*60}")
    print(f"[AGENT] Starting quiz chain")
    print(f"[AGENT] URL: {url}")
    print(f"{'='*60}\n")
    
    # 1. Reset timer for the first question
    from hybrid_tools.send_request import reset_submission_tracking
    reset_submission_tracking()
    
    # 2. Start logging threads
    start_periodic_uploads()
    start_time = time.time()
    
    initial_state = {
        "messages": [{"role": "user", "content": url}],
        "previous_answers": {},
        "context": {},
        "start_time": start_time
    }
    
    try:
        app.invoke(
            initial_state,
            config={"recursion_limit": RECURSION_LIMIT}
        )
        
        total_time = time.time() - start_time
        print(f"\n[AGENT] ✓ Tasks completed successfully")
        print(f"[AGENT] Total time: {total_time:.1f}s")
        
        stop_periodic_uploads()
        upload_current_log("Success")
        return "success"

    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n[AGENT] ✗ Error: {e}")
        
        stop_periodic_uploads()
        upload_current_log("Error")
        return f"error: {e}"