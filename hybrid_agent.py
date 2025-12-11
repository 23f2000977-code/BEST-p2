"""
Hybrid LangGraph Agent - Best of Both Worlds

Combines:
- LangGraph architecture
- Enhanced features for data science tasks
- Smart API Key Rotation
- Memory Management (OpenAI Safe Version)
- "Rage Quit" Logic
- FULL Remote Logging (GitHub Gist)
- Fixes for: 405 Errors, JSON Quoting, and Heatmap Math
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

EMAIL = os.getenv("TDS_EMAIL") or os.getenv("EMAIL")
SECRET = os.getenv("TDS_SECRET") or os.getenv("SECRET")
RECURSION_LIMIT = 5000

# -------------------------------------------------
# LOGGING INFRASTRUCTURE
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
    """Enhanced state with context tracking from your project."""
    messages: Annotated[List, add_messages]
    previous_answers: Dict[str, Any]  # Track answers for multi-question chains
    context: Dict[str, Any]  # Rich context from pages
    start_time: float  # For time tracking


# All available tools
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
# LLM CONFIGURATION
# -------------------------------------------------
# Simple configuration: Use Gemini or OpenAI
USE_GEMINI = os.getenv("USE_GEMINI", "true").lower() in ("true", "1", "yes")

OPENAI_MODEL = os.getenv("OPENAI_MODEL")  # Old variable
FALLBACK_OPENAI_MODEL = os.getenv("FALLBACK_OPENAI_MODEL", OPENAI_MODEL or "gpt-4o-mini")
PRIMARY_OPENAI_MODEL = os.getenv("PRIMARY_OPENAI_MODEL", FALLBACK_OPENAI_MODEL)

# Initialize API key rotator (for Gemini)
try:
    api_rotator = get_api_key_rotator()
    print(f"[AGENT] API Key Rotation: {api_rotator.key_count} key(s) available")
except Exception as e:
    print(f"[AGENT] Warning: API key rotation failed: {e}")
    print(f"[AGENT] Falling back to single API key")
    api_rotator = None

rate_limiter = InMemoryRateLimiter(
    requests_per_second=9/60,  # 9 requests per minute
    check_every_n_seconds=1,
    max_bucket_size=9
)

def create_gemini_llm():
    """Create Gemini LLM with rotation and fast-fail."""
    if not api_rotator:
        raise ValueError("No Rotator available for Gemini")
        
    # Get next VALID key (skipping exhausted ones)
    api_key = api_rotator.get_next_key()
    
    return init_chat_model(
        model_provider="google_genai",
        model="gemini-2.5-flash",
        api_key=api_key,
        rate_limiter=rate_limiter,
        max_retries=0  # CRITICAL: Don't wait 60s, fail immediately so we can rotate
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
# ENHANCED SYSTEM PROMPT
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

GENERAL PROCESS:
1. `get_rendered_html(url)`
2. `extract_context(html)` -> Look for API/Submit URLs
3. Solve task (use `transcribe_audio` for audio, `analyze_image` for images)
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
# MESSAGE TRIMMING UTILITY (OPENAI SAFE)
# -------------------------------------------------
def filter_messages(messages: List, max_keep=20) -> List:
    """
    OpenAI-Safe Memory Pruning.
    Ensures ToolMessages are not left as orphans without their AIMessage calls.
    """
    if len(messages) <= max_keep:
        return messages
    
    # Keep System Prompt
    system_prompt = messages[0]
    
    # Get recent messages
    recent = messages[-max_keep:]
    
    # SAFETY CHECK: If the first message is a ToolMessage (result),
    # but we deleted the AIMessage (call) before it, OpenAI will crash.
    # We must remove leading ToolMessages until we hit a Human or AI message.
    while recent and isinstance(recent[0], ToolMessage):
        # print(f"[AGENT] 🧹 Pruning orphan ToolMessage to satisfy OpenAI")
        recent.pop(0)
    
    # print(f"[AGENT] 🧹 Pruning memory: Keeping last {len(recent)} messages")
    return [system_prompt] + recent


# -------------------------------------------------
# AGENT NODE WITH SIMPLE CONFIGURATION
# -------------------------------------------------
def agent_node(state: AgentState):
    """Agent decision-making node with simple USE_GEMINI configuration."""
    print(f"\n[AGENT] 🤖 LLM thinking...")
    
    # ----------------------------------------
    # FIX: Trim messages before sending to LLM
    # ----------------------------------------
    trimmed_messages = filter_messages(state["messages"])
    
    if USE_GEMINI:
        # Use Gemini with OpenAI fallback
        # Check if all Gemini keys exhausted
        if api_rotator and api_rotator.are_all_keys_exhausted():
            print(f"[AGENT] 🔄 All Gemini keys exhausted, using OpenAI")
            return use_openai(state)
        
        # Try Gemini
        try:
            llm = create_gemini_llm()
            llm_with_prompt = prompt | llm
            # Use trimmed messages
            result = llm_with_prompt.invoke({"messages": trimmed_messages})
            log_llm_decision(result, "Gemini")
            return {"messages": state["messages"] + [result]}
            
        except Exception as e:
            error_msg = str(e)
            print(f"[AGENT] ❌ Gemini failed: {error_msg[:200]}")
            
            # Check if it's a quota error
            is_quota_error = any(keyword in error_msg.lower() 
                               for keyword in ["quota", "429", "resource_exhausted", "rate limit"])
            
            if is_quota_error and api_rotator:
                print(f"[AGENT] 🔄 Quota exceeded. Marking key as dead and retrying...")
                
                # Mark current key as dead
                current_key = api_rotator.get_current_key()
                api_rotator.mark_key_exhausted(current_key)
                
                if api_rotator.are_all_keys_exhausted():
                    print(f"[AGENT] 🔄 All Gemini keys exhausted, switching to OpenAI")
                    return use_openai(state, use_fallback=True)
                else:
                    print(f"[AGENT] 🔄 Retrying immediately with next key...")
                    return agent_node(state) # Recursive retry with new key
            
            # Fallback to OpenAI for other errors
            print(f"[AGENT] 🔄 Switching to OpenAI fallback")
            return use_openai(state, use_fallback=True)
    
    else:
        # Use OpenAI only
        return use_openai(state)

def use_openai(state: AgentState, use_fallback=False):
    """Use OpenAI LLM."""
    # ----------------------------------------
    # FIX: Trim messages before sending to LLM
    # ----------------------------------------
    trimmed_messages = filter_messages(state["messages"])
    
    try:
        llm = create_openai_llm(use_fallback=use_fallback)
        llm_with_prompt = prompt | llm
        # Use trimmed messages
        result = llm_with_prompt.invoke({"messages": trimmed_messages})
        model_name = FALLBACK_OPENAI_MODEL if use_fallback else PRIMARY_OPENAI_MODEL
        log_llm_decision(result, f"OpenAI ({model_name})")
        return {"messages": state["messages"] + [result]}
    except Exception as e:
        print(f"[AGENT] ❌ OpenAI failed: {str(e)[:200]}")
        raise Exception("LLM failed")

def log_llm_decision(result, llm_type="LLM"):
    """Log what the LLM decided to do."""
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
# GRAPH ROUTING
# -------------------------------------------------
def route(state):
    """Route based on last message."""
    last = state["messages"][-1]
    
    # Support both objects and dicts
    tool_calls = None
    if hasattr(last, "tool_calls"):
        tool_calls = getattr(last, "tool_calls", None)
    elif isinstance(last, dict):
        tool_calls = last.get("tool_calls")

    if tool_calls:
        return "tools"
    
    # Get content robustly
    content = None
    if hasattr(last, "content"):
        content = getattr(last, "content", None)
    elif isinstance(last, dict):
        content = last.get("content")

    if isinstance(content, str) and content.strip() == "END":
        return END
    if isinstance(content, list) and content[0].get("text", "").strip() == "END":
        return END
    
    return "agent"


# Build graph
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
# RUN AGENT
# -------------------------------------------------
def run_agent(url: str) -> str:
    """Run the agent on a quiz URL."""
    print(f"\n{'='*60}")
    print(f"[AGENT] Starting quiz chain")
    print(f"[AGENT] URL: {url}")
    print(f"{'='*60}\n")
    
    # Reset submission tracking
    from hybrid_tools.send_request import reset_submission_tracking
    reset_submission_tracking()
    
    # Start periodic log uploads (every 5 minutes)
    start_periodic_uploads()
    
    # Initialize state
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
        
        # Stop periodic uploads
        stop_periodic_uploads()
        
        # Upload full log file to GitHub Gist if configured
        try:
            from remote_logger import upload_to_github_gist
            import glob
            import os
            
            # Find the most recent log file
            log_files = glob.glob("hybrid_logs_*.txt")
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                with open(latest_log, 'r') as f:
                    log_content = f.read()
                
                upload_to_github_gist(
                    content=log_content,
                    description=f"Quiz Solver Success - {url} - {total_time:.1f}s"
                )
        except Exception as log_error:
            # Don't fail if logging fails
            pass
        
        return "success"
    except Exception as e:
        total_time = time.time() - start_time
        print(f"\n[AGENT] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Stop periodic uploads
        stop_periodic_uploads()
        
        # Upload full log file to GitHub Gist if configured
        try:
            from remote_logger import upload_to_github_gist
            import glob
            import os
            
            # Find the most recent log file
            log_files = glob.glob("hybrid_logs_*.txt")
            if log_files:
                latest_log = max(log_files, key=os.path.getctime)
                with open(latest_log, 'r') as f:
                    log_content = f.read()
                
                upload_to_github_gist(
                    content=log_content,
                    description=f"Quiz Solver Error - {url} - {str(e)[:50]}"
                )
        except Exception as log_error:
            pass
        
        return f"error: {e}"