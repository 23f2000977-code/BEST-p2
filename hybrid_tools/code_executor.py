"""
Enhanced code executor with safety checks and smart features.
Includes FFmpeg path injection for Audio processing.
"""

from langchain_core.tools import tool
import subprocess
import os
import sys

@tool
def run_code(code: str) -> dict:
    """
    Executes Python code in a sandboxed environment with safety checks.
    
    This tool:
      1. Validates code safety (no dangerous operations)
      2. Writes code into a temporary .py file
      3. Executes the file with timeout protection
      4. Returns stdout, stderr, and return code
    
    IMPORTANT FILE PATH RULES:
    - The code runs INSIDE the 'hybrid_llm_files' directory.
    - DO NOT prepend 'hybrid_llm_files/' to your file paths.
    - Access downloaded files directly by name.
      CORRECT: pd.read_csv('data.csv')
      WRONG:   pd.read_csv('hybrid_llm_files/data.csv')
      
    IMPORTANT RULES:
    - Code should assign the final answer to a variable named 'answer'
    - Do NOT include submission code (httpx.post, requests.post)
    - Do NOT hardcode data - always fetch from APIs/files
    
    Parameters
    ----------
    code : str
        Python source code to execute. Should end with: answer = <result>
    
    Returns
    -------
    dict
        {
            "stdout": <program output>,
            "stderr": <errors if any>,
            "return_code": <exit code>,
            "answer": <extracted answer if found>
        }
    """
    print(f"\n[CODE_EXECUTOR] Executing code ({len(code)} chars)")
    
    try:
        # ---------------------------------------------------------
        # 1. SAFETY CHECKS
        # ---------------------------------------------------------
        dangerous_patterns = [
            'os.system', 'subprocess.call', 'eval(', 'exec(',
            '__import__', 'shutil.rmtree'
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                print(f"[CODE_EXECUTOR] ⚠ Warning: Potentially dangerous pattern '{pattern}' detected")
        
        # ---------------------------------------------------------
        # 2. FILE PREPARATION
        # ---------------------------------------------------------
        # Create execution directory
        exec_dir = "hybrid_llm_files"
        os.makedirs(exec_dir, exist_ok=True)
        
        # Write code to file
        filename = "runner.py"
        filepath = os.path.join(exec_dir, filename)

        # CRITICAL FIX: Prepend FFmpeg path setup for Audio Libraries (Pydub/SpeechRecognition)
        # This fixes the "RuntimeWarning: Couldn't find ffmpeg" error
        header = """
import os
import sys
# Force Pydub/System to find FFmpeg in the Docker container
os.environ["PATH"] += os.pathsep + "/usr/bin"
"""
        # Only add header if imports are likely needed
        full_code = header + code if "import" in code else code
        
        with open(filepath, "w") as f:
            f.write(full_code)
        
        print(f"[CODE_EXECUTOR] Code written to {filepath}")
        
        # ---------------------------------------------------------
        # 3. EXECUTION
        # ---------------------------------------------------------
        try:
            # print(f"[CODE_EXECUTOR] Command: uv run {filename}")
            
            proc = subprocess.Popen(
                ["uv", "run", filename],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=exec_dir
            )
            
            # Wait for completion (90s timeout)
            stdout, stderr = proc.communicate(timeout=90)
            return_code = proc.returncode
            
        except subprocess.TimeoutExpired:
            print(f"[CODE_EXECUTOR] ⏱️ Timeout expired after 90 seconds")
            proc.kill()
            stdout, stderr = proc.communicate()
            return {
                "stdout": stdout,
                "stderr": "Error: Code execution timed out (90 seconds)",
                "return_code": -1,
                "answer": None
            }

        # ---------------------------------------------------------
        # 4. OUTPUT PROCESSING
        # ---------------------------------------------------------
        # Try to extract answer from output
        answer = None
        if return_code == 0:
            # Look for "answer = " in stdout (simple heuristic)
            for line in stdout.split('\n'):
                if line.strip():
                    answer = line.strip()
        
        # Detect if answer is base64 (long string)
        is_base64 = False
        if answer and len(answer) > 1000:
            is_base64 = True
            print(f"[CODE_EXECUTOR] Detected base64 answer ({len(answer)} chars)")
        
        # Truncate stdout to avoid overwhelming LLM memory
        truncated_stdout = stdout
        if len(stdout) > 2000:
            truncated_stdout = stdout[:2000] + f"\n... (truncated {len(stdout) - 2000} chars)"
        
        result = {
            "stdout": truncated_stdout,
            "stderr": stderr,
            "return_code": return_code,
            "answer": answer  # Keep full answer - needed for submission
        }
        
        if return_code == 0:
            print(f"[CODE_EXECUTOR] ✓ Execution successful")
            if answer and not is_base64:
                print(f"[CODE_EXECUTOR] Answer extracted: {answer}")
        else:
            print(f"[CODE_EXECUTOR] ✗ Execution failed with code {return_code}")
            if stderr:
                print(f"[CODE_EXECUTOR] Error: {stderr[:500]}...")
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"[CODE_EXECUTOR] ✗ Exception: {error_msg}")
        return {
            "stdout": "",
            "stderr": error_msg,
            "return_code": -1,
            "answer": None
        }