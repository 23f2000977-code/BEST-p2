"""
Standard Image Analysis tool using Gemini/OpenAI Vision.
"""

from langchain_core.tools import tool
import os
import base64
from api_key_rotator import get_api_key_rotator

@tool
def analyze_image(image_url: str, question: str = "Describe this image in detail") -> str:
    """
    Analyze an image using AI Vision APIs.
    
    Use this for:
    - Reading text from images (OCR)
    - Describing scenes or objects
    - Interpreting charts/graphs semantically
    
    DO NOT use this for:
    - Counting pixels exactly
    - Finding exact hex codes
    (For those, use the 'run_code' tool with the 'Pillow' library instead).
    """
    print(f"\n[IMAGE_ANALYZER] Analyzing image: {image_url}")
    print(f"[IMAGE_ANALYZER] Question: {question}")
    
    # 1. Download the image first
    from hybrid_tools.download_file import download_file
    local_path = download_file.invoke({"url": image_url})
    if "Error" in str(local_path) or not os.path.exists(local_path):
        return f"Failed to download image: {local_path}"
    
    # 2. Prepare Image
    try:
        with open(local_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        import mimetypes
        mime_type, _ = mimetypes.guess_type(local_path)
        if not mime_type: mime_type = "image/png"
        
        # 3. Try GEMINI
        try:
            from openai import OpenAI
            rotator = get_api_key_rotator()
            api_key = rotator.get_current_key()
            print(f"[IMAGE_ANALYZER] Using Gemini Key: ...{api_key[-4:]}")

            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                    ]
                }]
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"[IMAGE_ANALYZER] ⚠️ Gemini Failed: {str(e)[:100]}")
            if "429" in str(e): rotator.mark_key_exhausted(api_key)

            # 4. Fallback to OpenAI
            print(f"[IMAGE_ANALYZER] 🔄 Falling back to OpenAI Vision...")
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                    ]
                }]
            )
            return response.choices[0].message.content

    except Exception as e:
        return f"Error analyzing image: {e}"