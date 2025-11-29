"""
Image analysis tool using Gemini Vision API.
UPDATED: Uses API Key Rotator to avoid rate limits.
"""

from langchain_core.tools import tool
import os
import base64
from typing import Optional
# Import the rotator
from api_key_rotator import get_api_key_rotator

@tool
def analyze_image(image_url: str, question: str = "Describe this image in detail") -> str:
    """
    Analyze an image using Gemini Vision API.
    """
    print(f"\n[IMAGE_ANALYZER] Analyzing image: {image_url}")
    print(f"[IMAGE_ANALYZER] Question: {question}")
    
    # 1. Prepare Image Path
    if image_url.startswith('http'):
        from hybrid_tools.download_file import download_file
        local_path = download_file.invoke({"url": image_url})
        if "Error" in local_path:
            return f"Failed to download image: {local_path}"
        image_path = local_path
    else:
        image_path = image_url

    # 2. Encode Image
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        import mimetypes
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type: mime_type = "image/png"
    except Exception as e:
        return f"Error reading image file: {e}"

    # 3. Try GEMINI (With Rotator)
    try:
        from openai import OpenAI
        rotator = get_api_key_rotator()
        
        # KEY FIX: Get the CURRENT valid key from rotator, not just env var
        api_key = rotator.get_current_key()
        
        print(f"[IMAGE_ANALYZER] Using Gemini Key: ...{api_key[-4:]}")

        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                        }
                    ]
                }
            ],
        )
        return response.choices[0].message.content

    except Exception as e:
        error_msg = str(e)
        print(f"[IMAGE_ANALYZER] ⚠️ Gemini Vision Failed: {error_msg[:100]}")
        
        # If it was a quota error, mark that key as dead immediately
        if "429" in error_msg or "quota" in error_msg.lower():
            rotator.mark_key_exhausted(api_key)
            print(f"[IMAGE_ANALYZER] Marked key ...{api_key[-4:]} as exhausted.")

        # 4. Fallback to OpenAI (GPT-4o)
        print(f"[IMAGE_ANALYZER] 🔄 Falling back to OpenAI Vision...")
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini", # Cheaper/Faster fallback
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                            }
                        ]
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as openai_e:
            return f"Error analyzing image (Both APIs failed): {openai_e}"