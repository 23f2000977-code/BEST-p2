"""
Image analysis tool with Automatic Local Color Calculation.
"""

from langchain_core.tools import tool
import os
import base64
from api_key_rotator import get_api_key_rotator

def get_dominant_color_local(image_path: str) -> str:
    """
    Mathematically calculate the dominant color using K-Means clustering.
    Returns Hex code (e.g., #ff0000).
    """
    try:
        from PIL import Image
        import numpy as np
        from sklearn.cluster import KMeans
        from collections import Counter

        # Open image and convert to RGB
        img = Image.open(image_path)
        img = img.convert("RGB")
        img = img.resize((100, 100)) # Resize for speed
        
        # Convert to numpy array
        img_array = np.array(img)
        # Reshape to list of pixels
        pixels = img_array.reshape(-1, 3)

        # Use KMeans to find top colors
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Count labels to find most frequent
        counts = Counter(kmeans.labels_)
        dominant_center = kmeans.cluster_centers_[counts.most_common(1)[0][0]]
        
        # Convert to Hex
        dominant_hex = "#{:02x}{:02x}{:02x}".format(
            int(dominant_center[0]),
            int(dominant_center[1]),
            int(dominant_center[2])
        )
        return dominant_hex
    except Exception as e:
        return f"Error calculating color: {str(e)}"

@tool
def analyze_image(image_url: str, question: str = "Describe this image in detail") -> str:
    """
    Analyze an image. 
    
    INTELLIGENT ROUTING:
    - If question specifically asks for 'dominant color' or 'heatmap', uses Math (100% accurate).
    - ALL OTHER QUESTIONS use the standard Gemini/OpenAI Vision API.
    """
    print(f"\n[IMAGE_ANALYZER] Analyzing image: {image_url}")
    print(f"[IMAGE_ANALYZER] Question: {question}")
    
    # 1. Download the image first
    from hybrid_tools.download_file import download_file
    local_path = download_file.invoke({"url": image_url})
    if "Error" in str(local_path) or not os.path.exists(local_path):
        return f"Failed to download image: {local_path}"
    
    # 2. CHECK: Is this a color/heatmap question?
    # Specific keywords to ensure we don't accidentally break normal questions
    target_phrases = ["most frequent color", "dominant color", "heatmap"]
    
    is_heatmap_url = "heatmap" in image_url.lower()
    is_color_question = any(phrase in question.lower() for phrase in target_phrases)

    if is_heatmap_url or is_color_question:
        print(f"[IMAGE_ANALYZER] ⚡ Detected Heatmap/Color task. Using Local Math.")
        result = get_dominant_color_local(local_path)
        print(f"[IMAGE_ANALYZER] ✓ Calculated Color: {result}")
        return f"The dominant color is {result}"

    # 3. Standard Vision API (Gemini/OpenAI) - FOR EVERYTHING ELSE
    try:
        with open(local_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        import mimetypes
        mime_type, _ = mimetypes.guess_type(local_path)
        if not mime_type: mime_type = "image/png"
        
        # Try GEMINI
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

            # Fallback to OpenAI
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