"""
Multi-fallback audio transcription tool.
UPDATED: Uses API Key Rotator.
"""

from langchain_core.tools import tool
import os
import requests
# Import rotator
from api_key_rotator import get_api_key_rotator

@tool
def transcribe_audio(audio_url: str) -> str:
    """
    Transcribe audio file using multi-fallback approach.
    """
    print(f"\n[AUDIO] Transcribing: {audio_url}")
    
    try:
        # Download audio
        from hybrid_tools.download_file import download_file
        audio_path = download_file.invoke({"url": audio_url})
        if "Error" in audio_path: return f"Failed: {audio_path}"
        
        # 1. Try SpeechRecognition (Fastest/Free)
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                return recognizer.recognize_google(audio_data)
        except:
            pass

        # 2. Try Gemini (With Rotator)
        try:
            import base64
            from openai import OpenAI
            
            rotator = get_api_key_rotator()
            # KEY FIX: Use rotated key
            api_key = rotator.get_current_key()
            
            print(f"[AUDIO] Using Gemini Key: ...{api_key[-4:]}")
            
            client = OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            
            with open(audio_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode('utf-8')
            
            response = client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe exactly."},
                        {"type": "input_audio", "input_audio": {"data": b64_data, "format": "mp3"}}
                    ]
                }]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[AUDIO] Gemini failed: {e}")
            if "429" in str(e):
                rotator.mark_key_exhausted(api_key)

        # 3. Fallback to Whisper
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            with open(audio_path, "rb") as f:
                return client.audio.transcriptions.create(model="whisper-1", file=f).text
        except Exception as e:
            return f"Error: All methods failed. {e}"
            
    except Exception as e:
        return f"Error: {str(e)}"