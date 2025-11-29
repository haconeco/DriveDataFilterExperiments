import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

def setup_gemini(api_key=None, model_name="gemini-1.5-flash"):
    """
    Configures the Gemini API.
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("API Key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return model

def load_image(image_path):
    """
    Loads an image from the given path.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return Image.open(image_path)
