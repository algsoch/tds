import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")

def query_gemini_api(question: str) -> str:
    """Query the Gemini API with the provided question."""
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    
    data = {
        "contents": [
            {
                "parts": [
                    {"text": question}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024,
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        
        # Extract the response text from the Gemini API response
        if "candidates" in result and len(result["candidates"]) > 0:
            if "content" in result["candidates"][0]:
                content = result["candidates"][0]["content"]
                if "parts" in content and len(content["parts"]) > 0:
                    return content["parts"][0]["text"]
        
        return "No clear answer found from the AI model."
    
    except Exception as e:
        print(f"Error querying Gemini API: {str(e)}")
        return f"Error fetching response from Gemini API: {str(e)}"