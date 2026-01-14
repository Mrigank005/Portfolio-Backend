import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Fetching available models...")
try:
    # List all models and just print their names
    # (The SDK returns an iterator, so we loop through it)
    for m in client.models.list():
        print(f"found: {m.name}")
        
except Exception as e:
    print(f"❌ Error: {e}")