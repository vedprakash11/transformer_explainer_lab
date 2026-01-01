from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Read API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("GROQ_API_KEY:", GROQ_API_KEY)