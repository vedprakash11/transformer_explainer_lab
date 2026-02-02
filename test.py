from dotenv import load_dotenv
import os
groq_api_key = os.getenv("GROQ_API_KEY")
print("groq_api_key: ", groq_api_key)