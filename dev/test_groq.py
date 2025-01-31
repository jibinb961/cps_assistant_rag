
from dotenv import load_dotenv
from groq import Groq
import os
import datetime

load_dotenv()

# Load API key from environment variable (set it before running)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("ERROR: GROQ_API_KEY must be set in the environment variables.")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Define system and user messages
system_prompt = "You are an AI assistant for the College of Professional Studies at Northeastern University. Please try to find the program name which the student is looking for only return a JSON object with only one item that is the name of the Program."
query = "What is the tuition fee for MPS in Applied Machine Intelligence program in Boston?"

# Call the Groq API with streaming enabled
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True,  # Enable streaming
)

# Print streamed response
print("\nAI Response:\n")
for chunk in completion:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

print("\n\n--- End of Response ---")
