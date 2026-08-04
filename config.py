import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Light, fast, free-tier-friendly Groq models
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3-turbo")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

SERVICE_CATEGORIES = [
    "Emergency Services",
    "Healthcare",
    "Home Maintenance",
    "Transportation",
    "Cleaning",
    "General Services",
]

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Create a .env file (see .env.example) "
        "with a free Groq API key from https://console.groq.com/keys"
    )
