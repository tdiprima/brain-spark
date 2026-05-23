import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:latest")
PROFILE_FILE = os.environ.get("BRAIN_SPARK_PROFILE", "user_profile.json")
