import os
from dotenv import load_dotenv

load_dotenv()

# Configuration variables loaded from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Low-resource / low-bandwidth mode
# Enable by setting LOW_RESOURCE_MODE=1 in environment.
# If not explicitly set, default to ON for remote/server deployments:
#   set ROBIN_DEPLOYMENT=production (or: prod/remote/server)
_deployment = os.getenv("ROBIN_DEPLOYMENT", os.getenv("DEPLOYMENT", "")).strip().lower()
_default_low_resource = "1" if _deployment in {"prod", "production", "remote", "server"} else "0"
LOW_RESOURCE_MODE = os.getenv("LOW_RESOURCE_MODE", _default_low_resource) == "1"
# Optional tuning knobs (with safe defaults)
LOW_RESOURCE_THREADS = int(os.getenv("LOW_RESOURCE_THREADS", "2"))
LOW_RESOURCE_MAX_ENDPOINTS = int(os.getenv("LOW_RESOURCE_MAX_ENDPOINTS", "3"))
LOW_RESOURCE_SCRAPE_MAX_CHARS = int(os.getenv("LOW_RESOURCE_SCRAPE_MAX_CHARS", "800"))