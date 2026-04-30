"""
core/config.py
--------------
Centralised configuration for Al-Huda Islamic AI Assistant.
Loads environment variables, resolves paths, and exposes all
app-wide constants in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# This file lives at app/core/config.py
# _APP_ROOT  → app/
# _PROJECT_ROOT → project root (Al-Huda/)
_APP_ROOT: Path = Path(__file__).parent.parent.resolve()
_PROJECT_ROOT: Path = _APP_ROOT.parent

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
load_dotenv(_PROJECT_ROOT / ".env")

MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
if not MISTRAL_API_KEY:
    raise RuntimeError("MISTRAL_API_KEY is missing — add it to your .env file.")

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATABASE_FOLDER: str = str(_PROJECT_ROOT / "DataBase")
STATIC_DIR: Path = _APP_ROOT / "static"
TEMPLATES_DIR: Path = _APP_ROOT / "templates"
UPLOAD_DIR: Path = _PROJECT_ROOT / "uploads" / "profiles"
UPLOADS_BASE_DIR: Path = _PROJECT_ROOT / "uploads"
VECTOR_DB_PATH: str = str(_PROJECT_ROOT / "vector_db.pkl")

# ---------------------------------------------------------------------------
# Mistral AI settings
# ---------------------------------------------------------------------------
MISTRAL_MODEL: str = "mistral-large-latest"

GENERATION_CONFIG: dict = {
    "temperature": float(os.getenv("TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("MAX_TOKENS", "1024")),
    "top_p": float(os.getenv("TOP_P", "0.95")),
}

# ---------------------------------------------------------------------------
# Application limits
# ---------------------------------------------------------------------------
MAX_CHAT_MESSAGES: int = 20       # Max messages kept per conversation
MAX_MESSAGE_LENGTH: int = 5000    # Max characters per user message
MAX_RETRIES: int = 3              # Retry attempts for Mistral API calls
RETRY_DELAY: float = 2.0          # Base delay (seconds) between retries
REQUEST_TIMEOUT: int = 30         # HTTP request timeout for web scraping

# ---------------------------------------------------------------------------
# Supported document extensions
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS: list = [".pdf", ".txt", ".docx", ".doc"]

# ---------------------------------------------------------------------------
# Optional dependency flags (resolved at import time)
# ---------------------------------------------------------------------------
try:
    import docx  # noqa: F401
    DOCX_AVAILABLE: bool = True
except ImportError:
    DOCX_AVAILABLE: bool = False

    #Nothing JUST Testing
