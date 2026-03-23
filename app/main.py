"""
main.py
-------
Al-Huda Islamic AI Assistant — application entry point.

This file is intentionally kept slim. All logic lives in:
  app/core/     — configuration and logging
  app/services/ — business logic (vector DB, Mistral, document processing…)
  app/api/      — FastAPI route handlers and Pydantic models
"""

import asyncio
import logging
import os
import threading
import webbrowser
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Bootstrap logging before any other import writes to the logger
from app.core.logging_config import setup_logging
setup_logging()

from app.core.config import (
    MISTRAL_MODEL,
    DATABASE_FOLDER,
    MAX_CHAT_MESSAGES,
    STATIC_DIR,
    TEMPLATES_DIR,
    UPLOAD_DIR,
    UPLOADS_BASE_DIR,
)
from app.api.routes import chat, database, profile
from app.services.vector_db import vector_db
from app.services.document_processor import scan_database_folder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure required directories exist
# ---------------------------------------------------------------------------
for _dir in (STATIC_DIR, TEMPLATES_DIR, UPLOAD_DIR, UPLOADS_BASE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log startup info and kick off database scan without blocking."""
    logger.info("=" * 60)
    logger.info("🕌 Al-Huda Islamic AI Assistant v4.0 starting…")
    logger.info(f"   Mistral model : {MISTRAL_MODEL}")
    logger.info(f"   Vector DB     : {len(vector_db.texts)} entries")
    logger.info(f"   DataBase dir  : {DATABASE_FOLDER}")
    logger.info(f"   Max msg/chat  : {MAX_CHAT_MESSAGES}")
    logger.info("=" * 60)
    
    # Start background tasks
    asyncio.create_task(_scan_database_background())
    
    # Open browser in background after a small delay
    def open_browser():
        import time
        time.sleep(1)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception as exc:
            logger.warning(f"Could not open browser automatically: {exc}")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    yield
    # (shutdown logic can be added here in the future)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Al-Huda — Islamic AI Assistant",
    description="AI-powered Islamic knowledge assistant powered by Mistral AI",
    version="4.0.0",
    lifespan=lifespan,
)

# CORS — restrict to known origins (override via ALLOWED_ORIGINS env var)
_default_origins = ["http://127.0.0.1:8000", "http://localhost:8000"]
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()] or _default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static file mounts
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_BASE_DIR)), name="uploads")

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(chat.router)
app.include_router(profile.router)
app.include_router(database.router)


# ---------------------------------------------------------------------------
# Root — serve the frontend SPA
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_home():
    """Serve the main HTML frontend."""
    from fastapi import HTTPException
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        logger.error(f"index.html not found at {html_path}")
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse(str(html_path))


async def _scan_database_background():
    """Run the DataBase folder scan after the server has fully started."""
    await asyncio.sleep(1)
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, scan_database_folder)
        logger.info(f"Database scan complete — status: {result['status']}")
    except Exception as exc:
        logger.error(f"Database scan failed: {exc}")

