# 🕌 Al-Huda — Islamic AI Assistan

Al-Huda is an AI-powered Islamic knowledge assistant that provides accurate, verified information from the Quran and Hadith. It uses a **Retrieval-Augmented Generation (RAG)** approach — your questions are matched against 145,000+ indexed passages from authentic Islamic texts before being answered by Mistral AI.

---

## Features

- **Semantic Search** — Finds relevant Quran verses and Hadith from a FAISS vector index built from authentic PDFs
- **Mistral AI Chat** — Context-aware answers powered by `mistral-large-latest`
- **Streaming Responses** — Real-time Server-Sent Events streaming for a fluid chat experience
- **Citation Integrity** — Automatically flags responses that mention Islamic sources without proper references
- **Web Scraping** — Can extract and summarise content from URLs you paste into the chat
- **User Profiles** — Avatar upload and display name, stored for the session
- **Premium UI** — Single-page dark-mode app served directly from FastAPI

---

## Project Structure

```
Al-Huda/
├── app/
│   ├── main.py               # FastAPI entry point (lifespan, mounts, router wiring)
│   ├── core/
│   │   ├── config.py         # All constants and env-var loading (single source of truth)
│   │   └── logging_config.py # Rotating file + console logger
│   ├── api/
│   │   ├── models.py         # Pydantic request/response schemas
│   │   └── routes/
│   │       ├── chat.py       # /chat and /chat-stream endpoints
│   │       ├── profile.py    # /api/profile/* endpoints
│   │       └── database.py   # /health, /stats, /rescan-database endpoints
│   ├── services/
│   │   ├── vector_db.py      # FAISS vector store (async add/search, auto-save)
│   │   ├── document_processor.py  # PDF/DOCX/TXT ingestion and chunking
│   │   ├── mistral_client.py # Mistral SDK singleton + retry wrapper
│   │   ├── chat_memory.py    # Per-conversation message history
│   │   ├── rate_limiter.py   # Sliding-window rate limiter + retry decorator
│   │   └── web_scraper.py    # Async URL scraping + Mistral extraction
│   ├── templates/
│   │   └── index.html        # Frontend SPA
│   └── static/
│       ├── css/              # Stylesheets
│       └── js/               # formatter.js
├── DataBase/                 # Source PDFs (Sahih Bukhari, Sahih Muslim volumes)
├── uploads/                  # User avatar uploads (auto-created)
├── vector_db.pkl             # Persisted FAISS index (auto-generated, ~500 MB)
├── .env                      # API key and config (git-ignored)
├── requirements.txt          # Python dependencies
├── run.bat                   # One-click launcher (Windows)
└── setup.bat                 # First-time setup (venv + pip install)
```

---

## Quick Start (Windows)

### 1. First-time setup

```cmd
setup.bat
```

Creates the venv, installs all dependencies, and downloads the embedding model.

### 2. Add your Mistral API key

Edit `.env` and set:

```env
MISTRAL_API_KEY=your_key_here
```

Get a key at [console.mistral.ai](https://console.mistral.ai/).

### 3. Run

```cmd
run.bat
```

The server starts at **http://127.0.0.1:8000** and the browser opens automatically.

---

## Manual Launch

Always run from the **project root** (`Al-Huda/`):

```powershell
# With venv active:
.\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> ⚠️ **Do not** run `python app/main.py` or `cd app && python main.py` — package imports will fail.

---

## API Endpoints

| Method   | Path                           | Description                       |
| -------- | ------------------------------ | --------------------------------- |
| `GET`    | `/`                            | Frontend SPA                      |
| `POST`   | `/chat`                        | Full chat response (JSON)         |
| `POST`   | `/chat-stream`                 | Streaming chat (SSE)              |
| `GET`    | `/health`                      | System health snapshot            |
| `GET`    | `/stats`                       | Vector DB + chat memory stats     |
| `POST`   | `/rescan-database`             | Re-index all files in `DataBase/` |
| `DELETE` | `/clear-conversation/{id}`     | Clear a conversation's history    |
| `GET`    | `/api/profile/{user_id}`       | Get user profile                  |
| `POST`   | `/api/profile/{user_id}`       | Update user profile               |
| `POST`   | `/api/profile/{user_id}/image` | Upload avatar image               |

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Requirements

- Python 3.9+
- Mistral API key
- Windows (for `run.bat` / `setup.bat`; manual uvicorn command works on any OS)

---

## License

Licensed under the MIT License — see [LISENCE.txt](LISENCE.txt) for details.
