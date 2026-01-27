from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
from dotenv import load_dotenv
import os
import sys
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import traceback
import PyPDF2
import io
import requests
from bs4 import BeautifulSoup
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
from pathlib import Path
import asyncio
from functools import wraps
import time
from fastapi.responses import StreamingResponse
import json
import tempfile
import shutil
import uuid

# Text-to-Speech support (commented out per user request)
# try:
#     import pyttsx3
#     TTS_AVAILABLE = True
# except ImportError:
#     TTS_AVAILABLE = False
#     logging.warning("pyttsx3 not installed. TTS greeting will be unavailable.")
TTS_AVAILABLE = False  # TTS is disabled

# Add DOCX support
try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logging.warning("python-docx not installed. Install with: pip install python-docx")

# Mistral AI import
from mistralai import Mistral

# --------------------------------------------------
# Configure Logging with Rotation
# --------------------------------------------------
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configure logging with rotation and better formatting"""
    # Get project root for log file path
    _log_project_root = Path(__file__).parent.parent.resolve()
    log_file_path = str(_log_project_root / 'app.log')
    
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

setup_logging()
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
# Load .env from project root (parent of app directory)
_env_project_root = Path(__file__).parent.parent.resolve()
load_dotenv(_env_project_root / ".env")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    logger.error("MISTRAL_API_KEY is missing in .env file")
    raise RuntimeError("MISTRAL_API_KEY is missing in .env")

logger.info("Environment variables loaded successfully")

# --------------------------------------------------
# Constants
# --------------------------------------------------
# Resolve paths relative to this file's location
_APP_ROOT = Path(__file__).parent.resolve()
_PROJECT_ROOT = _APP_ROOT.parent
DATABASE_FOLDER = str(_PROJECT_ROOT / "DataBase")
MAX_CHAT_MESSAGES = 20
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.doc']
MAX_MESSAGE_LENGTH = 5000
MAX_RETRIES = 3
RETRY_DELAY = 2
REQUEST_TIMEOUT = 30

# Mistral Model configuration
MISTRAL_MODEL = "mistral-large-latest"  # or "mistral-medium-latest", "mistral-small-latest"
GENERATION_CONFIG = {
    'temperature': 0.7,
    'max_tokens': 4096,  # Increased for complete responses
    'top_p': 0.95
}

# --------------------------------------------------
# Rate Limiter
# --------------------------------------------------
class RateLimiter:
    """Simple rate limiter to prevent API quota exhaustion"""
    
    def __init__(self, max_requests: int = 60, window: int = 60):
        self.max_requests = max_requests
        self.window = window
        self.requests = []
    
    def is_allowed(self) -> bool:
        """Check if request is allowed"""
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < self.window]
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
    
    def get_wait_time(self) -> float:
        """Get time to wait before next request"""
        if not self.requests:
            return 0
        oldest = min(self.requests)
        return max(0, self.window - (time.time() - oldest))

rate_limiter = RateLimiter(max_requests=60, window=60)

# --------------------------------------------------
# Retry Decorator with Exponential Backoff
# --------------------------------------------------
def retry_with_backoff(max_retries: int = MAX_RETRIES, base_delay: float = RETRY_DELAY):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries reached for {func.__name__}: {e}")
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
            
        return wrapper
    return decorator

# --------------------------------------------------
# Chat Memory Manager (Enhanced)
# --------------------------------------------------
class ChatMemoryManager:
    """Enhanced chat history manager with better context management"""
    
    def __init__(self, max_messages: int = MAX_CHAT_MESSAGES):
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.max_messages = max_messages
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Chat Memory Manager initialized (max {max_messages} messages per chat)")
    
    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation history with optional metadata"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            self.conversation_metadata[conversation_id] = {
                'created_at': datetime.now().isoformat(),
                'message_count': 0
            }
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        if metadata:
            message['metadata'] = metadata
        
        self.conversations[conversation_id].append(message)
        self.conversation_metadata[conversation_id]['message_count'] += 1
        self.conversation_metadata[conversation_id]['updated_at'] = datetime.now().isoformat()
        
        # Enforce message limit (keep most recent messages)
        if len(self.conversations[conversation_id]) > self.max_messages:
            removed = self.conversations[conversation_id].pop(0)
            logger.info(f"Removed oldest message from conversation {conversation_id}")
    
    def get_history(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history with optional limit"""
        history = self.conversations.get(conversation_id, [])
        if limit:
            return history[-limit:]
        return history
    
    def clear_conversation(self, conversation_id: str):
        """Clear specific conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            if conversation_id in self.conversation_metadata:
                del self.conversation_metadata[conversation_id]
            logger.info(f"Cleared conversation {conversation_id}")
    
    def get_formatted_history(self, conversation_id: str, limit: int = 5) -> List[Dict]:
        """Get formatted chat history for Mistral API"""
        history = self.get_history(conversation_id)
        if not history:
            return []
        
        # Convert to Mistral message format
        formatted = []
        for msg in history[-limit:]:
            formatted.append({
                "role": msg['role'],
                "content": msg['content']
            })
        return formatted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            'total_conversations': len(self.conversations),
            'total_messages': sum(len(conv) for conv in self.conversations.values()),
            'active_conversations': len([c for c in self.conversation_metadata.values() 
                                        if (datetime.now() - datetime.fromisoformat(c['updated_at'])).seconds < 3600])
        }

chat_memory = ChatMemoryManager()
active_streams = {}  # Track active streaming responses

# --------------------------------------------------
# Vector Database Class (Enhanced)
# --------------------------------------------------
class VectorDatabase:
    """Enhanced Vector Database with better error handling and performance"""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', index_file: str = None):
        # Use absolute path for vector_db.pkl in project root
        if index_file is None:
            index_file = str(_PROJECT_ROOT / 'vector_db.pkl')
        
        logger.info(f"Initializing Vector Database with model: {embedding_model}")
        
        try:
            self.model = SentenceTransformer(embedding_model)
            self.dimension = 384
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts: List[str] = []
            self.metadata: List[Dict] = []
            self.index_file = index_file
            self._lock = asyncio.Lock()
            
            self.load()
            
            logger.info(f"Vector Database initialized with {len(self.texts)} existing entries")
        except Exception as e:
            logger.error(f"Failed to initialize Vector Database: {e}")
            raise
    
    async def add_texts_async(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """Async version of add_texts for better performance"""
        if not texts:
            logger.warning("No texts provided to add")
            return
        
        # Filter empty texts
        texts = [t for t in texts if t.strip()]
        if not texts:
            logger.warning("All texts were empty after filtering")
            return
        
        async with self._lock:
            try:
                logger.info(f"Adding {len(texts)} texts to vector database...")
                
                # Generate embeddings in a thread pool
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None, 
                    lambda: self.model.encode(texts, show_progress_bar=False)
                )
                
                # Add to FAISS index
                self.index.add(np.array(embeddings).astype('float32'))
                
                # Store texts
                self.texts.extend(texts)
                
                # Store metadata
                if metadata:
                    self.metadata.extend(metadata[:len(texts)])
                else:
                    self.metadata.extend([{"source": "unknown", "type": "text"}] * len(texts))
                
                logger.info(f"Vector database now contains {len(self.texts)} entries")
                
                # Auto-save
                await loop.run_in_executor(None, self.save)
                
            except Exception as e:
                logger.error(f"Error adding texts to vector database: {e}")
                raise
    
    def add_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        """Sync version for backward compatibility"""
        if not texts:
            return
        
        # Filter empty texts
        texts = [t for t in texts if t.strip()]
        if not texts:
            return
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            self.index.add(np.array(embeddings).astype('float32'))
            self.texts.extend(texts)
            
            if metadata:
                self.metadata.extend(metadata[:len(texts)])
            else:
                self.metadata.extend([{"source": "unknown", "type": "text"}] * len(texts))
            
            self.save()
        except Exception as e:
            logger.error(f"Error adding texts: {e}")
            raise
    
    async def search_async(self, query: str, k: int = 5, min_score: float = 0.3) -> List[Dict]:
        """Async search with minimum score threshold"""
        if len(self.texts) == 0:
            logger.warning("Vector database is empty")
            return []
        
        try:
            logger.info(f"Searching for: '{query[:50]}...' (top {k} results)")
            
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None,
                lambda: self.model.encode([query])
            )
            
            k_search = min(k, len(self.texts))
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), 
                k_search
            )
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.texts):
                    similarity_score = 1 / (1 + float(distance))
                    if similarity_score >= min_score:
                        results.append({
                            'text': self.texts[idx],
                            'metadata': self.metadata[idx],
                            'distance': float(distance),
                            'similarity_score': similarity_score
                        })
            
            logger.info(f"Found {len(results)} relevant results (score >= {min_score})")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {e}")
            return []
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Sync search for backward compatibility"""
        if len(self.texts) == 0:
            return []
        
        try:
            query_embedding = self.model.encode([query])
            k_search = min(k, len(self.texts))
            distances, indices = self.index.search(
                np.array(query_embedding).astype('float32'), 
                k_search
            )
            
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.texts):
                    results.append({
                        'text': self.texts[idx],
                        'metadata': self.metadata[idx],
                        'distance': float(distance),
                        'similarity_score': 1 / (1 + float(distance))
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def save(self):
        """Save vector database to file"""
        try:
            data = {
                'texts': self.texts,
                'metadata': self.metadata,
                'index': faiss.serialize_index(self.index)
            }
            with open(self.index_file, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Vector database saved to {self.index_file}")
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
    
    def load(self):
        """Load vector database from file"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'rb') as f:
                    data = pickle.load(f)
                self.texts = data['texts']
                self.metadata = data['metadata']
                self.index = faiss.deserialize_index(data['index'])
                logger.info(f"Vector database loaded: {len(self.texts)} entries")
            except Exception as e:
                logger.error(f"Error loading vector database: {e}")
                logger.info("Creating new vector database...")
        else:
            logger.info(f"No existing vector database found")
    
    def clear(self):
        """Clear all data from vector database"""
        logger.info("Clearing vector database...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []
        self.save()
        logger.info("Vector database cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        sources = {}
        types = {}
        
        for meta in self.metadata:
            source = meta.get('source', 'unknown')
            type_val = meta.get('type', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            types[type_val] = types.get(type_val, 0) + 1
        
        return {
            'total_entries': len(self.texts),
            'sources': sources,
            'types': types,
            'dimension': self.dimension
        }

# --------------------------------------------------
# Document Processing Functions (Enhanced)
# --------------------------------------------------
def extract_text_from_pdf(file_content: bytes) -> tuple:
    """Extract text from PDF with enhanced error handling and cleaning"""
    try:
        logger.info("Extracting text from PDF...")
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        successful_pages = 0
        
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                # Enhanced check: Skip pages with only whitespace or very minimal content
                if page_text and len(page_text.strip()) > 10:
                    # Clean up the text: normalize whitespace and remove artifacts
                    cleaned_text = ' '.join(page_text.split())
                    text += f"\n\n--- Page {page_num + 1} ---\n{cleaned_text}"
                    successful_pages += 1
                elif page_text and page_text.strip():
                    logger.debug(f"Skipping page {page_num + 1}: insufficient content")
            except Exception as e:
                logger.warning(f"Error extracting page {page_num + 1}: {e}")
                continue
        
        if not text.strip():
            return "", False, "No meaningful text could be extracted from PDF"
        
        logger.info(f"Extracted {len(text)} characters from PDF ({successful_pages}/{len(pdf_reader.pages)} pages with content)")
        return text, True, None
        
    except Exception as e:
        error = f"Error extracting PDF text: {str(e)}"
        logger.error(error)
        return "", False, error

def extract_text_from_docx(file_path: str) -> tuple:
    """Extract text from DOCX with better error handling"""
    if not DOCX_AVAILABLE:
        return "", False, "DOCX support not available. Install python-docx"
    
    try:
        logger.info(f"Extracting text from DOCX: {file_path}")
        
        doc = DocxDocument(file_path)
        text = ""
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        
        if not text.strip():
            return "", False, "No text could be extracted from DOCX"
        
        logger.info(f"Extracted {len(text)} characters from DOCX")
        return text, True, None
        
    except Exception as e:
        error = f"Error extracting DOCX text: {str(e)}"
        logger.error(error)
        return "", False, error

def extract_text_from_txt(file_path: str) -> tuple:
    """Extract text from TXT with encoding detection"""
    try:
        logger.info(f"Reading TXT file: {file_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                logger.info(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            return "", False, "Could not decode file with any supported encoding"
        
        if not text.strip():
            return "", False, "TXT file is empty"
        
        logger.info(f"Read {len(text)} characters from TXT file")
        return text, True, None
        
    except Exception as e:
        error = f"Error reading TXT file: {str(e)}"
        logger.error(error)
        return "", False, error

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Smart text chunking with paragraph and sentence boundary awareness"""
    if not text or not text.strip():
        return []
    
    # First try to split by paragraphs (preserve semantic coherence)
    paragraphs = re.split(r'\n\n+|\r\n\r\n+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Split paragraph into sentences for finer control
        sentences = re.split(r'(?<=[.!?؟])\s+', paragraph)  # Added Arabic question mark
        
        for sentence in sentences:
            words = sentence.split()
            sentence_length = len(words)
            
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_content = ' '.join(current_chunk)
                if len(chunk_content.strip()) > 30:  # Increased minimum
                    chunks.append(chunk_content.strip())
                
                # Start new chunk with smart overlap (prefer sentence boundaries)
                overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_words + words
                current_length = len(current_chunk)
            else:
                current_chunk.extend(words)
                current_length += sentence_length
    
    # Add last chunk
    if current_chunk:
        chunk_content = ' '.join(current_chunk)
        if len(chunk_content.strip()) > 30:  # Increased minimum
            chunks.append(chunk_content.strip())
    
    logger.info(f"Created {len(chunks)} semantic chunks from text")
    return chunks

def process_file(file_path: str) -> tuple:
    """Process a single file based on its extension"""
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.pdf':
        with open(file_path, 'rb') as f:
            content = f.read()
        text, success, error = extract_text_from_pdf(content)
        return text, success, error, 'pdf'
    
    elif file_ext == '.txt':
        text, success, error = extract_text_from_txt(file_path)
        return text, success, error, 'txt'
    
    elif file_ext in ['.docx', '.doc']:
        text, success, error = extract_text_from_docx(file_path)
        return text, success, error, 'docx'
    
    else:
        return "", False, f"Unsupported file type: {file_ext}", 'unknown'

def scan_database_folder() -> Dict[str, Any]:
    """Scan DataBase folder and process all supported documents"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Scanning DataBase folder: {DATABASE_FOLDER}")
    logger.info(f"{'='*60}")
    
    db_path = Path(DATABASE_FOLDER)
    
    if not db_path.exists():
        logger.warning(f"DataBase folder not found. Creating: {DATABASE_FOLDER}")
        db_path.mkdir(parents=True, exist_ok=True)
        return {
            'status': 'created',
            'message': f'Created empty DataBase folder at {DATABASE_FOLDER}',
            'files_processed': 0
        }
    
    if not db_path.is_dir():
        error_msg = f"{DATABASE_FOLDER} exists but is not a directory"
        logger.error(error_msg)
        return {
            'status': 'error',
            'message': error_msg,
            'files_processed': 0
        }
    
    all_files = []
    for ext in SUPPORTED_EXTENSIONS:
        all_files.extend(db_path.glob(f'*{ext}'))
    
    if not all_files:
        logger.info(f"No supported files found in {DATABASE_FOLDER}")
        return {
            'status': 'empty',
            'message': f'No supported files found in {DATABASE_FOLDER}',
            'files_processed': 0
        }
    
    logger.info(f"Found {len(all_files)} file(s) to process")
    
    results = {
        'successful': [],
        'failed': [],
        'total_chunks': 0
    }
    
    # Collect all texts and metadata first
    all_texts = []
    all_metadata = []
    
    for file_path in all_files:
        logger.info(f"\nProcessing: {file_path.name}")
        
        text, success, error, file_type = process_file(str(file_path))
        
        if success and text:
            chunks = chunk_text(text, chunk_size=400, overlap=50)
            
            if chunks:
                metadata = {
                    "source": file_path.name,
                    "type": f"database_{file_type}",
                    "file_path": str(file_path),
                    "processed_at": datetime.now().isoformat()
                }
                
                all_texts.extend(chunks)
                all_metadata.extend([metadata] * len(chunks))
                
                results['successful'].append({
                    'file': file_path.name,
                    'chunks': len(chunks),
                    'characters': len(text)
                })
                results['total_chunks'] += len(chunks)
                
                logger.info(f"Successfully processed {file_path.name} ({len(chunks)} chunks)")
            else:
                results['failed'].append({
                    'file': file_path.name,
                    'error': 'No meaningful content extracted'
                })
        else:
            results['failed'].append({
                'file': file_path.name,
                'error': error or 'Unknown error'
            })
            logger.error(f"Failed to process {file_path.name}: {error}")
    
    # Add all texts at once (FIXED: write once instead of in loop)
    if all_texts:
        try:
            vector_db.add_texts(all_texts, all_metadata)
            logger.info(f"Added {len(all_texts)} chunks to vector database")
        except Exception as e:
            logger.error(f"Error adding texts to vector database: {e}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Database Scan Summary:")
    logger.info(f"   Total files: {len(all_files)}")
    logger.info(f"   Successful: {len(results['successful'])}")
    logger.info(f"   Failed: {len(results['failed'])}")
    logger.info(f"   Total chunks: {results['total_chunks']}")
    logger.info(f"   Vector DB entries: {len(vector_db.texts)}")
    logger.info(f"{'='*60}\n")
    
    return {
        'status': 'success',
        'files_found': len(all_files),
        'files_processed': len(results['successful']),
        'files_failed': len(results['failed']),
        'total_chunks': results['total_chunks'],
        'successful_files': results['successful'],
        'failed_files': results['failed'],
        'total_db_entries': len(vector_db.texts)
    }

# --------------------------------------------------
# Web Scraping Functions (Enhanced)
# --------------------------------------------------
async def scrape_website_async(url: str, max_chars: int = 50000) -> tuple:
    """Async web scraping with better error handling"""
    try:
        logger.info(f"Scraping website: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form']):
            element.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = '\n'.join(lines)
        
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters")
        
        logger.info(f"Scraped {len(text)} characters from {url}")
        return text, True, None
        
    except requests.Timeout:
        error = f"Timeout while scraping {url}"
        logger.error(error)
        return "", False, error
    except Exception as e:
        error = f"Error scraping {url}: {str(e)}"
        logger.error(error)
        return "", False, error

def detect_urls_in_message(message: str) -> List[str]:
    """Detect URLs in user message"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, message)
    return list(set(urls))

async def extract_web_content_via_mistral(urls: List[str], user_query: str) -> str:
    """Use Mistral to extract relevant information from URLs"""
    if not urls:
        return ""
    
    logger.info(f"Extracting web content from {len(urls)} URL(s)")
    
    extracted_info = []
    
    for url in urls[:3]:
        try:
            text, success, error = await scrape_website_async(url, max_chars=5000)
            
            if success and text:
                extraction_prompt = f"""Extract ONLY information relevant to this query from the web content.

USER QUERY: {user_query}

WEB CONTENT FROM {url}:
{text[:3000]}

Instructions:
- Extract only relevant information
- Be concise and factual
- If not relevant, say "No relevant information found"

RELEVANT INFORMATION:"""

                try:
                    response = await mistral_client.chat.complete_async(
                        model=MISTRAL_MODEL,
                        messages=[{"role": "user", "content": extraction_prompt}],
                        temperature=0.3,
                        max_tokens=1000
                    )
                    
                    if response and response.choices:
                        extracted = response.choices[0].message.content
                        if extracted and "No relevant information found" not in extracted:
                            extracted_info.append(f"\n[From {url}]:\n{extracted}\n")
                            logger.info(f"Extracted info from {url}")
                
                except Exception as e:
                    logger.error(f"Mistral extraction error for {url}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            continue
    
    return "\n".join(extracted_info) if extracted_info else ""

# --------------------------------------------------
# Citation Verification (Enhanced)
# --------------------------------------------------
def verify_islamic_citations(text: str) -> str:
    """Add warnings if citations are missing or potentially fabricated"""
    quran_patterns = [r"Qur'?an", r'Surah', r'Ayah', r'verse']
    hadith_patterns = [r'Hadith', r'narrated', r'reported by', r'Bukhari', 
                      r'Muslim', r'Tirmidhi', r'Abu Dawud', r"Nasa'?i", r'Ibn Majah']
    
    has_quran_mention = any(re.search(p, text, re.IGNORECASE) for p in quran_patterns)
    has_hadith_mention = any(re.search(p, text, re.IGNORECASE) for p in hadith_patterns)
    
    has_quran_citation = bool(re.search(r'Surah\s+\w+\s*[:\-]\s*\d+', text, re.IGNORECASE))
    has_hadith_citation = bool(re.search(r"(Bukhari|Muslim|Tirmidhi|Abu Dawud|Nasa'?i|Ibn Majah)\s+\d+", text, re.IGNORECASE))
    
    warnings = []
    
    if has_quran_mention and not has_quran_citation:
        warnings.append("\n\n**Citation Notice**: Quran mentioned but no specific verse reference provided.")
    
    if has_hadith_mention and not has_hadith_citation:
        warnings.append("\n\n**Citation Notice**: Hadith mentioned but no specific source reference provided.")
    
    return text + "".join(warnings) if warnings else text

# --------------------------------------------------
# Initialize Components
# --------------------------------------------------
try:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    logger.info("✅ Mistral client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Mistral client: {e}")
    raise RuntimeError(f"Failed to initialize Mistral client: {e}")

vector_db = VectorDatabase()

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="Islamic AI Assistant - Mistral Backend",
    description="AI-powered Islamic knowledge assistant with Mistral AI",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Path Configuration - use already-defined path constants
# --------------------------------------------------
STATIC_DIR = _APP_ROOT / "static"
TEMPLATES_DIR = _APP_ROOT / "templates"
UPLOAD_DIR = _PROJECT_ROOT / "uploads" / "profiles"
UPLOADS_BASE_DIR = _PROJECT_ROOT / "uploads"

# Ensure directories exist
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files directory
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount uploads directory
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_BASE_DIR)), name="uploads")

# In-memory profile storage (persists for server session)
user_profiles: Dict[str, Dict[str, Any]] = {}

# --------------------------------------------------
# User Profile Model
# --------------------------------------------------
class UserProfile(BaseModel):
    name: str = Field(default="User", min_length=1, max_length=100)
    image_path: Optional[str] = None

# --------------------------------------------------
# Request/Response Models
# --------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    conversation_id: Optional[str] = None
    use_vector_search: bool = True
    
    @validator('message')
    def validate_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v

class ChatResponse(BaseModel):
    reply: str
    timestamp: str
    conversation_id: str
    status: str = "success"
    sources_used: Optional[List[Dict]] = None
    context_chunks: Optional[int] = None
    web_extraction_performed: bool = False
    messages_in_chat: Optional[int] = None
    processing_time: Optional[float] = None

# --------------------------------------------------
# API Endpoints
# --------------------------------------------------
@app.get("/")
async def serve_home():
    """Serve the main HTML frontend"""
    try:
        html_path = TEMPLATES_DIR / "index.html"
        
        if not html_path.exists():
            logger.error(f"index.html not found at {html_path}")
            raise HTTPException(status_code=404, detail="Frontend file not found")
        
        return FileResponse(str(html_path))
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mistral_api": "connected",
        "mistral_model": MISTRAL_MODEL,
        "vector_db": vector_db.get_stats(),
        "chat_memory": chat_memory.get_stats(),
        "database_folder": DATABASE_FOLDER,
        "database_folder_exists": Path(DATABASE_FOLDER).exists(),
        "rate_limiter": {
            "requests_in_window": len(rate_limiter.requests),
            "max_requests": rate_limiter.max_requests
        },
        "features": {
            "pdf_support": True,
            "txt_support": True,
            "docx_support": DOCX_AVAILABLE,
            "web_scraping": True,
            "web_extraction": True,
            "vector_search": True,
            "chat_memory": True,
            "rate_limiting": True,
            "async_processing": True,
            "tts_greeting": TTS_AVAILABLE
        }
    }

# --------------------------------------------------
# TTS Greeting Endpoint (commented out per user request)
# --------------------------------------------------
# @app.get("/api/greeting/{username}")
# async def generate_greeting(username: str = ""):
#     """Generate TTS greeting audio for user"""
#     if not TTS_AVAILABLE:
#         logger.warning("TTS not available - pyttsx3 not installed")
#         raise HTTPException(status_code=503, detail="TTS service unavailable")
#     
#     try:
#         # Sanitize username
#         clean_name = re.sub(r'[^a-zA-Z\s]', '', username).strip()
#         
#         # Generate greeting text
#         if clean_name:
#             greeting = f"Assalamu alaykum, {clean_name}"
#         else:
#             greeting = "Assalamu alaykum"
#         
#         logger.info(f"Generating TTS greeting: {greeting}")
#         
#         # Create TTS engine
#         engine = pyttsx3.init()
#         
#         # Configure for friendly, natural voice
#         voices = engine.getProperty('voices')
#         preferred_voice = None
#         
#         # Priority: Look for more natural-sounding voices
#         for voice in voices:
#             voice_name = voice.name.lower()
#             # Prefer Zira (female, friendly) or any Microsoft voice that's not David
#             if 'zira' in voice_name:
#                 preferred_voice = voice
#                 break
#             elif 'hazel' in voice_name or 'susan' in voice_name:
#                 preferred_voice = voice
#                 break
#         
#         # Fallback: Use first available voice if no preferred found
#         if preferred_voice:
#             engine.setProperty('voice', preferred_voice.id)
#             logger.info(f"Using voice: {preferred_voice.name}")
#         elif voices:
#             engine.setProperty('voice', voices[0].id)
#             logger.info(f"Using default voice: {voices[0].name}")
#         
#         # Set friendly rate and volume - slower and warmer
#         engine.setProperty('rate', 130)  # Slower for warmth and clarity
#         engine.setProperty('volume', 1.0)
#         
#         # Generate to temp file
#         temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
#         temp_path = temp_file.name
#         temp_file.close()
#         
#         engine.save_to_file(greeting, temp_path)
#         engine.runAndWait()
#         
#         # Check if file was created
#         if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
#             raise Exception("Failed to generate audio file")
#         
#         logger.info(f"TTS greeting generated successfully: {temp_path}")
#         
#         # Return audio file
#         return FileResponse(
#             temp_path, 
#             media_type="audio/wav",
#             filename="greeting.wav",
#             headers={"Cache-Control": "no-cache"}
#         )
#         
#     except Exception as e:
#         logger.error(f"TTS greeting error: {e}\n{traceback.format_exc()}")
#         raise HTTPException(status_code=500, detail=f"Could not generate greeting: {str(e)}")

# --------------------------------------------------
# Profile Management Endpoints
# --------------------------------------------------
@app.get("/api/profile/{user_id}")
async def get_profile(user_id: str):
    """Get user profile"""
    try:
        profile = user_profiles.get(user_id, {"name": "User", "image_path": None})
        return {"status": "success", "profile": profile}
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/profile/{user_id}")
async def update_profile(user_id: str, profile: UserProfile):
    """Update user profile (name and/or image path)"""
    try:
        # Get existing profile or create new
        existing = user_profiles.get(user_id, {})
        
        # Update fields
        existing["name"] = profile.name
        if profile.image_path is not None:
            existing["image_path"] = profile.image_path
        
        user_profiles[user_id] = existing
        logger.info(f"Profile updated for user {user_id}: {existing}")
        
        return {"status": "success", "profile": existing}
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/profile/{user_id}/image")
async def upload_profile_image(user_id: str, file: UploadFile = File(...)):
    """Upload profile image for user"""
    try:
        # Validate file type
        allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
            )
        
        # Validate file size (max 5MB)
        content = await file.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB")
        
        # Generate unique filename
        file_ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
        if file_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            file_ext = '.jpg'
        
        filename = f"{user_id}_{uuid.uuid4().hex[:8]}{file_ext}"
        file_path = UPLOAD_DIR / filename
        
        # Delete old profile image if exists
        existing_profile = user_profiles.get(user_id, {})
        if existing_profile.get("image_path"):
            old_path = Path("." + existing_profile["image_path"])
            if old_path.exists():
                try:
                    old_path.unlink()
                except:
                    pass
        
        # Save new file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Update profile with image path
        image_url = f"/uploads/profiles/{filename}"
        if user_id not in user_profiles:
            user_profiles[user_id] = {"name": "User"}
        user_profiles[user_id]["image_path"] = image_url
        
        logger.info(f"Profile image uploaded for user {user_id}: {image_url}")
        
        return {
            "status": "success", 
            "image_path": image_url,
            "profile": user_profiles[user_id]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading profile image: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rescan-database")
async def rescan_database():
    """Manually trigger DataBase folder rescan"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, scan_database_folder)
        return result
    except Exception as e:
        logger.error(f"Error rescanning database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@retry_with_backoff(max_retries=3, base_delay=2)
async def call_mistral_api(messages: List[Dict]) -> str:
    """Call Mistral API with retry logic and rate limiting"""
    
    # Check rate limit
    if not rate_limiter.is_allowed():
        wait_time = rate_limiter.get_wait_time()
        logger.warning(f"Rate limit reached. Waiting {wait_time:.2f}s")
        await asyncio.sleep(wait_time)
    
    try:
        logger.info("Calling Mistral API...")
        
        response = await mistral_client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
            **GENERATION_CONFIG
        )
        
        if response and response.choices:
            return response.choices[0].message.content
        else:
            raise Exception("Empty response from Mistral API")
            
    except Exception as e:
        logger.error(f"Mistral API error: {e}")
        
        # Handle specific error types
        error_str = str(e).lower()
        if '429' in error_str or 'quota' in error_str or 'rate limit' in error_str:
            raise Exception("API quota exceeded. Please try again in a moment.")
        elif 'timeout' in error_str:
            raise Exception("Request timed out. Please try again.")
        else:
            raise Exception(f"API error: {str(e)}")
            
@app.post("/chat-stream")
async def chat_stream_endpoint(req: ChatRequest):
    """Stream chat responses with Server-Sent Events"""
    start_time = time.time()
    user_message = req.message
    conversation_id = req.conversation_id or f"chat_{int(time.time() * 1000)}"
    
    async def generate_stream():
        try:
            # Add user message to memory
            chat_memory.add_message(conversation_id, 'user', user_message)
            
            # Detect URLs and extract web content
            detected_urls = detect_urls_in_message(user_message)
            web_content = ""
            
            if detected_urls:
                web_content = await extract_web_content_via_mistral(detected_urls, user_message)
            
            # Vector search
            relevant_context = ""
            sources_used = []
            
            if req.use_vector_search and len(vector_db.texts) > 0:
                search_results = await vector_db.search_async(user_message, k=5, min_score=0.3)
                
                if search_results:
                    relevant_context = "\n\nRELEVANT CONTEXT FROM DATABASE:\n"
                    for i, result in enumerate(search_results, 1):
                        relevant_context += f"\n[Context {i}]\n{result['text']}\n"
                        sources_used.append({
                            "source": result['metadata'].get('source', 'unknown'),
                            "relevance": float(result['similarity_score'])
                        })
            
            # Build system message
            system_message = """You are a warm, knowledgeable Islamic AI Assistant with deep expertise in Hadith, Quranic teachings, and Islamic scholarship.

                YOUR PERSONALITY & COMMUNICATION STYLE:
                - Be warm, welcoming, and genuinely caring - treat every questioner as a fellow seeker of knowledge
                - Use a conversational yet respectful tone - like a wise friend who happens to be a scholar
                - Show enthusiasm for helping others understand Islam's beautiful teachings
                - Begin responses with brief, genuine acknowledgments when appropriate ("That's a wonderful question!", "I'm happy to help explain this")
                - Express concepts in accessible, relatable language while maintaining scholarly accuracy
                - Share the wisdom and beauty behind Islamic teachings, not just the rules
                - Use "we" and "us" to create a sense of shared journey in seeking knowledge
                
                YOUR ROLE:
                - Provide accurate information based exclusively on authentic Islamic sources
                - Maintain respectful, compassionate, and scholarly communication
                - Cite Hadith references with complete source information when available
                - Guide users with wisdom, understanding, and adherence to Islamic principles
                - Recognize the spiritual and practical needs of users seeking Islamic guidance
                - Deliver complete, well-structured responses that fully answer questions without abrupt endings
                - Make complex topics accessible without oversimplifying

                RESPONSE LENGTH AND COMPLETENESS STANDARDS:

                **Response Length Guidelines:**
                - **Minimum Length**: Never provide responses shorter than 3-4 substantial sentences unless the question is extremely simple
                - **Optimal Length**: Aim for 200-600 words for most responses (comprehensive yet digestible)
                - **Maximum Length**: For complex topics, responses may extend to 800-1000 words, but ensure completion
                - **CRITICAL**: Always complete your thoughts and responses fully - never end abruptly mid-sentence or mid-section

                **Ensuring Complete Responses:**
                - Plan your response structure before writing to ensure you can complete it within context limits
                - If a topic requires extensive coverage, prioritize the most essential information
                - Always provide a proper conclusion - never leave responses hanging or incomplete
                - If you must be concise due to space, summarize remaining points rather than cutting off
                - For multi-part questions, address all parts even if briefly
                - End every response with a complete sentence, conclusion, or practical guidance

                **Response Completeness Checklist:**
                ✓ Does my response have a clear beginning, middle, and end?
                ✓ Have I addressed all parts of the user's question?
                ✓ Is my final sentence a complete thought?
                ✓ Did I provide a concluding statement or practical guidance?
                ✓ Would the user feel satisfied that their question was fully answered?

                RESPONSE QUALITY STANDARDS:
                - Provide comprehensive yet concise answers
                - Use clear, accessible language while maintaining scholarly accuracy
                - Structure responses with natural paragraphs for readability
                - Use bullet points or numbered lists only when presenting multiple distinct items, steps, or categories
                - Begin with direct answers before elaborating on details
                - Conclude with practical guidance, application, or a summary statement
                - Ensure proper flow and transitions between sections

                CRITICAL FORMATTING INSTRUCTIONS:

                **QURAN VERSE FORMATTING:**
                Use this exact syntax for Quranic verses - ALWAYS include both Arabic and English when citing verses:

                [Quran:Surah Name Chapter:Verse|Arabic Text|English Translation]

                **MANDATORY REQUIREMENTS:**
                - ALWAYS provide the Arabic text when citing a Quranic verse
                - ALWAYS provide the English translation alongside the Arabic
                - Use authentic translations (Sahih International, Pickthall, or similar)
                - For verse ranges, you may summarize but still provide key Arabic phrases

                **Correct Examples:**
                - `[Quran:Al-Baqarah 2:286|لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا وُسْعَهَا|Allah does not burden a soul beyond that it can bear]`
                - `[Quran:Al-Ikhlas 112:1|قُلْ هُوَ اللَّهُ أَحَدٌ|Say: He is Allah, the One and Only]`
                - `[Quran:Al-Fatiha 1:1-7|بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ...|In the name of Allah, the Most Gracious, the Most Merciful...]`

                **Incorrect Examples (DO NOT USE):**
                - `[Quran:Al-Baqarah 2:286||Allah does not burden a soul...]` ❌ Missing Arabic
                - `[Quran:Al-Baqarah 2:286|لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا وُسْعَهَا|]` ❌ Missing translation
                - `(Quran 2:286)` ❌ Wrong format entirely

                **HADITH FORMATTING:**
                Use this exact syntax for Hadith - ALWAYS include the actual hadith text when available:

                [Hadith:Collection Name and Number|"The actual hadith text in English"|Narrator (optional)]

                **MANDATORY REQUIREMENTS:**
                - ALWAYS provide the hadith text when citing (not just reference number)
                - Use quotation marks around the hadith text
                - Include the narrator when known (e.g., "Narrated by Abu Huraira")
                - If Arabic text is particularly significant, include it before English translation
                - For well-known hadith, provide complete wording

                **Correct Examples:**
                - `[Hadith:Sahih Bukhari 6502|"The best among you are those who have the best manners and character."|Narrated by Abdullah ibn Amr]`
                - `[Hadith:Sahih Muslim 2564|"Taqwa is here," and he pointed to his chest three times.|Narrated by Abu Huraira]`
                - `[Hadith:Sunan Abu Dawud 4031|"He who does not thank people, does not thank Allah."|Narrated by Abu Huraira]`

                **For references without full text (use sparingly):**
                - `This principle is confirmed in [Sahih Bukhari 1234]` - acceptable only when the exact wording isn't critical
                - However, PREFER including the full hadith text whenever possible for user benefit

                **GENERAL FORMATTING RULES:**
                - Do NOT write `[Small Green Box]`, `[Blue/Purple Box]`, or any placeholder text
                - Do NOT use generic references like "as mentioned in hadith" without specifics
                - Use ONLY the exact syntax: `[Quran:...]` and `[Hadith:...]`
                - The system will automatically render these in beautiful styled boxes
                - Empty fields (||) mean missing content - avoid this by always including both text and translation

                **TABLE FORMATTING - CRITICAL RULES:**

                When creating tables for Islamic content (Zakat calculations, prayer times, pillars, etc.):

                1. **Always use proper markdown table syntax:**
                ```
                | Column 1 | Column 2 | Column 3 |
                |----------|----------|----------|
                | Data 1   | Data 2   | Data 3   |
                | Data 4   | Data 5   | Data 6   |
                ```

                2. **Mandatory table requirements:**
                - MUST have a header row with column names
                - MUST have a separator row with dashes (`|----------|`)
                - ALL rows MUST have the same number of columns
                - Use spaces for alignment to improve readability
                - Add blank line before and after table

                3. **Example of correct table:**
                ```
                | Pillar | Arabic Name | Obligation |
                |--------|-------------|------------|
                | 1. Faith | Shahada | Daily affirmation |
                | 2. Prayer | Salah | 5 times daily |
                | 3. Charity | Zakat | Annual 2.5% |
                | 4. Fasting | Sawm | Ramadan month |
                | 5. Pilgrimage | Hajj | Once in lifetime |
                ```

                4. **Zakat calculation table example:**
                ```
                | Category | Nisab (Minimum Threshold) | Zakat Rate | Notes |
                |----------|---------------------------|------------|-------|
                | Gold & Silver | 85g (gold) / 595g (silver) | 2.5% | Includes jewelry above nisab |
                | Cash & Savings | Equivalent to 85g gold | 2.5% | Bank deposits, stocks |
                | Business Goods | Market value above nisab | 2.5% | Inventory, trade goods |
                | Agricultural Produce | 653kg (wheat, dates, etc.) | 5% (irrigated) / 10% (rain-fed) | Based on harvest |
                ```

                **TABLE ERRORS TO AVOID:**
                - ❌ Missing header row
                - ❌ Missing separator row (`|---|---|`)
                - ❌ Inconsistent column counts
                - ❌ No spacing around pipes
                - ❌ Tables without context (always introduce tables with text)

                **HEADING HIERARCHY - ENHANCED:**

                Use proper markdown headings with clear visual hierarchy:
                ```
                # Major Topic (H1)
                Main discussion of the topic...

                ## Main Section (H2)
                Content for this section...

                ### Subsection (H3)
                Supporting details...

                #### Minor Point (H4)
                Fine details...
                ```

                **Heading Rules:**
                - H1 (`#`) - Use for major topics (1-3 per response maximum)
                - H2 (`##`) - Use for main sections under topics
                - H3 (`###`) - Use for subsections and supporting points
                - H4 (`####`) - Use sparingly for minor distinctions
                - Always add blank line before and after headings
                - First heading in response should not have blank line before it
                - ALWAYS bold headings by default (markdown will handle this)

                **LIST FORMATTING - CRITICAL RULES:**

                **When to use lists:**
                - Multiple distinct items (3 or more)
                - Sequential steps or procedures
                - Categories or classifications
                - DO NOT use for just 2 items - use sentences instead

                **Unordered lists (bullet points):**
                ```
                Introduction sentence before the list:

                * **First Item** - Explanation with adequate detail
                * **Second Item** - Full context and information
                * **Third Item** - Complete description

                Concluding sentence after the list.
                ```

                **Ordered lists (numbered):**
                ```
                Introduction sentence describing the sequence:

                1. **First Step** - Clear instruction with explanation
                2. **Second Step** - Detailed guidance
                3. **Third Step** - Complete information

                Summary or transition sentence.
                ```

                **LIST RULES (MANDATORY):**
                - ✅ ALWAYS use `* ` or `- ` for unordered lists (with space after)
                - ✅ ALWAYS use `1. `, `2. `, etc. for ordered lists (with space after)
                - ✅ ALWAYS include introductory sentence before list
                - ✅ ALWAYS include concluding sentence after list
                - ✅ Make each item substantial (not just 1-2 words)
                - ✅ Use consistent formatting within each list
                - ✅ Add blank line before and after lists
                - ✅ NO blank lines between list items
                - ❌ NEVER use lists for just 2 items
                - ❌ NEVER put heading directly before list without intro text

                **Example of WRONG list usage:**
                ```
                The Pillars
                * Shahada
                * Salah
                ```

                **Example of CORRECT list usage:**
                ```
                The Five Pillars of Islam

                The five pillars represent the core practices of Islam:

                * **Shahada (Declaration of Faith)** - The testimony that there is no god but Allah and Muhammad is His messenger
                * **Salah (Prayer)** - The five daily prayers connecting believers with Allah
                * **Zakat (Charity)** - Obligatory annual charity of 2.5% of wealth
                * **Sawm (Fasting)** - Fasting during the month of Ramadan
                * **Hajj (Pilgrimage)** - Pilgrimage to Mecca once in a lifetime for those able

                These five practices form the foundation of Islamic worship and community life.
                ```

                **SPACING STANDARDS (MANDATORY):**

                Proper spacing ensures readability:

                - **Paragraphs**: One blank line between paragraphs
                - **Headings**: One blank line before and after (except first heading)
                - **Lists**: One blank line before and after entire list, NO blank lines between items
                - **Tables**: One blank line before and after
                - **Quran/Hadith boxes**: One blank line before and after
                - **After H1**: Content should start after one blank line
                - **After H2**: Content should start immediately or after one blank line

                **BOLD TEXT USAGE:**

                Use `**text**` for bold formatting:
                - **Key Islamic terms**: **Salah**, **Zakat**, **Taqwa**, **Shahada**
                - **Important concepts**: **This is critical**, **Note carefully**
                - **Warnings**: **Important:** Always verify with scholars
                - **Allah's names**: **Ar-Rahman** (The Most Merciful)
                - **Section emphasis**: Use sparingly - don't bold entire sentences
                - **Headings**: Automatically bolded by markdown

                CRITICAL RULES:

                1. SOURCE INTEGRITY:
                - Answer ONLY based on the Hadith database and verified Islamic knowledge
                - If specific information is not in your database, explicitly state: "I don't have this specific information in my current knowledge base, and I recommend consulting with a qualified scholar."
                - Never fabricate Hadith, references, or Islamic rulings
                - Distinguish clearly between Hadith, Quranic verses, and scholarly opinions
                - Always indicate when you're providing scholarly interpretation versus direct textual evidence
                - When uncertain, say "This matter requires verification" rather than guessing

                2. ACCURACY & VERIFICATION:
                - Verify that questions can be answered from authenticated sources
                - If uncertain about authenticity, acknowledge limitations clearly
                - For complex fiqh (jurisprudence) questions, mention that different schools of thought may have varying interpretations
                - Avoid definitive rulings on matters requiring qualified scholarship (ifta)
                - When presenting a minority opinion, clearly identify it as such
                - Always cite the grading of hadith when known (Sahih, Hasan, Da'if, Mawdu')
                - If a hadith is weak or disputed, mention this clearly

                3. SCOPE MANAGEMENT:
                - For off-topic questions, politely acknowledge and redirect: "While I appreciate your question, my expertise is specifically in Islamic teachings, Quran, and Hadith. I'd be happy to help with questions related to Islamic guidance and scholarship."
                - For questions requiring personal religious rulings (fatwas), advise: "This matter requires a personalized fatwa from a qualified scholar who can consider your specific circumstances."
                - For medical, legal, or professional advice, recommend appropriate experts while offering relevant Islamic perspective if applicable
                - Stay within your role as an educational resource, not a replacement for qualified scholarship or personal spiritual guidance

                4. RESPONSE STRUCTURE - OPTIMAL FLOW:

                **a. Direct Answer (1-3 sentences):**
                - Begin immediately with the core response to the question
                - Don't use lengthy preambles like "Thank you for your question..."
                - Get straight to the point while remaining respectful

                **b. Primary Evidence (Quran/Hadith):**
                - Provide the most relevant Quranic verse or Hadith using proper formatting
                - ALWAYS include both Arabic and English for Quran
                - ALWAYS include the full hadith text, not just reference numbers
                - Add 1 blank line before and after citations

                **c. Explanation & Context:**
                - Explain the meaning and relevance of the evidence
                - Provide historical, linguistic, or scholarly context if needed
                - Address the wisdom (hikmah) behind the teaching

                **d. Supporting Evidence (if needed):**
                - Additional Quran verses or Hadith that support the point
                - Scholarly consensus (ijma') if applicable
                - Different schools of thought if relevant

                **e. Practical Application:**
                - Offer actionable guidance for implementing the teaching
                - Provide real-life examples or scenarios
                - Address common challenges or obstacles

                **f. Conclusion & Limitations:**
                - Provide a brief concluding statement (1-2 sentences)
                - Acknowledge if the topic requires deeper consultation with scholars
                - Suggest areas for further study if appropriate
                - ALWAYS end with a complete thought - never trail off

                **QUALITY CHECKLIST (Verify before responding):**

                ✓ Is my answer based on authenticated sources from my database?
                ✓ Have I used the correct `[Hadith:...]` syntax with full hadith text?
                ✓ Have I used the correct `[Quran:...]` syntax with BOTH Arabic and English?
                ✓ Did I avoid writing placeholder text like "[Box]" or generic references?
                ✓ Did I use appropriate honorifics (ﷺ, رضي الله عنه, etc.)?
                ✓ Is my language clear, accessible, and free of unnecessary jargon?
                ✓ Have I provided context and explanation where needed?
                ✓ Did I avoid overstepping into areas requiring specialized scholarship?
                ✓ Is my response respectful, compassionate, and encouraging?
                ✓ Have I addressed the question directly in the opening sentences?
                ✓ Are all headings properly formatted and hierarchically correct?
                ✓ Are all tables properly formatted with headers and separators?
                ✓ Is there adequate white space and paragraph breaks for readability?
                ✓ Have I completed my response fully without abrupt endings?
                ✓ Does my response have a proper conclusion or closing statement?
                ✓ Is my response length appropriate (200-600 words standard, up to 1000 for complex)?
                ✓ Have I addressed all parts of a multi-part question?
                ✓ Are all lists introduced and concluded with sentences?
                ✓ Did I use lists only when appropriate (3+ items)?
                ✓ Is spacing consistent around all elements?

                **Remember:** Your purpose is to facilitate authentic Islamic knowledge, promote understanding, and guide users toward the straight path (al-sirat al-mustaqim) with wisdom (hikmah), compassion (rahmah), and scholarly integrity.

                You are a bridge to knowledge, not a replacement for qualified scholarship. You provide educational guidance, not personal fatwas.

                **Always prioritize:**
                1. **Accuracy** - Only share verified knowledge
                2. **Completeness** - Finish every response properly
                3. **Clarity** - Make complex topics accessible
                4. **Humility** - Acknowledge limitations and guide to scholars
                5. **Compassion** - Meet people with mercy and understanding

                When in doubt, acknowledge limitations and guide users to qualified scholars. It is better to say "I don't have verified information on this specific point" or "This requires consultation with a qualified scholar" than to provide uncertain or fabricated information.

                May your responses be a means of benefit, guidance, and drawing closer to Allah. Āmīn.
                """
            
            user_message_with_context = f"""{relevant_context if relevant_context else "No specific context available from database."}

{f"WEB CONTENT EXTRACTED:{web_content}" if web_content else ""}

USER QUESTION:
{user_message}

Please provide a helpful, accurate response with proper citations:"""
            
            messages = [
                {"role": "system", "content": system_message},
                *chat_memory.get_formatted_history(conversation_id, limit=5),
                {"role": "user", "content": user_message_with_context}
            ]
            
            # FIXED: Check if streaming is available, otherwise use regular completion
            full_response = ""
            
            try:
                # Try streaming first (if available in your Mistral SDK version)
                if hasattr(mistral_client.chat, 'stream_async'):
                    response = await mistral_client.chat.stream_async(
                        model=MISTRAL_MODEL,
                        messages=messages,
                        **GENERATION_CONFIG
                    )
                    
                    async for chunk in response:
                        if chunk.data.choices:
                            delta = chunk.data.choices[0].delta.content
                            if delta:
                                full_response += delta
                                yield f"data: {json.dumps({'type': 'chunk', 'content': delta})}\n\n"
                                await asyncio.sleep(0.01)
                else:
                    # Fallback: Use regular completion and simulate streaming
                    response = await mistral_client.chat.complete_async(
                        model=MISTRAL_MODEL,
                        messages=messages,
                        **GENERATION_CONFIG
                    )
                    
                    if response and response.choices:
                        full_response = response.choices[0].message.content
                        
                        # Simulate streaming by sending chunks
                        chunk_size = 20  # characters per chunk
                        for i in range(0, len(full_response), chunk_size):
                            chunk = full_response[i:i + chunk_size]
                            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                            await asyncio.sleep(0.05)  # Simulate typing delay
                
            except AttributeError:
                # If streaming not available, use regular completion
                response = await mistral_client.chat.complete_async(
                    model=MISTRAL_MODEL,
                    messages=messages,
                    **GENERATION_CONFIG
                )
                
                if response and response.choices:
                    full_response = response.choices[0].message.content
                    
                    # Simulate streaming
                    chunk_size = 20
                    for i in range(0, len(full_response), chunk_size):
                        chunk = full_response[i:i + chunk_size]
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                        await asyncio.sleep(0.05)
            
            # Add to memory
            chat_memory.add_message(conversation_id, 'assistant', full_response)
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'sources': sources_used})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}\n{traceback.format_exc()}")
            error_msg = "I apologize, but I encountered an error. Please try again."
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )            

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Enhanced chat endpoint with full async support"""
    start_time = time.time()
    user_message = req.message
    conversation_id = req.conversation_id or f"chat_{int(time.time() * 1000)}"
    
    logger.info(f"\n{'='*60}")
    logger.info(f"📨 New message in conversation {conversation_id}")
    logger.info(f"   Message: '{user_message[:100]}...'")
    
    try:
        # Add user message to memory
        chat_memory.add_message(conversation_id, 'user', user_message)
        
        # Detect URLs
        detected_urls = detect_urls_in_message(user_message)
        web_content = ""
        web_extraction_performed = False
        
        if detected_urls:
            logger.info(f"🔗 Detected {len(detected_urls)} URL(s)")
            web_content = await extract_web_content_via_mistral(detected_urls, user_message)
            if web_content:
                web_extraction_performed = True
                logger.info("✅ Web content extracted")
        
        # Vector search
        relevant_context = ""
        sources_used = []
        context_chunks = 0
        
        if req.use_vector_search and len(vector_db.texts) > 0:
            logger.info("🔍 Performing vector search...")
            search_results = await vector_db.search_async(user_message, k=5, min_score=0.3)
            
            if search_results:
                context_chunks = len(search_results)
                relevant_context = "\n\nRELEVANT CONTEXT FROM DATABASE:\n"
                
                for i, result in enumerate(search_results, 1):
                    relevance = result['similarity_score']
                    relevant_context += f"\n[Context {i}] (Relevance: {relevance:.2%})\n"
                    relevant_context += f"Source: {result['metadata'].get('source', 'unknown')}\n"
                    relevant_context += f"{result['text']}\n"
                    relevant_context += "-" * 50 + "\n"
                    
                    sources_used.append({
                        "source": result['metadata'].get('source', 'unknown'),
                        "type": result['metadata'].get('type', 'unknown'),
                        "relevance": float(relevance)
                    })
                
                logger.info(f"✅ Found {len(search_results)} relevant contexts")
        
        # Get conversation history
        history_messages = chat_memory.get_formatted_history(conversation_id, limit=5)
        
        # Build system message
        system_message = """You are an Islamic AI Assistant with deep expertise in Hadith, Quranic teachings, and Islamic scholarship.

    YOUR ROLE:
    - Provide accurate information based exclusively on authentic Islamic sources
    - Maintain respectful, compassionate, and scholarly communication
    - Cite Hadith references with complete source information when available
    - Guide users with wisdom, understanding, and adherence to Islamic principles
    - Recognize the spiritual and practical needs of users seeking Islamic guidance

    RESPONSE QUALITY STANDARDS:
    - Provide comprehensive yet concise answers
    - Use clear, accessible language while maintaining scholarly accuracy
    - Structure responses with natural paragraphs for readability
    - Use bullet points or numbered lists only when presenting multiple distinct items, steps, or categories
    - Begin with direct answers before elaborating on details
    - Conclude with practical guidance or application where relevant

    HADITH CITATION FORMAT:
    - Standard format: [Sahih Bukhari 1234] or [Sahih Muslim, Book 5, Hadith 2345]
    - Include authenticity grading when known (Sahih, Hasan, Da'if)
    - Provide contextual information about the Hadith when it enhances understanding
    - Example: "The Prophet ﷺ said: '...' [Sahih Bukhari 6502 - Book of Manners]"

    CRITICAL RULES:
    1. SOURCE INTEGRITY:
    - Answer ONLY based on the Hadith database and verified Islamic knowledge provided to you
    - If specific information is not in your database, explicitly state: "I don't have this specific information in my current knowledge base"
    - Never fabricate Hadith, references, or Islamic rulings
    - Distinguish clearly between Hadith, Quranic verses, and scholarly opinions

    2. ACCURACY & VERIFICATION:
    - Verify that questions can be answered from your authenticated sources
    - If uncertain about authenticity, acknowledge limitations
    - For complex fiqh (jurisprudence) questions, mention that different schools of thought may have varying interpretations
    - Avoid definitive rulings on matters requiring qualified scholarship (ifta)

    3. SCOPE MANAGEMENT:
    - For off-topic questions, politely acknowledge and redirect: "While I appreciate your question, my expertise is in Islamic teachings and Hadith. I'd be happy to help with questions related to Islamic guidance."
    - For questions requiring personal religious rulings, advise consulting qualified local scholars
    - For medical, legal, or professional advice, recommend appropriate experts while offering relevant Islamic perspective if applicable

    4. CONTEXTUAL AWARENESS:
    - Consider the context and spirit of Hadith, not just literal text
    - Provide background information when Hadith require historical or cultural context
    - Address potential misunderstandings with gentleness and clarity
    - Recognize when questions stem from genuine seeking versus argumentation

    5. SENSITIVITY & RESPECT:
    - Show respect for all Muslims regardless of their level of knowledge
    - Use appropriate honorifics: Prophet Muhammad ﷺ (peace be upon him), Sahaba (may Allah be pleased with them)
    - Be sensitive to diverse Islamic traditions and schools of thought
    - Avoid sectarian bias; present mainstream scholarly consensus when it exists

    6. RESPONSE STRUCTURE:
    - Direct Answer: Begin with the core response
    - Evidence: Provide relevant Hadith or Quranic support
    - Context: Explain background or nuances if needed
    - Application: Offer practical guidance when appropriate
    - Limitations: Acknowledge if the topic requires deeper scholarship

    7. WHAT TO AVOID:
    - Lengthy preambles or unnecessary apologies
    - Over-formatting with excessive bold text or lists
    - Providing answers outside your authenticated database
    - Making definitive rulings on contentious scholarly matters
    - Using overly complex Arabic terminology without explanation

    ENGAGEMENT PRINCIPLES:
    - Meet users where they are in their Islamic journey
    - Encourage seeking knowledge and understanding
    - Promote the beauty and wisdom of Islamic teachings
    - Foster positive character development and spiritual growth
    - Guide toward authentic scholarship and reliable resources when needed

    Remember: Your purpose is to facilitate authentic Islamic knowledge, promote understanding, and guide users toward the straight path with wisdom and compassion."""

        # Build user message with context
        user_message_with_context = f"""{relevant_context if relevant_context else "No specific context available from database."}

    {f"WEB CONTENT EXTRACTED:{web_content}" if web_content else ""}

    USER QUESTION:
    {user_message}

    Please provide a helpful, accurate response with proper citations:"""

        # Build messages for Mistral API
        messages = [
            {"role": "system", "content": system_message}
        ]
        
        # Add conversation history
        messages.extend(history_messages)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message_with_context})
        
        # Call Mistral API
        full_response = await call_mistral_api(messages)
        
        # Verify citations
        full_response = verify_islamic_citations(full_response)
        
        # Add assistant response to memory
        chat_memory.add_message(
            conversation_id, 
            'assistant', 
            full_response,
            metadata={
                'sources_used': len(sources_used),
                'web_extraction': web_extraction_performed
            }
        )
        
        messages_in_chat = len(chat_memory.get_history(conversation_id))
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Response generated in {processing_time:.2f}s")
        logger.info(f"   Response length: {len(full_response)} characters")
        logger.info(f"   Messages in chat: {messages_in_chat}/{MAX_CHAT_MESSAGES}")
        logger.info(f"{'='*60}\n")
        
        return ChatResponse(
            reply=full_response,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id,
            status="success",
            sources_used=sources_used if sources_used else None,
            context_chunks=context_chunks if context_chunks > 0 else None,
            web_extraction_performed=web_extraction_performed,
            messages_in_chat=messages_in_chat,
            processing_time=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}\n{traceback.format_exc()}")
        
        error_message = str(e)
        if "quota" in error_message.lower() or "429" in error_message:
            error_message = "I'm experiencing high demand right now. Please try again in a moment. 🙏"
        elif "timeout" in error_message.lower():
            error_message = "The request took too long. Please try again with a shorter message."
        else:
            error_message = "An error occurred. Please try again."
        
        return ChatResponse(
            reply=error_message,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id,
            status="error",
            processing_time=round(time.time() - start_time, 2)
        )

@app.get("/stats")
async def get_stats():
    """Get comprehensive system statistics"""
    return {
        "vector_db": vector_db.get_stats(),
        "chat_memory": chat_memory.get_stats(),
        "database_folder": DATABASE_FOLDER,
        "database_folder_exists": Path(DATABASE_FOLDER).exists(),
        "supported_formats": ["PDF", "TXT", "DOCX"],
        "docx_support_available": DOCX_AVAILABLE,
        "max_messages_per_chat": MAX_CHAT_MESSAGES,
        "rate_limiter": {
            "max_requests_per_window": rate_limiter.max_requests,
            "window_seconds": rate_limiter.window,
            "current_requests": len(rate_limiter.requests)
        }
    }

@app.delete("/clear-conversation/{conversation_id}")
async def clear_conversation(conversation_id: str):
    """Clear a specific conversation from memory"""
    try:
        chat_memory.clear_conversation(conversation_id)
        return {
            "status": "success",
            "message": f"Conversation {conversation_id} cleared",
            "conversation_id": conversation_id
        }
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations")
async def list_conversations():
    """List all active conversations"""
    return {
        "conversations": [
            {
                "conversation_id": conv_id,
                "metadata": chat_memory.conversation_metadata.get(conv_id, {}),
                "message_count": len(chat_memory.conversations.get(conv_id, []))
            }
            for conv_id in chat_memory.conversations.keys()
        ]
    }

# --------------------------------------------------
# Startup Event
# --------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Actions to perform when the application starts"""
    logger.info("\n" + "="*60)
    logger.info("🚀 Islamic AI Assistant Starting...")
    logger.info(f"   Version: 4.0.0 (Mistral AI)")
    logger.info(f"   Mistral Model: {MISTRAL_MODEL}")
    logger.info(f"   Vector DB entries: {len(vector_db.texts)}")
    logger.info(f"   DataBase folder: {DATABASE_FOLDER}")
    logger.info(f"   Max messages/chat: {MAX_CHAT_MESSAGES}")
    logger.info(f"   Features: Enhanced API, Rate Limiting, Async Processing")
    logger.info("="*60 + "\n")
    
    # FIXED: Run database scan as background task, not blocking
    logger.info("🔍 Scheduling DataBase folder scan...")
    asyncio.create_task(scan_database_in_background())

async def scan_database_in_background():
    """Run database scan without blocking startup"""
    await asyncio.sleep(1)  # Let server start first
    try:
        loop = asyncio.get_event_loop()
        db_scan_result = await loop.run_in_executor(None, scan_database_folder)
        logger.info(f"✅ Database scan complete: {db_scan_result['status']}")
    except Exception as e:
        logger.error(f"❌ Error during database scan: {e}")

# --------------------------------------------------
# Run Server
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    
    def open_browser():
        """Open browser after server starts"""
        time.sleep(4)  # Changed from 2 to 4 seconds
        webbrowser.open("http://127.0.0.1:8000")
        logger.info("🌐 Browser opened at http://127.0.0.1:8000")
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    print("\n" + "="*60)
    print("🕌 Islamic AI Assistant v4.0 - Mistral AI Backend")
    print("🌐 Browser opening at http://127.0.0.1:8000")
    print(f"📂 DataBase folder: {DATABASE_FOLDER}")
    print(f"🤖 Model: {MISTRAL_MODEL}")
    print("✨ Features: Rate Limiting, Async Processing, Better Error Handling")
    print("⌨️  Press CTRL+C to stop")
    print("="*60 + "\n")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000, 
        log_level="info",
        access_log=True
    )
    