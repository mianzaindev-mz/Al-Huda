"""
services/vector_db.py
---------------------
FAISS-backed vector database for semantic search over the Islamic
knowledge corpus. Supports both async and sync add/search operations,
automatic persistence, and runtime statistics.
"""

import asyncio
import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import VECTOR_DB_PATH

logger = logging.getLogger(__name__)

# Embedding dimension for 'all-MiniLM-L6-v2'
_EMBEDDING_DIM = 384


class VectorDatabase:
    """FAISS vector store with async-first add/search and auto-save."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_file: Optional[str] = None,
    ) -> None:
        self.index_file = index_file or VECTOR_DB_PATH
        self.dimension = _EMBEDDING_DIM
        logger.info(f"Initialising VectorDatabase (model: {embedding_model})")

        try:
            self.model = SentenceTransformer(embedding_model)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.texts: List[str] = []
            self.metadata: List[Dict] = []
            self._lock = asyncio.Lock()
            self.load()
            logger.info(f"VectorDatabase ready — {len(self.texts)} existing entries")
        except Exception as exc:
            logger.error(f"Failed to initialise VectorDatabase: {exc}")
            raise

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    async def add_texts_async(
        self, texts: List[str], metadata: Optional[List[Dict]] = None
    ) -> None:
        """Encode and index *texts* asynchronously (non-blocking)."""
        texts = [t for t in texts if t.strip()]
        if not texts:
            logger.warning("add_texts_async: no non-empty texts provided")
            return

        async with self._lock:
            try:
                loop = asyncio.get_running_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: self.model.encode(texts, show_progress_bar=False),
                )
                self.index.add(np.array(embeddings).astype("float32"))
                self.texts.extend(texts)
                self.metadata.extend(
                    metadata[: len(texts)]
                    if metadata
                    else [{"source": "unknown", "type": "text"}] * len(texts)
                )
                logger.info(f"VectorDatabase now has {len(self.texts)} entries")
                await loop.run_in_executor(None, self.save)
            except Exception as exc:
                logger.error(f"add_texts_async failed: {exc}")
                raise

    def add_texts(
        self, texts: List[str], metadata: Optional[List[Dict]] = None
    ) -> None:
        """Synchronous version of add_texts (kept for startup compatibility)."""
        texts = [t for t in texts if t.strip()]
        if not texts:
            return
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            self.index.add(np.array(embeddings).astype("float32"))
            self.texts.extend(texts)
            self.metadata.extend(
                metadata[: len(texts)]
                if metadata
                else [{"source": "unknown", "type": "text"}] * len(texts)
            )
            self.save()
        except Exception as exc:
            logger.error(f"add_texts failed: {exc}")
            raise

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_async(
        self, query: str, k: int = 5, min_score: float = 0.3
    ) -> List[Dict]:
        """Return the top-k results above *min_score* (similarity 0-1)."""
        if not self.texts:
            return []
        try:
            loop = asyncio.get_running_loop()
            q_embed = await loop.run_in_executor(
                None, lambda: self.model.encode([query])
            )
            k_actual = min(k, len(self.texts))
            distances, indices = self.index.search(
                np.array(q_embed).astype("float32"), k_actual
            )
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.texts):
                    score = 1 / (1 + float(dist))
                    if score >= min_score:
                        results.append(
                            {
                                "text": self.texts[idx],
                                "metadata": self.metadata[idx],
                                "distance": float(dist),
                                "similarity_score": score,
                            }
                        )
            logger.info(f"Vector search: {len(results)} results (min_score={min_score})")
            return results
        except Exception as exc:
            logger.error(f"search_async failed: {exc}")
            return []

    def search(self, query: str, k: int = 5, min_score: float = 0.3) -> List[Dict]:
        """Synchronous search (backward compatibility). Default min_score matches async version."""
        if not self.texts:
            return []
        try:
            q_embed = self.model.encode([query])
            k_actual = min(k, len(self.texts))
            distances, indices = self.index.search(
                np.array(q_embed).astype("float32"), k_actual
            )
            return [
                {
                    "text": self.texts[idx],
                    "metadata": self.metadata[idx],
                    "distance": float(dist),
                    "similarity_score": 1 / (1 + float(dist)),
                }
                for idx, dist in zip(indices[0], distances[0])
                if idx < len(self.texts) and 1 / (1 + float(dist)) >= min_score
            ]
        except Exception as exc:
            logger.error(f"search failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Pickle the index, texts, and metadata to disk."""
        try:
            data = {
                "texts": self.texts,
                "metadata": self.metadata,
                "index": faiss.serialize_index(self.index),
            }
            with open(self.index_file, "wb") as fh:
                pickle.dump(data, fh)
            logger.info(f"VectorDatabase saved → {self.index_file}")
        except Exception as exc:
            logger.error(f"save failed: {exc}")

    def load(self) -> None:
        """Restore index, texts, and metadata from disk if the file exists."""
        if not os.path.exists(self.index_file):
            logger.info("No existing vector database found — starting fresh")
            return
        try:
            with open(self.index_file, "rb") as fh:
                data = pickle.load(fh)
            self.texts = data["texts"]
            self.metadata = data["metadata"]
            self.index = faiss.deserialize_index(data["index"])
            logger.info(f"VectorDatabase loaded — {len(self.texts)} entries")
        except Exception as exc:
            logger.error(f"load failed: {exc} — starting with empty database")

    def clear(self) -> None:
        """Remove all entries and save the empty state."""
        logger.info("Clearing VectorDatabase…")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        self.metadata = []
        self.save()
        logger.info("VectorDatabase cleared")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary of sources and types currently indexed."""
        sources: Dict[str, int] = {}
        types: Dict[str, int] = {}
        for meta in self.metadata:
            src = meta.get("source", "unknown")
            typ = meta.get("type", "unknown")
            sources[src] = sources.get(src, 0) + 1
            types[typ] = types.get(typ, 0) + 1
        return {
            "total_entries": len(self.texts),
            "dimension": self.dimension,
            "sources": sources,
            "types": types,
        }


# Singleton — initialised once and shared across the app
vector_db = VectorDatabase()
