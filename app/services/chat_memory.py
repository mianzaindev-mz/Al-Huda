"""
services/chat_memory.py
-----------------------
Per-conversation message history with a configurable max-message
cap. Keeps track of timestamps and metadata so the caller can
inspect conversation health or prune stale sessions.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.config import MAX_CHAT_MESSAGES

logger = logging.getLogger(__name__)


class ChatMemoryManager:
    """In-memory chat history manager with per-conversation message limits."""

    def __init__(self, max_messages: int = MAX_CHAT_MESSAGES) -> None:
        self.max_messages = max_messages
        # conversation_id → list of message dicts
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        # conversation_id → metadata dict (created_at, updated_at, message_count)
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}
        logger.info(f"ChatMemoryManager ready (max {max_messages} messages / conversation)")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Append a message to the conversation, trimming the oldest if over limit."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            self.conversation_metadata[conversation_id] = {
                "created_at": datetime.now().isoformat(),
                "message_count": 0,
            }

        message: Dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            message["metadata"] = metadata

        self.conversations[conversation_id].append(message)
        self.conversation_metadata[conversation_id]["message_count"] += 1
        self.conversation_metadata[conversation_id]["updated_at"] = datetime.now().isoformat()

        # Enforce cap — drop oldest message
        if len(self.conversations[conversation_id]) > self.max_messages:
            self.conversations[conversation_id].pop(0)
            logger.debug(f"Oldest message removed from conversation '{conversation_id}'")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_history(
        self, conversation_id: str, limit: Optional[int] = None
    ) -> List[Dict]:
        """Return the full (or limited) history for a conversation."""
        history = self.conversations.get(conversation_id, [])
        return history[-limit:] if limit else history

    def get_formatted_history(
        self, conversation_id: str, limit: int = 5
    ) -> List[Dict]:
        """Return history formatted for the Mistral API (role + content only)."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.get_history(conversation_id)[-limit:]
        ]

    # ------------------------------------------------------------------
    # Manage
    # ------------------------------------------------------------------

    def clear_conversation(self, conversation_id: str) -> None:
        """Delete all messages and metadata for a conversation."""
        self.conversations.pop(conversation_id, None)
        self.conversation_metadata.pop(conversation_id, None)
        logger.info(f"Conversation '{conversation_id}' cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate statistics across all active conversations."""
        active = sum(
            1
            for meta in self.conversation_metadata.values()
            if "updated_at" in meta
            and (
                datetime.now() - datetime.fromisoformat(meta["updated_at"])
            ).total_seconds() < 3600
        )
        return {
            "total_conversations": len(self.conversations),
            "total_messages": sum(len(v) for v in self.conversations.values()),
            "active_conversations": active,
        }


# Singleton shared across the app
chat_memory = ChatMemoryManager()
