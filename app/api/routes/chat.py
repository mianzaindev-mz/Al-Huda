"""
api/routes/chat.py
------------------
/chat  — standard (non-streaming) chat endpoint
/chat-stream — Server-Sent Events streaming endpoint

Both endpoints share the same context-building logic:
  1. Detect and scrape any URLs in the message
  2. Run vector search against the Islamic knowledge base
  3. Call Mistral with a full system prompt + chat history
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.models import ChatRequest, ChatResponse
from app.core.config import GENERATION_CONFIG, MAX_CHAT_MESSAGES, MISTRAL_MODEL
from app.services.chat_memory import chat_memory
from app.services.mistral_client import call_mistral_api, mistral_client, verify_islamic_citations
from app.services.vector_db import vector_db
from app.services.web_scraper import detect_urls_in_message, extract_web_content_via_mistral

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a warm, knowledgeable Islamic AI Assistant with deep expertise in \
Hadith, Quranic teachings, and Islamic scholarship.

YOUR PERSONALITY & COMMUNICATION STYLE:
- Be warm, welcoming, and genuinely caring — treat every questioner as a fellow seeker of knowledge
- Use a conversational yet respectful tone — like a wise friend who happens to be a scholar
- Show enthusiasm for helping others understand Islam's beautiful teachings
- Express concepts in accessible, relatable language while maintaining scholarly accuracy
- Share the wisdom and beauty behind Islamic teachings, not just the rules
- Use "we" and "us" to create a sense of shared journey in seeking knowledge

YOUR ROLE:
- Provide accurate information based exclusively on authentic Islamic sources
- Maintain respectful, compassionate, and scholarly communication
- Cite Hadith references with complete source information when available
- Guide users with wisdom, understanding, and adherence to Islamic principles
- Deliver complete, well-structured responses that fully answer questions without abrupt endings

RESPONSE LENGTH GUIDELINES:
- Minimum: 3-4 substantial sentences unless the question is extremely simple
- Optimal: 200-600 words for most responses
- Maximum: up to 1000 words for complex topics — always finish the thought
- CRITICAL: Always complete your responses; never end mid-sentence

FORMATTING RULES:

Quran verses — MANDATORY syntax (always include both Arabic and English):
  [Quran:Surah Name Chapter:Verse|Arabic Text|English Translation]
  Example: [Quran:Al-Baqarah 2:286|لَا يُكَلِّفُ اللَّهُ نَفْسًا إِلَّا وُسْعَهَا|Allah does not burden a soul beyond that it can bear]

Hadith — MANDATORY syntax (always include the actual hadith text):
  [Hadith:Collection Name and Number|"The actual hadith text in English"|Narrator (optional)]
  Example: [Hadith:Sahih Bukhari 6502|"The best among you are those who have the best manners and character."|Narrated by Abdullah ibn Amr]

Tables: Use proper markdown (header row + separator row + data rows). Always introduce a table with a sentence.

Lists: Use * or - for unordered, 1. 2. etc. for ordered. Always include an intro sentence before and a closing sentence after. Only use lists for 3+ items.

Headings: Use # H1, ## H2, ### H3. Add blank lines before and after.

CRITICAL RULES:
1. SOURCE INTEGRITY — Answer ONLY from authenticated Islamic sources. Never fabricate Hadith or rulings.
2. ACCURACY — If uncertain, say "This requires verification" or refer the user to a qualified scholar.
3. SCOPE — For off-topic questions, redirect politely. For personal fatwas, advise consulting a local scholar.
4. COMPLETION — End every response with a complete sentence or concluding statement.

Remember: You are an educational resource, not a replacement for qualified Islamic scholarship. \
May your responses be a means of genuine benefit and guidance. Āmīn."""


# ---------------------------------------------------------------------------
# Shared context builder
# ---------------------------------------------------------------------------

async def _build_context(req: ChatRequest, conversation_id: str):
    """Fetch vector search results and optional web content for a message."""
    user_message = req.message

    # 1. Web content from any URLs in the message
    detected_urls = detect_urls_in_message(user_message)
    web_content = ""
    web_extraction_performed = False
    if detected_urls:
        logger.info(f"Detected {len(detected_urls)} URL(s) in message")
        web_content = await extract_web_content_via_mistral(detected_urls, user_message)
        if web_content:
            web_extraction_performed = True

    # 2. Semantic vector search
    relevant_context = ""
    sources_used: List[Dict] = []
    context_chunks = 0

    if req.use_vector_search and vector_db.texts:
        search_results = await vector_db.search_async(user_message, k=5, min_score=0.3)
        if search_results:
            context_chunks = len(search_results)
            context_parts = ["\n\nRELEVANT CONTEXT FROM DATABASE:"]
            for i, result in enumerate(search_results, 1):
                relevance = result["similarity_score"]
                context_parts.append(
                    f"\n[Context {i}] (Relevance: {relevance:.1%})\n"
                    f"Source: {result['metadata'].get('source', 'unknown')}\n"
                    f"{result['text']}\n"
                    f"{'-' * 50}"
                )
                sources_used.append({
                    "source": result["metadata"].get("source", "unknown"),
                    "type": result["metadata"].get("type", "unknown"),
                    "relevance": float(relevance),
                })
            relevant_context = "\n".join(context_parts)
            logger.info(f"Vector search: {len(search_results)} contexts retrieved")

    # 3. Compose user content block
    # Build context hierarchy: database → web content → user question
    content_parts = [
        relevant_context if relevant_context else "No specific Islamic knowledge base context available."
    ]
    
    if web_content:
        content_parts.append(f"\nWEB CONTENT EXTRACTED:\n{web_content}")
    
    content_parts.extend([
        f"\nUSER QUESTION:\n{user_message}",
        "\nPlease provide a helpful, accurate response with proper citations:"
    ])
    
    user_content = "".join(content_parts)

    return user_content, sources_used, context_chunks, web_extraction_performed


# ---------------------------------------------------------------------------
# /chat — standard endpoint
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Full chat response (non-streaming). Includes vector search + chat history."""
    start_time = time.time()
    conversation_id = req.conversation_id or f"chat_{int(time.time() * 1000)}"

    logger.info(f"[chat] conversation={conversation_id} | message='{req.message[:80]}…'")

    try:
        chat_memory.add_message(conversation_id, "user", req.message)

        user_content, sources_used, context_chunks, web_extraction_performed = (
            await _build_context(req, conversation_id)
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *chat_memory.get_formatted_history(conversation_id, limit=5),
            {"role": "user", "content": user_content},
        ]

        full_response = await call_mistral_api(messages)
        full_response = verify_islamic_citations(full_response)

        chat_memory.add_message(
            conversation_id,
            "assistant",
            full_response,
            metadata={"sources_used": len(sources_used), "web_extraction": web_extraction_performed},
        )

        messages_in_chat = len(chat_memory.get_history(conversation_id))
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"[chat] Response ready in {processing_time}s ({len(full_response)} chars)")

        return ChatResponse(
            reply=full_response,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id,
            status="success",
            sources_used=sources_used or None,
            context_chunks=context_chunks or None,
            web_extraction_performed=web_extraction_performed,
            messages_in_chat=messages_in_chat,
            processing_time=processing_time,
        )

    except Exception as exc:
        logger.error(f"[chat] Error: {exc}", exc_info=True)
        err = str(exc).lower()
        user_msg = (
            "I'm experiencing high demand right now. Please try again in a moment. 🙏"
            if "quota" in err or "429" in err
            else "The request took too long. Please try with a shorter message."
            if "timeout" in err
            else "An error occurred. Please try again."
        )
        return ChatResponse(
            reply=user_msg,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id,
            status="error",
            processing_time=round(time.time() - start_time, 2),
        )


# ---------------------------------------------------------------------------
# /chat-stream — Server-Sent Events endpoint
# ---------------------------------------------------------------------------

@router.post("/chat-stream")
async def chat_stream_endpoint(req: ChatRequest):
    """Streaming chat using Server-Sent Events. Falls back to simulated streaming
    if the Mistral SDK version does not expose stream_async."""
    conversation_id = req.conversation_id or f"chat_{int(time.time() * 1000)}"

    async def generate_stream():
        try:
            chat_memory.add_message(conversation_id, "user", req.message)

            user_content, sources_used, _, _ = await _build_context(req, conversation_id)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *chat_memory.get_formatted_history(conversation_id, limit=5),
                {"role": "user", "content": user_content},
            ]

            full_response = ""

            # Try native streaming; fall back to simulated chunking
            if hasattr(mistral_client.chat, "stream_async"):
                async for chunk in await mistral_client.chat.stream_async(
                    model=MISTRAL_MODEL, messages=messages, **GENERATION_CONFIG
                ):
                    if chunk.data.choices:
                        delta = chunk.data.choices[0].delta.content
                        if delta:
                            full_response += delta
                            yield f"data: {json.dumps({'type': 'chunk', 'content': delta})}\n\n"
                            await asyncio.sleep(0.01)
            else:
                # Fallback: get full response then stream it in appropriate chunks
                # Scale chunk size and delay based on response length for natural flow
                response = await mistral_client.chat.complete_async(
                    model=MISTRAL_MODEL, messages=messages, **GENERATION_CONFIG
                )
                if response and response.choices:
                    full_response = response.choices[0].message.content
                    # Adaptive chunking: larger responses get bigger chunks
                    chunk_size = max(15, min(50, len(full_response) // 20))
                    chunk_delay = 0.02 if len(full_response) > 500 else 0.04
                    
                    for i in range(0, len(full_response), chunk_size):
                        piece = full_response[i : i + chunk_size]
                        yield f"data: {json.dumps({'type': 'chunk', 'content': piece})}\n\n"
                        await asyncio.sleep(chunk_delay)

            chat_memory.add_message(conversation_id, "assistant", full_response)
            yield f"data: {json.dumps({'type': 'complete', 'sources': sources_used})}\n\n"

        except Exception as exc:
            logger.error(f"[chat-stream] Error: {exc}", exc_info=True)
            err_payload = {"type": "error", "message": "I apologize, but I encountered an error. Please try again."}
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
