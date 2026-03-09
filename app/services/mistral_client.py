"""
services/mistral_client.py
--------------------------
Initialises the Mistral AI client singleton and exposes:
  - `mistral_client`  — shared Mistral instance
  - `call_mistral_api()` — rate-limited, retried API call helper
  - `verify_islamic_citations()` — citation integrity check on AI output
"""

import logging
import re
from typing import Dict, List

from mistralai import Mistral

from app.core.config import GENERATION_CONFIG, MISTRAL_API_KEY, MISTRAL_MODEL
from app.services.rate_limiter import rate_limiter, retry_with_backoff

logger = logging.getLogger(__name__)

# Compile regex patterns once for performance
_QURAN_MENTION_PATTERN = re.compile(r"Qur'?an|Surah|Ayah|verse", re.IGNORECASE)
_HADITH_MENTION_PATTERN = re.compile(
    r"Hadith|narrated|reported by|Bukhari|Muslim|Tirmidhi|Abu Dawud|Nasa'?i|Ibn Majah",
    re.IGNORECASE,
)
_QURAN_REF_PATTERN = re.compile(r"Surah\s+\w+\s*[:\-]\s*\d+", re.IGNORECASE)
_HADITH_REF_PATTERN = re.compile(
    r"(Bukhari|Muslim|Tirmidhi|Abu Dawud|Nasa'?i|Ibn Majah)\s+\d+",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

try:
    mistral_client = Mistral(api_key=MISTRAL_API_KEY)
    logger.info("Mistral client initialised successfully")
except Exception as exc:
    logger.error(f"Failed to initialise Mistral client: {exc}")
    raise RuntimeError(f"Failed to initialise Mistral client: {exc}")


# ---------------------------------------------------------------------------
# API call helper
# ---------------------------------------------------------------------------

@retry_with_backoff(max_retries=3, base_delay=2)
async def call_mistral_api(messages: List[Dict]) -> str:
    """Call the Mistral chat completion API with rate limiting and retry logic.

    Args:
        messages: List of role/content message dicts.

    Returns:
        The assistant's reply as a plain string.

    Raises:
        Exception: Propagated after all retries are exhausted.
    """
    import asyncio

    # Honour rate limit — wait if needed
    if not rate_limiter.is_allowed():
        wait = rate_limiter.get_wait_time()
        logger.warning(f"Rate limit reached — waiting {wait:.2f}s")
        await asyncio.sleep(wait)

    try:
        logger.info("Calling Mistral API…")
        response = await mistral_client.chat.complete_async(
            model=MISTRAL_MODEL,
            messages=messages,
            **GENERATION_CONFIG,
        )
        if response and response.choices:
            return response.choices[0].message.content
        raise Exception("Empty response from Mistral API")

    except Exception as exc:
        logger.error(f"Mistral API error: {exc}")
        err_str = str(exc).lower()
        if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
            raise Exception("API quota exceeded. Please try again in a moment.")
        if "timeout" in err_str:
            raise Exception("Request timed out. Please try again.")
        raise Exception(f"API error: {exc}")


# ---------------------------------------------------------------------------
# Citation integrity check
# ---------------------------------------------------------------------------

def verify_islamic_citations(text: str) -> str:
    """Append a notice if the AI mentions Quran/Hadith without a proper citation.

    This is a lightweight safety net — not a replacement for scholarly review.
    Uses precompiled regex patterns for better performance.
    """
    has_quran = _QURAN_MENTION_PATTERN.search(text) is not None
    has_hadith = _HADITH_MENTION_PATTERN.search(text) is not None

    # Check for actual references (e.g. "Surah Al-Baqarah 2:286")
    has_quran_ref = _QURAN_REF_PATTERN.search(text) is not None
    has_hadith_ref = _HADITH_REF_PATTERN.search(text) is not None

    warnings: List[str] = []
    if has_quran and not has_quran_ref:
        warnings.append(
            "\n\n**Citation Notice**: A Quranic verse was mentioned but no specific "
            "reference was provided. Please verify with an authentic copy of the Quran."
        )
    if has_hadith and not has_hadith_ref:
        warnings.append(
            "\n\n**Citation Notice**: A Hadith was mentioned but no specific source "
            "reference was provided. Please verify with authenticated collections."
        )

    return text + "".join(warnings) if warnings else text
