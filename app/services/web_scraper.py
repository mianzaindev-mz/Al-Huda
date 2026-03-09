"""
services/web_scraper.py
-----------------------
Web content retrieval utilities:
  - Async website scraping with BeautifulSoup
  - URL detection in user messages
  - Mistral-powered content extraction from scraped pages
"""

import logging
import re
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from app.core.config import MISTRAL_MODEL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

async def scrape_website_async(url: str, max_chars: int = 50_000) -> Tuple[str, bool, Optional[str]]:
    """Fetch *url* in a thread pool and return clean plain text.

    Returns (text, success, error_message).
    """
    import asyncio

    try:
        logger.info(f"Scraping: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT),
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Remove non-content elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
        text = "\n".join(lines)

        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Text from {url} truncated to {max_chars} chars")

        logger.info(f"Scraped {len(text)} chars from {url}")
        return text, True, None

    except requests.Timeout:
        err = f"Timeout while scraping {url}"
        logger.error(err)
        return "", False, err
    except Exception as exc:
        err = f"Error scraping {url}: {exc}"
        logger.error(err)
        return "", False, err


# ---------------------------------------------------------------------------
# URL detection
# ---------------------------------------------------------------------------

def detect_urls_in_message(message: str) -> List[str]:
    """Return a deduplicated list of URLs found in *message*."""
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F]{2}))+"
    return list(set(re.findall(pattern, message)))


# ---------------------------------------------------------------------------
# Mistral-powered web content extraction
# ---------------------------------------------------------------------------

async def extract_web_content_via_mistral(
    urls: List[str], user_query: str
) -> str:
    """Scrape each URL and ask Mistral to distill only query-relevant content.

    Processes up to 3 URLs (limits API cost and latency). Returns combined string.
    Empty string if no useful content extracted from any URL.
    """
    from app.services.mistral_client import mistral_client   # Late import avoids circular dep

    if not urls:
        return ""

    # Enforce URL limit
    max_urls = 3
    urls_to_process = urls[:max_urls]
    if len(urls) > max_urls:
        logger.info(f"URL limit enforced: processing {max_urls}/{len(urls)} URLs")

    logger.info(f"Extracting web content from {len(urls_to_process)} URL(s)")
    extracted_parts: List[str] = []

    for url in urls_to_process:
        text, ok, _ = await scrape_website_async(url, max_chars=5_000)
        if not (ok and text):
            continue

        prompt = (
            f"Extract ONLY information relevant to this query from the web content.\n\n"
            f"USER QUERY: {user_query}\n\n"
            f"WEB CONTENT FROM {url}:\n{text[:3000]}\n\n"
            f"Instructions:\n"
            f"- Extract only relevant information\n"
            f"- Be concise and factual\n"
            f"- If not relevant, say 'No relevant information found'\n\n"
            f"RELEVANT INFORMATION:"
        )

        try:
            response = await mistral_client.chat.complete_async(
                model=MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000,
            )
            if response and response.choices:
                extracted = response.choices[0].message.content
                if extracted and "No relevant information found" not in extracted:
                    extracted_parts.append(f"\n[From {url}]:\n{extracted}\n")
                    logger.info(f"Extracted relevant content from {url}")
        except Exception as exc:
            logger.error(f"Mistral extraction error for {url}: {exc}")

    return "\n".join(extracted_parts)
