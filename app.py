"""
Tool Gateway — OpenAPI-compliant proxy for OpenWebUI.

Register this in OpenWebUI:
  Admin Panel → Settings → Integrations → Manage Tool Servers → +
  URL: http://tool-gateway:8000
"""

import os
import httpx
import logging
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SEARXNG_URL    = os.environ.get("SEARXNG_URL",    "http://searxng:8080")
CRAWL4AI_URL   = os.environ.get("CRAWL4AI_URL",   "http://crawl4ai:11235")
PERPLEXICA_URL = os.environ.get("PERPLEXICA_URL", "http://perplexica:3000")
RERANKER_URL   = os.environ.get("RERANKER_URL",   "http://reranker:8090")

PERPLEXICA_CHAT_MODEL    = os.environ.get("PERPLEXICA_CHAT_MODEL",    "gpt-5-mini")
PERPLEXICA_EMBEDDING_KEY = os.environ.get("PERPLEXICA_EMBEDDING_KEY", "Xenova/all-MiniLM-L6-v2")

# Resolved at startup — never hardcoded
_perplexica_chat_provider_id: Optional[str] = None
_perplexica_embedding_provider_id: Optional[str] = None


async def resolve_perplexica_providers() -> bool:
    """
    Fetch provider list from Perplexica and find the IDs for the
    configured chat model and embedding model.
    Returns True on success, False on failure.
    """
    global _perplexica_chat_provider_id, _perplexica_embedding_provider_id

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{PERPLEXICA_URL}/api/providers")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning(f"Could not reach Perplexica providers endpoint: {e}")
        return False

    providers = data.get("providers", [])

    for provider in providers:
        # Find chat provider — the one that has our configured model key
        for model in provider.get("chatModels", []):
            if model.get("key", "").strip() == PERPLEXICA_CHAT_MODEL.strip():
                _perplexica_chat_provider_id = provider["id"]
                log.info(
                    f"Resolved chat provider: {provider['name']} "
                    f"(id={provider['id']}) for model {PERPLEXICA_CHAT_MODEL}"
                )

        # Find embedding provider — the one that has our configured embedding key
        for model in provider.get("embeddingModels", []):
            if model.get("key", "").strip() == PERPLEXICA_EMBEDDING_KEY.strip():
                _perplexica_embedding_provider_id = provider["id"]
                log.info(
                    f"Resolved embedding provider: {provider['name']} "
                    f"(id={provider['id']}) for model {PERPLEXICA_EMBEDDING_KEY}"
                )

    if _perplexica_chat_provider_id and _perplexica_embedding_provider_id:
        return True

    if not _perplexica_chat_provider_id:
        log.warning(f"Could not find chat model '{PERPLEXICA_CHAT_MODEL}' in Perplexica providers")
    if not _perplexica_embedding_provider_id:
        log.warning(f"Could not find embedding model '{PERPLEXICA_EMBEDDING_KEY}' in Perplexica providers")

    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Resolve Perplexica provider IDs on startup
    success = await resolve_perplexica_providers()
    if success:
        log.info("Perplexica providers resolved successfully")
    else:
        log.warning(
            "Perplexica providers could not be resolved at startup. "
            "perplexica_search will retry on first call."
        )
    yield


app = FastAPI(
    title="Research Tool Gateway",
    description=(
        "Unified research tools for OpenWebUI. "
        "Provides web search, deep URL crawling, and synthesized research."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Response models ───────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]

class CrawlResponse(BaseModel):
    url: str
    markdown: str
    title: Optional[str] = None

class PerplexicaResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Search the web",
    description=(
        "Search the internet using SearxNG. Returns titles, URLs, and snippets. "
        "Use this to discover sources before crawling. "
        "Call crawl_url on promising results for full content."
    ),
    operation_id="web_search",
)
async def web_search(
    q: str = Query(..., description="Search query"),
    num_results: int = Query(10, ge=1, le=30, description="Number of results to return"),
    engines: Optional[str] = Query(None, description="Comma-separated engine list, e.g. 'google,brave'"),
):
    params = {
        "q": q,
        "format": "json",
        "pageno": 1,
    }
    if num_results:
        params["results"] = num_results
    if engines:
        params["engines"] = engines

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(f"{SEARXNG_URL}/search", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"SearxNG error: {e}")

    data = resp.json()
    results = [
        SearchResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            snippet=r.get("content", ""),
        )
        for r in data.get("results", [])[:num_results]
    ]
    return SearchResponse(query=q, results=results)


@app.get(
    "/crawl",
    response_model=CrawlResponse,
    summary="Extract content from a URL",
    description=(
        "Deeply read a webpage using Crawl4AI + Playwright. "
        "Returns cleaned Markdown suitable for LLM consumption. "
        "Handles JavaScript-rendered pages, SPAs, and paywalled content better than simple fetches. "
        "Use after web_search to extract full evidence from promising URLs."
    ),
    operation_id="crawl_url",
)
async def crawl_url(
    url: str = Query(..., description="Full URL to crawl"),
    max_length: int = Query(8000, ge=500, le=50000, description="Max characters of markdown to return"),
):
    payload = {
        "urls": [url],
        "priority": 10,
        "word_count_threshold": 10,
        "extraction_strategy": "NoExtractionStrategy",
        "chunking_strategy": {"type": "RegexChunking"},
        "bypass_cache": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(f"{CRAWL4AI_URL}/crawl", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Crawl4AI error: {e}")

    result = data.get("result", {})
    markdown = result.get("markdown", result.get("cleaned_html", ""))
    title = result.get("metadata", {}).get("title", None)

    if len(markdown) > max_length:
        markdown = markdown[:max_length] + f"\n\n[TRUNCATED — {len(markdown)} chars total]"

    return CrawlResponse(url=url, markdown=markdown, title=title)


@app.post(
    "/perplexica",
    response_model=PerplexicaResponse,
    summary="Synthesized web research",
    description=(
        "Use Perplexica to perform fast synthesized web research. "
        "Returns a structured answer with cited sources. "
        "Best for exploratory overviews and quick synthesis. "
        "For deeper evidence gathering, follow up with crawl_url on returned sources."
    ),
    operation_id="perplexica_search",
)
async def perplexica_search(
    query: str = Query(..., description="Research question or topic"),
    focus_mode: str = Query(
        "webSearch",
        description="Focus mode: webSearch, academicSearch, youtubeSearch, redditSearch",
    ),
    optimization_mode: str = Query(
        "balanced",
        description="Optimization mode: speed or balanced",
    ),
):
    global _perplexica_chat_provider_id, _perplexica_embedding_provider_id

    # Retry resolution if startup failed
    if not _perplexica_chat_provider_id or not _perplexica_embedding_provider_id:
        log.info("Provider IDs not resolved, retrying...")
        await resolve_perplexica_providers()

    if not _perplexica_chat_provider_id or not _perplexica_embedding_provider_id:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Cannot resolve Perplexica provider IDs. "
                f"Check that '{PERPLEXICA_CHAT_MODEL}' exists in Perplexica's configured models."
            ),
        )

    payload = {
        "chatModel": {
            "providerId": _perplexica_chat_provider_id,
            "key": PERPLEXICA_CHAT_MODEL,
        },
        "embeddingModel": {
            "providerId": _perplexica_embedding_provider_id,
            "key": PERPLEXICA_EMBEDDING_KEY,
        },
        "optimizationMode": optimization_mode,
        "sources": ["web"],
        "query": query,
        "history": [],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        try:
            resp = await client.post(f"{PERPLEXICA_URL}/api/search", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Perplexica error: {e}")

    sources = [s.get("metadata", {}).get("url", "") for s in data.get("sources", [])]
    return PerplexicaResponse(
        query=query,
        answer=data.get("message", ""),
        sources=[s for s in sources if s],
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "perplexica_chat_provider_id": _perplexica_chat_provider_id,
        "perplexica_embedding_provider_id": _perplexica_embedding_provider_id,
    }
