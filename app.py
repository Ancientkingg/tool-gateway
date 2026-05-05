"""
Tool Gateway — OpenAPI-compliant proxy for OpenWebUI.

Register this in OpenWebUI:
  Settings → Connections → Add OpenAPI Tool Server
  URL: http://tool-gateway:8000
"""

import os
import httpx
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

SEARXNG_URL    = os.environ.get("SEARXNG_URL",    "http://searxng:8080")
CRAWL4AI_URL   = os.environ.get("CRAWL4AI_URL",   "http://crawl4ai:11235")
PERPLEXICA_URL = os.environ.get("PERPLEXICA_URL", "http://perplexica:3001")
RERANKER_URL   = os.environ.get("RERANKER_URL",   "http://reranker:8090")

app = FastAPI(
    title="Research Tool Gateway",
    description=(
        "Unified research tools for OpenWebUI. "
        "Provides web search, deep URL crawling, and synthesized research."
    ),
    version="1.0.0",
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
    focus_mode: str = Query("webSearch", description="Focus mode: webSearch, academicSearch, youtubeSearch, redditSearch"),
    optimization_mode: str = Query("balanced", description="Optimization mode: speed or balanced"),
):
    payload = {
        "chatModel": {
            "providerId": "5d6da317-24d3-4e18-a44b-37f936368309",
            "key": "gpt-5-mini",
        },
        "embeddingModel": {
            "providerId": "69063d28-76b9-4500-a2a6-eda0c42e7480",
            "key": "Xenova/all-MiniLM-L6-v2",
        },
        "optimizationMode": optimization_mode,
        "sources": ["web"],
        "query": query,
        "history": [],
        "stream": False,
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
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
    return {"status": "ok"}
