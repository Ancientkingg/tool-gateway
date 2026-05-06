"""
Tool Gateway — OpenAPI-compliant proxy for OpenWebUI.

Register this in OpenWebUI:
  Admin Panel → Settings → Integrations → Manage Tool Servers → +
  URL: http://tool-gateway:8000
"""

import os
import json
import httpx
import logging
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("tool-gateway")

SEARXNG_URL    = os.environ.get("SEARXNG_URL",    "http://searxng:8080")
CRAWL4AI_URL   = os.environ.get("CRAWL4AI_URL",   "http://crawl4ai:11235")
PERPLEXICA_URL = os.environ.get("PERPLEXICA_URL", "http://perplexica:3000")
RERANKER_URL   = os.environ.get("RERANKER_URL",   "http://reranker:8090")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

PERPLEXICA_CHAT_MODEL    = os.environ.get("PERPLEXICA_CHAT_MODEL",    "gpt-5-mini")
PERPLEXICA_EMBEDDING_KEY = os.environ.get("PERPLEXICA_EMBEDDING_KEY", "Xenova/all-MiniLM-L6-v2")
PERPLEXICA_TIMEOUT       = float(os.environ.get("PERPLEXICA_TIMEOUT", "180"))

# Synthesize pipeline config
SYNTHESIZE_SEARCH_COUNT  = int(os.environ.get("SYNTHESIZE_SEARCH_COUNT", "15"))  # how many to fetch from SearxNG
SYNTHESIZE_CRAWL_COUNT   = int(os.environ.get("SYNTHESIZE_CRAWL_COUNT",  "5"))   # how many to crawl after reranking
SYNTHESIZE_MAX_LENGTH    = int(os.environ.get("SYNTHESIZE_MAX_LENGTH",   "4000")) # chars per crawled page

log.info(f"Config — SEARXNG_URL={SEARXNG_URL}")
log.info(f"Config — CRAWL4AI_URL={CRAWL4AI_URL}")
log.info(f"Config — PERPLEXICA_URL={PERPLEXICA_URL}")
log.info(f"Config — RERANKER_URL={RERANKER_URL}")
log.info(f"Config — TAVILY_API_KEY={'set' if TAVILY_API_KEY else 'NOT SET'}")
log.info(f"Config — PERPLEXICA_CHAT_MODEL={PERPLEXICA_CHAT_MODEL}")
log.info(f"Config — PERPLEXICA_TIMEOUT={PERPLEXICA_TIMEOUT}")
log.info(f"Config — SYNTHESIZE_SEARCH_COUNT={SYNTHESIZE_SEARCH_COUNT}")
log.info(f"Config — SYNTHESIZE_CRAWL_COUNT={SYNTHESIZE_CRAWL_COUNT}")

_perplexica_chat_provider_id: Optional[str] = None
_perplexica_embedding_provider_id: Optional[str] = None


# ── Provider resolution ───────────────────────────────────────────────────────

async def resolve_perplexica_providers() -> bool:
    global _perplexica_chat_provider_id, _perplexica_embedding_provider_id

    log.debug(f"Fetching Perplexica providers from {PERPLEXICA_URL}/api/providers")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{PERPLEXICA_URL}/api/providers")
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning(f"Could not reach Perplexica providers endpoint: {e}")
        return False

    providers = data.get("providers", [])
    log.debug(f"Perplexica returned {len(providers)} providers")

    for provider in providers:
        provider_name = provider.get("name", "unknown")
        chat_keys  = [m.get("key", "") for m in provider.get("chatModels", [])]
        embed_keys = [m.get("key", "") for m in provider.get("embeddingModels", [])]
        log.debug(f"Provider '{provider_name}' — chat: {chat_keys} — embed: {embed_keys}")

        for model in provider.get("chatModels", []):
            if model.get("key", "").strip() == PERPLEXICA_CHAT_MODEL.strip():
                _perplexica_chat_provider_id = provider["id"]
                log.info(f"Resolved chat provider: {provider_name} (id={provider['id']}) for '{PERPLEXICA_CHAT_MODEL}'")

        for model in provider.get("embeddingModels", []):
            if model.get("key", "").strip() == PERPLEXICA_EMBEDDING_KEY.strip():
                _perplexica_embedding_provider_id = provider["id"]
                log.info(f"Resolved embedding provider: {provider_name} (id={provider['id']}) for '{PERPLEXICA_EMBEDDING_KEY}'")

    if _perplexica_chat_provider_id and _perplexica_embedding_provider_id:
        log.info("All Perplexica provider IDs resolved successfully")
        return True

    if not _perplexica_chat_provider_id:
        all_chat = [m.get("key") for p in providers for m in p.get("chatModels", [])]
        log.warning(f"Chat model '{PERPLEXICA_CHAT_MODEL}' not found. Available: {all_chat}")
    if not _perplexica_embedding_provider_id:
        all_embed = [m.get("key") for p in providers for m in p.get("embeddingModels", [])]
        log.warning(f"Embedding model '{PERPLEXICA_EMBEDDING_KEY}' not found. Available: {all_embed}")

    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Tool Gateway starting up...")
    await resolve_perplexica_providers()
    log.info("Startup complete")
    yield
    log.info("Tool Gateway shutting down")


app = FastAPI(
    title="Research Tool Gateway",
    description=(
        "Unified research tools for OpenWebUI. "
        "Provides web search, deep URL crawling, synthesized research, "
        "and a tiered research pipeline."
    ),
    version="2.0.0",
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

class TavilyResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]

class SynthesizeSource(BaseModel):
    url: str
    title: Optional[str]
    content: str
    rerank_score: Optional[float]

class SynthesizeResponse(BaseModel):
    query: str
    sources: list[SynthesizeSource]
    combined_content: str


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _rerank(query: str, texts: list[str]) -> list[float]:
    """
    Score a list of texts against a query using the reranker.
    Returns a list of scores in the same order as texts.
    Falls back to equal scores if reranker is unavailable.
    """
    log.debug(f"Reranking {len(texts)} texts against query={query!r}")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{RERANKER_URL}/rerank",
                json={
                    "query": query,
                    "texts": texts,
                    "truncate": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # TEI returns list of {"index": N, "score": float}
        scores_by_index = {item["index"]: item["score"] for item in data}
        scores = [scores_by_index.get(i, 0.0) for i in range(len(texts))]
        log.debug(f"Reranker scores: min={min(scores):.3f} max={max(scores):.3f}")
        return scores

    except Exception as e:
        log.warning(f"Reranker unavailable ({e}), falling back to search rank order")
        # Return descending scores so search rank order is preserved
        return [1.0 / (i + 1) for i in range(len(texts))]


async def _crawl_single(client: httpx.AsyncClient, url: str, max_length: int) -> dict:
    """Crawl a single URL via Crawl4AI. Returns dict with url, title, content."""
    log.debug(f"Crawling: {url}")
    try:
        resp = await client.post(
            f"{CRAWL4AI_URL}/crawl",
            json={
                "urls": [url],
                "priority": 10,
                "word_count_threshold": 10,
                "extraction_strategy": "NoExtractionStrategy",
                "chunking_strategy": {"type": "RegexChunking"},
                "bypass_cache": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result", {})
        markdown = result.get("markdown", result.get("cleaned_html", ""))
        title = result.get("metadata", {}).get("title", None)

        if len(markdown) > max_length:
            markdown = markdown[:max_length] + f"\n\n[TRUNCATED — {len(markdown)} chars total]"

        log.debug(f"Crawled {url!r}: {len(markdown)} chars (title={title!r})")
        return {"url": url, "title": title, "content": markdown, "ok": True}

    except Exception as e:
        log.warning(f"Failed to crawl {url!r}: {e}")
        return {"url": url, "title": None, "content": "", "ok": False}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get(
    "/search",
    response_model=SearchResponse,
    summary="Search the web",
    description=(
        "Search the internet using SearxNG. Returns titles, URLs, and snippets. "
        "Use this to discover sources before crawling."
    ),
    operation_id="web_search",
)
async def web_search(
    q: str = Query(..., description="Search query"),
    num_results: int = Query(10, ge=1, le=30, description="Number of results to return"),
    engines: Optional[str] = Query(None, description="Comma-separated engine list"),
):
    log.info(f"web_search: q={q!r} num_results={num_results} engines={engines!r}")

    params = {"q": q, "format": "json", "pageno": 1}
    if num_results:
        params["results"] = num_results
    if engines:
        params["engines"] = engines

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(f"{SEARXNG_URL}/search", params=params)
            log.debug(f"SearxNG status: {resp.status_code}")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.error(f"SearxNG failed: {e}")
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
    log.info(f"web_search OK: {len(results)} results for {q!r}")
    return SearchResponse(query=q, results=results)


@app.get(
    "/crawl",
    response_model=CrawlResponse,
    summary="Extract content from a URL",
    description=(
        "Deeply read a webpage using Crawl4AI + Playwright. "
        "Returns cleaned Markdown. Handles JS-rendered pages and SPAs."
    ),
    operation_id="crawl_url",
)
async def crawl_url(
    url: str = Query(..., description="Full URL to crawl"),
    max_length: int = Query(8000, ge=500, le=50000, description="Max characters of markdown to return"),
):
    log.info(f"crawl_url: url={url!r} max_length={max_length}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        result = await _crawl_single(client, url, max_length)

    if not result["ok"] and not result["content"]:
        raise HTTPException(status_code=502, detail=f"Crawl4AI failed for {url}")

    log.info(f"crawl_url OK: {len(result['content'])} chars from {url!r}")
    return CrawlResponse(
        url=url,
        markdown=result["content"],
        title=result["title"],
    )


@app.post(
    "/tavily",
    response_model=TavilyResponse,
    summary="Tavily AI search",
    description=(
        "Search using Tavily — purpose-built for LLMs. "
        "Returns a pre-synthesized answer with cited sources. "
        "Highest quality results. Requires TAVILY_API_KEY."
    ),
    operation_id="tavily_search",
)
async def tavily_search(
    query: str = Query(..., description="Research question or topic"),
    search_depth: str = Query("advanced", description="Search depth: basic or advanced"),
    max_results: int = Query(10, ge=1, le=20, description="Number of results"),
):
    log.info(f"tavily_search: query={query!r} depth={search_depth}")

    if not TAVILY_API_KEY:
        log.error("Tavily search called but TAVILY_API_KEY is not set")
        raise HTTPException(
            status_code=503,
            detail="TAVILY_API_KEY is not configured on the tool gateway.",
        )

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "search_depth": search_depth,
        "include_answer": True,
        "include_raw_content": False,
        "max_results": max_results,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post("https://api.tavily.com/search", json=payload)
            log.debug(f"Tavily status: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            log.error(f"Tavily request failed: {e}")
            raise HTTPException(status_code=502, detail=f"Tavily error: {e}")

    answer = data.get("answer", "")
    sources = [r.get("url", "") for r in data.get("results", []) if r.get("url")]

    log.info(f"tavily_search OK: answer={len(answer)} chars, {len(sources)} sources")
    return TavilyResponse(query=query, answer=answer, sources=sources)


@app.post(
    "/synthesize",
    response_model=SynthesizeResponse,
    summary="Self-hosted research pipeline",
    description=(
        "Full self-hosted research pipeline: "
        "SearxNG search → reranker scores results → crawl top N pages. "
        "Returns combined page content for the model to synthesize. "
        "Unlimited usage, no external API required."
    ),
    operation_id="synthesize_search",
)
async def synthesize_search(
    query: str = Query(..., description="Research question or topic"),
    num_search: int = Query(SYNTHESIZE_SEARCH_COUNT, ge=5, le=30, description="Results to fetch from SearxNG"),
    num_crawl: int = Query(SYNTHESIZE_CRAWL_COUNT, ge=1, le=10, description="Top results to crawl after reranking"),
    max_length: int = Query(SYNTHESIZE_MAX_LENGTH, ge=500, le=20000, description="Max chars per crawled page"),
):
    log.info(f"synthesize_search: query={query!r} num_search={num_search} num_crawl={num_crawl}")

    # ── 1. Search SearxNG ─────────────────────────────────────────────────────
    log.debug("Step 1: SearxNG search")
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "pageno": 1, "results": num_search},
            )
            resp.raise_for_status()
            search_data = resp.json()
        except httpx.HTTPError as e:
            log.error(f"SearxNG failed in synthesize: {e}")
            raise HTTPException(status_code=502, detail=f"SearxNG error: {e}")

    raw_results = search_data.get("results", [])[:num_search]
    log.info(f"SearxNG returned {len(raw_results)} results")

    if not raw_results:
        raise HTTPException(status_code=404, detail="SearxNG returned no results for this query")

    # ── 2. Rerank results ─────────────────────────────────────────────────────
    log.debug("Step 2: Reranking results")
    snippets = [
        f"{r.get('title', '')} — {r.get('content', '')}"
        for r in raw_results
    ]
    scores = await _rerank(query, snippets)

    # Sort by score descending, take top num_crawl
    ranked = sorted(
        zip(raw_results, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:num_crawl]

    log.info(f"Top {num_crawl} after reranking:")
    for result, score in ranked:
        log.info(f"  score={score:.3f} url={result.get('url', '')!r}")

    # ── 3. Crawl top results ──────────────────────────────────────────────────
    log.debug(f"Step 3: Crawling top {num_crawl} URLs")
    crawl_sources = []
    combined_parts = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for result, score in ranked:
            url = result.get("url", "")
            if not url:
                continue

            crawled = await _crawl_single(client, url, max_length)

            source = SynthesizeSource(
                url=url,
                title=crawled["title"] or result.get("title", ""),
                content=crawled["content"] if crawled["ok"] else result.get("content", ""),
                rerank_score=round(score, 4),
            )
            crawl_sources.append(source)

            if source.content:
                combined_parts.append(
                    f"## Source: {source.title or url}\n"
                    f"URL: {url}\n"
                    f"Rerank score: {score:.3f}\n\n"
                    f"{source.content}"
                )

    combined = "\n\n---\n\n".join(combined_parts)
    log.info(
        f"synthesize_search OK: {len(crawl_sources)} sources, "
        f"{len(combined)} combined chars"
    )

    return SynthesizeResponse(
        query=query,
        sources=crawl_sources,
        combined_content=combined,
    )


@app.post(
    "/perplexica",
    response_model=PerplexicaResponse,
    summary="Perplexica synthesized research",
    description=(
        "Synthesized research via Perplexica. "
        "Last resort — prefer /tavily or /synthesize. "
        "Returns an answer with cited sources."
    ),
    operation_id="perplexica_search",
)
async def perplexica_search(
    query: str = Query(..., description="Research question or topic"),
    focus_mode: str = Query("webSearch", description="Focus mode: webSearch, academicSearch, youtubeSearch, redditSearch"),
    optimization_mode: str = Query("balanced", description="speed or balanced"),
):
    global _perplexica_chat_provider_id, _perplexica_embedding_provider_id

    log.info(f"perplexica_search: query={query!r} focus={focus_mode}")

    if not _perplexica_chat_provider_id or not _perplexica_embedding_provider_id:
        log.info("Provider IDs not resolved, retrying...")
        await resolve_perplexica_providers()

    if not _perplexica_chat_provider_id or not _perplexica_embedding_provider_id:
        log.error("Provider IDs still unresolved after retry")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot resolve Perplexica provider IDs for '{PERPLEXICA_CHAT_MODEL}'.",
        )

    payload = {
        "chatModel":     {"providerId": _perplexica_chat_provider_id,    "key": PERPLEXICA_CHAT_MODEL},
        "embeddingModel":{"providerId": _perplexica_embedding_provider_id,"key": PERPLEXICA_EMBEDDING_KEY},
        "optimizationMode": optimization_mode,
        "sources": ["web"],
        "query": query,
        "history": [],
    }

    log.debug(f"Perplexica payload: {payload}")
    log.debug(f"Opening SSE stream (timeout={PERPLEXICA_TIMEOUT}s)")

    sources: list[str] = []
    message_chunks: list[str] = []
    event_count = 0

    try:
        async with httpx.AsyncClient(timeout=PERPLEXICA_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{PERPLEXICA_URL}/api/search",
                json=payload,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                log.debug(f"Perplexica stream status: {resp.status_code}")
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if not raw:
                        continue
                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError as je:
                        log.warning(f"SSE decode error: {je} raw={raw[:100]!r}")
                        continue

                    etype = event.get("type")
                    event_count += 1
                    log.debug(f"SSE event #{event_count}: type={etype!r}")

                    if etype == "sources":
                        for s in event.get("data", []):
                            url = s.get("metadata", {}).get("url", "")
                            if url:
                                sources.append(url)
                        log.debug(f"Sources so far: {len(sources)}")

                    elif etype == "message":
                        chunk = event.get("data", "")
                        if chunk:
                            message_chunks.append(chunk)

                    elif etype == "messageEnd":
                        log.debug("SSE messageEnd — closing stream")
                        break

                    elif etype == "error":
                        error_data = event.get("data", "unknown error")
                        log.error(f"Perplexica SSE error: {error_data}")
                        raise HTTPException(status_code=502, detail=f"Perplexica error: {error_data}")

    except httpx.TimeoutException:
        log.error(
            f"Perplexica timed out after {PERPLEXICA_TIMEOUT}s — "
            f"{event_count} events, {len(message_chunks)} chunks, {len(sources)} sources"
        )
        if message_chunks:
            log.info("Returning partial response collected before timeout")
        else:
            raise HTTPException(status_code=504, detail=f"Perplexica timed out after {PERPLEXICA_TIMEOUT}s")

    except httpx.HTTPError as e:
        log.error(f"Perplexica HTTP error: {e}")
        raise HTTPException(status_code=502, detail=f"Perplexica error: {e}")

    except Exception as e:
        log.exception(f"Unexpected Perplexica error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    answer = "".join(message_chunks)
    log.info(f"perplexica_search OK: {len(answer)} chars, {len(sources)} sources, {event_count} events")

    return PerplexicaResponse(query=query, answer=answer, sources=sources)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
        "tavily_configured": bool(TAVILY_API_KEY),
        "perplexica_chat_provider_id": _perplexica_chat_provider_id,
        "perplexica_embedding_provider_id": _perplexica_embedding_provider_id,
        "perplexica_chat_model": PERPLEXICA_CHAT_MODEL,
        "perplexica_embedding_key": PERPLEXICA_EMBEDDING_KEY,
        "perplexica_timeout": PERPLEXICA_TIMEOUT,
        "synthesize_config": {
            "search_count": SYNTHESIZE_SEARCH_COUNT,
            "crawl_count": SYNTHESIZE_CRAWL_COUNT,
            "max_length_per_page": SYNTHESIZE_MAX_LENGTH,
        },
        "services": {
            "searxng": SEARXNG_URL,
            "crawl4ai": CRAWL4AI_URL,
            "perplexica": PERPLEXICA_URL,
            "reranker": RERANKER_URL,
        },
    }
