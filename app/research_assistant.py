from __future__ import annotations

import asyncio
import contextlib
import contextvars
import importlib.metadata
import html
import json
import os
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import warnings
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from agents import (
    Agent,
    Runner,
    custom_span,
    flush_traces,
    function_tool,
    gen_trace_id,
    trace,
)
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8011/v1")
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "https://search.furyhawk.lol")
LOCAL_TRACE_DIR = os.getenv("LOCAL_TRACE_DIR", ".debug_traces")
MODEL = os.getenv("OPENAI_MODEL", "gemma-4-E4B-it-GGUF")
ENABLE_REMOTE_TRACING = os.getenv("ENABLE_REMOTE_TRACING", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Ensure the OpenAI-compatible client used by the agents SDK points at the local server.
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

warnings.filterwarnings("ignore", message=".*extra field.*SDK model.*")

ProgressCallback = Callable[[str], Awaitable[None]]
_progress_callback: contextvars.ContextVar[ProgressCallback | None] = (
    contextvars.ContextVar(
        "progress_callback",
        default=None,
    )
)
_local_trace_buffer: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("local_trace_buffer", default=None)
)

TRACE_FIELD_MAX_CHARS = 3500


def _is_openai_cloud_base_url() -> bool:
    return OPENAI_BASE_URL.rstrip("/") == "https://api.openai.com/v1"


def should_use_remote_tracing() -> bool:
    if not ENABLE_REMOTE_TRACING:
        return False
    if not _is_openai_cloud_base_url():
        return False
    # Avoid cloud trace upload attempts when a local placeholder key is still configured.
    if not OPENAI_API_KEY or OPENAI_API_KEY == "local":
        return False
    return True


def _safe_trace_name(trace_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", (trace_id or "").strip())
    if not sanitized:
        sanitized = datetime.now().strftime("trace_%Y%m%d_%H%M%S")
    return sanitized


def _local_trace_path(trace_id: str) -> Path:
    base_path = Path(LOCAL_TRACE_DIR)
    safe_name = _safe_trace_name(trace_id)

    # Allow both directory-style and file-style LOCAL_TRACE_DIR values.
    if base_path.suffix.lower() == ".json":
        candidate = base_path
    else:
        candidate = base_path / f"{safe_name}.json"

    # If candidate exists as a directory, place the trace file inside it.
    if candidate.exists() and candidate.is_dir():
        candidate = candidate / f"{safe_name}.json"

    return candidate


def record_local_trace_event(event: str, details: dict[str, Any]) -> None:
    buffer = _local_trace_buffer.get()
    if buffer is None:
        return
    buffer.append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            "details": details,
        }
    )


def flush_local_trace(trace_id: str) -> str:
    buffer = _local_trace_buffer.get()
    if buffer is None:
        return ""

    trace_path = _local_trace_path(trace_id)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trace_id": trace_id,
        "openai_base_url": OPENAI_BASE_URL,
        "model": MODEL,
        "events": buffer,
    }
    try:
        trace_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(trace_path)
    except IsADirectoryError:
        fallback = Path(LOCAL_TRACE_DIR) / f"{_safe_trace_name(trace_id)}.json"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(fallback)


class RetrievalError(RuntimeError):
    """Raised when local retrieval tools fail."""


class Judgment(BaseModel):
    is_good_enough: bool = Field(
        description="Whether the answer is sufficient for the user query, meaning score >= 0.85."
    )
    score: float = Field(ge=0, le=1, description="Quality score from 0 to 1.")
    reason: str = Field(description="Short explanation of the decision.")
    missing_information: list[str] = Field(
        default_factory=list, description="Important gaps to fix."
    )


class MarkdownResearchReport(BaseModel):
    title: str = Field(description="Research report title.")
    executive_summary: str = Field(description="Short answer-first summary.")
    key_findings: list[str] = Field(description="Most important findings.")
    markdown_report: str = Field(
        description="Complete Markdown report with polished headings, clear analysis, reader-friendly structure, and citations."
    )
    citations: list[str] = Field(
        default_factory=list, description="Source URLs used in the report."
    )
    confidence: str = Field(description="Low, medium, or high confidence.")
    method_used: str = Field(description="Retrieval path used by the manager agent.")


async def emit_progress(message: str) -> None:
    callback = _progress_callback.get()
    if callback is not None:
        await callback(message)


def openai_trace_url(trace_id: str) -> str:
    if not _is_openai_cloud_base_url():
        return str(_local_trace_path(trace_id))
    return f"https://platform.openai.com/logs/trace?trace_id={trace_id}"


def environment_status() -> tuple[bool, list[str], str, str]:
    missing = [
        name
        for name, value in {
            "SEARXNG_BASE_URL": SEARXNG_BASE_URL,
        }.items()
        if not value
    ]
    searxng_version = "http endpoint"
    try:
        openai_version = importlib.metadata.version("openai-agents")
    except importlib.metadata.PackageNotFoundError:
        openai_version = "not installed"
    return not missing, missing, searxng_version, openai_version


def sdk_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "__dict__"):
        return {
            key: value for key, value in vars(result).items() if not key.startswith("_")
        }
    return {"value": str(result)}


def compact_json(data: Any, max_chars: int = 8000) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    if len(text) <= max_chars:
        return text

    # Keep the return value as syntactically valid JSON when truncation is needed.
    preview_budget = max(80, max_chars - 120)
    fallback_payload = {
        "truncated": True,
        "original_chars": len(text),
        "preview": compact_text(text, max_chars=preview_budget),
    }
    fallback_text = json.dumps(
        fallback_payload,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    if len(fallback_text) <= max_chars:
        return fallback_text

    # Final minimal JSON fallback for very small max_chars values.
    return json.dumps({"truncated": True}, ensure_ascii=False)


def compact_text(text: Any, max_chars: int = TRACE_FIELD_MAX_CHARS) -> str:
    rendered = str(text)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max_chars - 15].rstrip() + " ... [truncated]"


def compact_exception_message(exc: BaseException, max_chars: int = 400) -> str:
    message = str(exc).strip()
    if message:
        message = f"{type(exc).__name__}: {message}"
    else:
        message = type(exc).__name__
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 15].rstrip() + " ... [truncated]"


def is_invalid_json_behavior_error(exc: BaseException) -> bool:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(type(current).__name__)
        parts.append(str(current))
        next_exc: BaseException | None = None
        if isinstance(current.__cause__, BaseException):
            next_exc = current.__cause__
        elif isinstance(current.__context__, BaseException):
            next_exc = current.__context__
        current = next_exc

    haystack = "\n".join(parts).lower()
    return "invalid json" in haystack and (
        "modelbehaviorerror" in haystack or "invalid json when parsing" in haystack
    )


def current_year_context() -> str:
    return str(datetime.now().year)


def normalize_search_links(
    links: list[dict[str, Any]], limit: int = 8
) -> list[dict[str, Any]]:
    rows = []
    for link in links[:limit]:
        markdown = link.get("markdown_content") or ""
        rows.append(
            {
                "title": link.get("title") or "Untitled",
                "url": link.get("url") or "",
                "description": link.get("description") or "",
                "markdown_chars": len(markdown),
                "markdown_preview": markdown[:1500] if markdown else "",
            }
        )
    return rows


def _text_from_html(raw_html: str) -> str:
    cleaned = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", raw_html, flags=re.IGNORECASE)
    cleaned = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _extract_title_from_html(raw_html: str) -> str:
    title_match = re.search(r"<title>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    if not title_match:
        return "Untitled"
    return re.sub(r"\s+", " ", html.unescape(title_match.group(1))).strip() or "Untitled"


def _fetch_url_content(url: str, timeout: int = 20, max_chars: int = 120000) -> dict[str, Any]:
    request_headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
        "User-Agent": "multi-agent-research-assistant/0.1",
    }

    with urlopen(Request(url, headers=request_headers), timeout=timeout) as response:
        raw_bytes = response.read()
        content_type = response.headers.get("Content-Type", "")
        charset_match = re.search(r"charset=([^;\s]+)", content_type, flags=re.IGNORECASE)
        encoding = (charset_match.group(1).strip('"\'') if charset_match else "utf-8") or "utf-8"

    decoded = raw_bytes.decode(encoding, errors="replace")
    if len(decoded) > max_chars:
        decoded = decoded[:max_chars]

    title = _extract_title_from_html(decoded)
    text_content = _text_from_html(decoded)
    return {
        "url": url,
        "title": title,
        "content_type": content_type,
        "text": text_content,
    }


def _answer_query_impl(query: str) -> str:
    with custom_span("searxng.answer_query", {"query": query}):
        search_payload_text = _search_web_impl(query, limit=6)
        payload = json.loads(search_payload_text)
        results = payload.get("results", [])

        evidence = []
        for item in results[:5]:
            evidence.append(
                {
                    "title": item.get("title") or "Untitled",
                    "url": item.get("url") or "",
                    "snippet": item.get("description") or "",
                    "engine": item.get("engine") or "searxng",
                }
            )

        answer_summary = " ".join(
            f"[{row['title']}] {row['snippet']}" for row in evidence if row.get("snippet")
        )[:1800]

        return compact_json(
            {
                "query": query,
                "provider": "searxng",
                "summary": answer_summary,
                "evidence": evidence,
            },
            max_chars=6000,
        )


def _search_web_impl(query: str, limit: int = 8) -> str:
    request_params = {
        "q": query,
        "format": "json",
        "language": "auto",
        "safesearch": "0",
    }
    request_url = f"{SEARXNG_BASE_URL.rstrip('/')}/search?{urlencode(request_params)}"
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "multi-agent-research-assistant/0.1",
    }

    with custom_span(
        "searxng.search_web",
        {"query": query, "limit": limit, "base_url": SEARXNG_BASE_URL},
    ):
        with urlopen(Request(request_url, headers=request_headers), timeout=20) as response:
            raw_response = response.read().decode("utf-8", errors="replace")

        try:
            data = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            record_local_trace_event(
                "searxng_json_decode_error",
                {
                    "query": query,
                    "error": str(exc),
                    "response_preview": raw_response[:1200],
                },
            )
            raise RuntimeError(
                "SearXNG returned invalid JSON. See local trace for response preview."
            ) from exc

        raw_results = data.get("results") if isinstance(data, dict) else []
        if not isinstance(raw_results, list):
            record_local_trace_event(
                "searxng_unexpected_schema",
                {
                    "query": query,
                    "results_type": type(raw_results).__name__,
                    "payload_preview": compact_json(data, max_chars=1200),
                },
            )
            raw_results = []

        normalized_results = []
        for item in raw_results[:limit]:
            if not isinstance(item, dict):
                continue
            normalized_results.append(
                {
                    "title": item.get("title") or "Untitled",
                    "url": item.get("url") or item.get("link") or "",
                    "description": item.get("content")
                    or item.get("description")
                    or item.get("snippet")
                    or "",
                    "engine": item.get("engine") or "searxng",
                }
            )

        return compact_json(
            {
                "query": query,
                "provider": "searxng",
                "base_url": SEARXNG_BASE_URL,
                "results": normalized_results,
                "raw_summary": {
                    "result_count": len(raw_results),
                    "keys": list(data.keys()) if isinstance(data, dict) else [],
                },
            }
        )


def _search_with_scrape_impl(query: str, limit: int = 5) -> str:
    with custom_span("searxng.search_with_scrape", {"query": query, "limit": limit}):
        data = json.loads(_search_web_impl(query, limit=max(limit, 3)))
        raw_results = data.get("results", [])
        scraped_rows: list[dict[str, Any]] = []

        for item in raw_results[:limit]:
            url = item.get("url") or ""
            if not url:
                continue
            try:
                page = _fetch_url_content(url, timeout=20)
                scraped_rows.append(
                    {
                        "title": item.get("title") or page.get("title") or "Untitled",
                        "url": url,
                        "description": item.get("description") or "",
                        "markdown_chars": len(page.get("text", "")),
                        "markdown_preview": page.get("text", "")[:1500],
                    }
                )
            except Exception as exc:
                scraped_rows.append(
                    {
                        "title": item.get("title") or "Untitled",
                        "url": url,
                        "description": item.get("description") or "",
                        "error": compact_exception_message(exc),
                    }
                )

        return compact_json(
            {
                "query": query,
                "provider": "searxng_plus_direct_fetch",
                "results": scraped_rows,
                "raw_summary": {
                    "result_count": len(raw_results),
                    "keys": list(data.keys()) if isinstance(data, dict) else [],
                },
            },
            max_chars=7000,
        )


def _scrape_url_impl(url: str) -> str:
    with custom_span("url.scrape", {"url": url}):
        page = _fetch_url_content(url, timeout=20)
        return compact_json(
            {
                "url": url,
                "scrape": {
                    "title": page.get("title") or "Untitled",
                    "content_type": page.get("content_type") or "",
                    "markdown_content": page.get("text", "")[:9000],
                },
            },
            max_chars=7000,
        )


@function_tool
async def answer_query(query: str) -> str:
    """Answer a research query using local SearXNG evidence."""
    await emit_progress("Collecting initial evidence from local SearXNG.")
    try:
        result = await asyncio.to_thread(_answer_query_impl, query)
    except Exception as exc:
        raise RetrievalError(
            f"Initial SearXNG retrieval failed: {compact_exception_message(exc)}"
        ) from exc
    await emit_progress("Initial SearXNG evidence collected.")
    return result


@function_tool
async def search_web(query: str, limit: int = 8) -> str:
    """Search the web using local SearXNG and return normalized results."""
    await emit_progress(f"Searching the web with local SearXNG: {query}")
    try:
        result = await asyncio.to_thread(_search_web_impl, query, limit)
    except Exception as exc:
        raise RetrievalError(
            f"Local SearXNG search failed: {compact_exception_message(exc)}"
        ) from exc
    await emit_progress("Local SearXNG search returned results.")
    return result


@function_tool
async def search_with_scrape(query: str, limit: int = 5) -> str:
    """Search the web and scrape each returned link using SearXNG plus direct fetch."""
    await emit_progress(f"Running search with scrape via local retrieval: {query}")
    try:
        result = await asyncio.to_thread(_search_with_scrape_impl, query, limit)
    except Exception as exc:
        raise RetrievalError(
            f"Search with scrape failed: {compact_exception_message(exc)}"
        ) from exc
    await emit_progress("Search with scrape returned source content.")
    return result


@function_tool
async def scrape_url(url: str) -> str:
    """Scrape one URL and return compact page content."""
    await emit_progress(f"Scraping selected source: {url}")
    try:
        result = await asyncio.to_thread(_scrape_url_impl, url)
    except Exception as exc:
        raise RetrievalError(
            f"URL scrape failed: {compact_exception_message(exc)}"
        ) from exc
    await emit_progress("Selected source scrape completed.")
    return result


def _format_missing_information(missing_information: list[str]) -> str:
    if not missing_information:
        return "none"
    return "; ".join(missing_information[:3])


def _first_heading_or_default(markdown_text: str, query: str) -> str:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or f"Research report: {query[:80]}"
    return f"Research report: {query[:80]}"


def _extract_summary(markdown_text: str) -> str:
    paragraph_lines: list[str] = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph_lines:
                break
            continue
        if stripped.startswith("#"):
            continue
        paragraph_lines.append(stripped)
    if paragraph_lines:
        return " ".join(paragraph_lines)[:500]
    return "Summary unavailable from model output."


def _extract_key_findings(markdown_text: str) -> list[str]:
    findings: list[str] = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            findings.append(stripped[2:].strip())
        if len(findings) >= 5:
            break
    return findings


def _extract_citations(markdown_text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)\]]+", markdown_text)
    # Keep order while removing duplicates.
    seen: set[str] = set()
    ordered = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def coerce_markdown_research_report(
    raw_output: Any, query: str
) -> tuple[MarkdownResearchReport, bool]:
    if isinstance(raw_output, MarkdownResearchReport):
        return raw_output, False

    if isinstance(raw_output, dict):
        return MarkdownResearchReport.model_validate(raw_output), False

    text = str(raw_output).strip()
    if not text:
        raise RuntimeError("Model returned an empty report.")

    try:
        return MarkdownResearchReport.model_validate_json(text), False
    except Exception:
        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return MarkdownResearchReport.model_validate(parsed), False
    except Exception:
        pass

    report = MarkdownResearchReport(
        title=_first_heading_or_default(text, query),
        executive_summary=_extract_summary(text),
        key_findings=_extract_key_findings(text),
        markdown_report=text,
        citations=_extract_citations(text),
        confidence="medium",
        method_used="manager_orchestrated_local_model_fallback",
    )
    return report, True


judge_agent = Agent(
    name="Judge agent",
    model=MODEL,
    instructions=(
        "You judge whether the provided answer is good enough for the original research question. "
        "Reward direct, specific, source-backed answers. Reject vague, stale, or unsupported answers. "
        "Be strict: is_good_enough must be true only when score >= 0.85 and the evidence directly answers "
        "the question with concrete source content, topic-specific detail, and appropriate recency. "
        "For current events, product status, policies, pricing, or factual claims that may change, require recent "
        "primary or highly reputable sources. Do not mark evidence sufficient if any critical gap remains. "
        "Calibrate scores this way: 0.85-1.0 means sufficient to stop with strong source support and no critical gaps; "
        "0.75-0.84 means strong but still missing one important source, detail, recency check, or coverage area; "
        "0.50-0.74 means relevant partial evidence that needs more research; 0.25-0.49 means thin, vague, stale, "
        "or weakly related evidence; below 0.25 is only for empty, unusable, or mostly unrelated evidence. "
        "Do not mark evidence sufficient just because it is plausible or directionally correct. "
        "Return only the structured judgment."
    ),
    output_type=Judgment,
)


@function_tool
async def judge_answer_quality(
    original_question: str, evidence: str, stage: str = "current evidence"
) -> str:
    """Judge whether evidence is sufficient for the original question and emit the score."""
    await emit_progress(f"Judge evaluating {stage}.")
    prompt = f"""
Original research question:
{original_question}

Evidence stage:
{stage}

Evidence to judge:
{compact_text(evidence, max_chars=6000)}

Return a structured judgment for whether this evidence is sufficient to answer the original question.
"""
    with custom_span("judge.answer_quality", {"stage": stage}):
        try:
            result = await Runner.run(judge_agent, prompt, max_turns=3)
        except Exception as exc:
            raise RuntimeError(
                f"Judge agent failed: {compact_exception_message(exc)}"
            ) from exc
    judgment = result.final_output
    await emit_progress(
        f"Judge score: {judgment.score:.2f} "
        f"({'sufficient' if judgment.is_good_enough else 'needs more evidence'}). "
        f"{judgment.reason}"
    )
    if judgment.missing_information:
        await emit_progress(
            f"Judge missing information: {_format_missing_information(judgment.missing_information)}"
        )
    return judgment.model_dump_json()


analyst_agent = Agent(
    name="Analyst agent",
    model=MODEL,
    instructions=(
        "You write a proper Markdown research report from the evidence. "
        "Write for a professional reader who wants a clear, polished research brief on any topic. "
        "Adapt the report to the user's question. The markdown_report must be substantial, easy to scan, and use these general sections only: "
        "Executive Summary, Key Findings, Context, Evidence Review, Detailed Analysis, Implications, Source Notes, and References. "
        "If the topic is event-driven, include timeline details inside Context or Detailed Analysis instead of adding a separate Timeline section. "
        "If the topic is comparative, include a compact comparison table inside Detailed Analysis. "
        "Do not include sections titled Limitations, Next Steps, Recommendations, or Action Items. "
        "Avoid bare caveats like 'I relied on...'. Instead, integrate source quality naturally in Source Notes. "
        "Use short paragraphs, bullets where helpful, and citations as Markdown links. Add enough context that a "
        "non-expert reader understands the issue, why it matters, and what evidence supports it. "
        "Do not use emoji, return-arrow symbols, backlink icons, or decorative icons anywhere in the report. "
        "In References, list only plain Markdown bullets or numbered items with the source name and URL. "
        "Return only the structured report."
    ),
    output_type=MarkdownResearchReport,
)

analyst_agent_unstructured = Agent(
    name="Analyst agent",
    model=MODEL,
    instructions=analyst_agent.instructions,
)


@function_tool
async def write_markdown_research_report(input_text: str) -> str:
    """Write the final structured Markdown research report from the gathered evidence."""
    await emit_progress("Analyst writing final research report.")
    with custom_span("analyst.write_report", {}):
        try:
            result = await Runner.run(analyst_agent, input_text, max_turns=4)
            report = result.final_output
        except Exception as exc:
            if not is_invalid_json_behavior_error(exc):
                raise RuntimeError(
                    f"Analyst agent failed: {compact_exception_message(exc)}"
                ) from exc

            record_local_trace_event(
                "analyst_run_retry_unstructured",
                {
                    "reason": compact_exception_message(exc, max_chars=1000),
                },
            )
            await emit_progress(
                "Analyst returned non-JSON output; retrying without strict structured parsing."
            )
            try:
                retry_result = await Runner.run(
                    analyst_agent_unstructured,
                    input_text,
                    max_turns=4,
                )
            except Exception as retry_exc:
                raise RuntimeError(
                    "Analyst agent failed after JSON fallback retry: "
                    f"{compact_exception_message(retry_exc)}"
                ) from retry_exc

            report, used_fallback = coerce_markdown_research_report(
                retry_result.final_output,
                input_text,
            )
            if used_fallback:
                await emit_progress(
                    "Analyst markdown was normalized into the structured report format."
                )
            return report.model_dump_json()

    return report.model_dump_json()

manager_agent = Agent(
    name="Manager research agent",
    model=MODEL,
    instructions=(
        "You are the orchestrator for a multi-agent research assistant. You must manage the workflow, "
        "not answer from your own memory. For current, recent, latest, ongoing, or time-sensitive "
        "topics, if the user query does not mention a year, add the current year "
        f"({current_year_context()}) to the answer_query, search_with_scrape, and search_web query text "
        "so the tools look for recent results. If the user already mentions a year, preserve that year. "
        "Do not treat older sources as sufficient when newer coverage is needed. Follow this policy exactly:\n"
        "1. Always call answer_query first to get a simple initial answer for the user's question.\n"
        "2. Immediately call judge_answer_quality on the original question plus the answer_query result. "
        "If the judge returns is_good_enough=true and score >= 0.85, stop researching and call "
        "write_markdown_research_report with the question, answer result, and judgment.\n"
        "3. If the first judgment is weak, call search_with_scrape for the original question. "
        "Immediately call judge_answer_quality again on the original question plus the answer_query result, "
        "first judgment, and search_with_scrape result. If this second judge returns is_good_enough=true "
        "and score >= 0.85, stop researching and call write_markdown_research_report with all evidence.\n"
        "4. If the second judgment is still weak, do not call the judge again. Run multiple targeted "
        "search_web calls first, using the judge's missing_information to form the searches. Inspect the "
        "search results, choose at least the top 3 relevant source URLs most likely to answer the missing "
        "points, then call scrape_url on each of those top 3 pages. Scrape more than 3 only if clearly needed.\n"
        "5. Call write_markdown_research_report exactly once at the end, using every answer, judgment, "
        "search result, and scraped page. The analyst must produce the final MarkdownResearchReport.\n"
        "6. Return only the final MarkdownResearchReport. Do not return a casual chat answer, tool transcript, or plan."
    ),
    tools=[
        answer_query,
        judge_answer_quality,
        search_with_scrape,
        search_web,
        scrape_url,
        write_markdown_research_report,
    ],
    output_type=MarkdownResearchReport,
)

manager_agent_unstructured = Agent(
    name="Manager research agent",
    model=MODEL,
    instructions=manager_agent.instructions,
    tools=[
        answer_query,
        judge_answer_quality,
        search_with_scrape,
        search_web,
        scrape_url,
        write_markdown_research_report,
    ],
)


async def run_research_assistant(
    query: str, progress: ProgressCallback | None = None
) -> tuple[MarkdownResearchReport, str]:
    token = _progress_callback.set(progress)
    local_trace_token: contextvars.Token[list[dict[str, Any]] | None] | None = None
    trace_id = gen_trace_id()
    use_remote_tracing = should_use_remote_tracing()
    if not use_remote_tracing:
        local_trace_token = _local_trace_buffer.set([])
        record_local_trace_event("local_trace_started", {"query": query})
    trace_url = openai_trace_url(trace_id)

    try:
        record_local_trace_event(
            "manager_run_started",
            {"trace_id": trace_id, "workflow": "multi_agent_research_assistant_local_search"},
        )
        await emit_progress("Starting manager research agent.")
        prompt = f"""
Research question:
{query}

Return a polished, reader-friendly Markdown research report with substantial detail for the user's specific question. Follow the required workflow exactly:
- Use answer_query first for a simple initial answer.
- Use the judge agent immediately after the simple answer to decide whether to stop or continue.
- If the first judge says the answer is not sufficient, run search_with_scrape.
- Use the judge agent immediately after search_with_scrape to decide whether to stop or continue.
- If the second judge still says the evidence is weak, do not judge again. Run multiple targeted search_web calls, choose at least the top 3 relevant source URLs from the search results, and scrape those top 3 pages for context.
- Analyst agent writes the final Markdown report from all answer, judge, search, and scrape evidence. Do not include Limitations or Next Steps sections.
"""
        trace_context = (
            trace(
                workflow_name="multi_agent_research_assistant_local_search",
                trace_id=trace_id,
                metadata={
                    "query": compact_text(query, max_chars=1200),
                    "app": "reflex_research_assistant",
                },
            )
            if use_remote_tracing
            else contextlib.nullcontext()
        )
        with trace_context:
            with custom_span("manager.run", {"query": query}):
                try:
                    result = await Runner.run(manager_agent, prompt, max_turns=30)
                except Exception as exc:
                    if not is_invalid_json_behavior_error(exc):
                        raise RuntimeError(
                            f"Manager agent failed: {compact_exception_message(exc)}"
                        ) from exc

                    record_local_trace_event(
                        "manager_run_retry_unstructured",
                        {
                            "reason": compact_exception_message(exc, max_chars=1000),
                        },
                    )
                    await emit_progress(
                        "Manager returned non-JSON output; retrying without strict structured parsing."
                    )

                    try:
                        result = await Runner.run(
                            manager_agent_unstructured,
                            prompt,
                            max_turns=30,
                        )
                    except Exception as retry_exc:
                        raise RuntimeError(
                            "Manager agent failed after JSON fallback retry: "
                            f"{compact_exception_message(retry_exc)}"
                        ) from retry_exc

        report, used_fallback = coerce_markdown_research_report(result.final_output, query)
        record_local_trace_event(
            "manager_run_completed",
            {
                "used_report_fallback": used_fallback,
                "trace_url": trace_url,
            },
        )
        if used_fallback:
            await emit_progress(
                "Model returned plain markdown; normalized it into a structured report format."
            )
        if use_remote_tracing:
            await emit_progress("Manager run completed. Flushing OpenAI traces.")
            flush_traces()
            await emit_progress("Trace flushed. Rendering Markdown report.")
        else:
            await emit_progress("Manager run completed. Rendering Markdown report.")
        return report, trace_url
    except Exception as exc:
        record_local_trace_event(
            "manager_run_error",
            {
                "error_type": type(exc).__name__,
                "error": compact_exception_message(exc, max_chars=1200),
            },
        )
        raise
    finally:
        if local_trace_token is not None:
            local_trace_file = flush_local_trace(trace_id)
            if local_trace_file:
                await emit_progress(f"Local debug trace written: {local_trace_file}")
            _local_trace_buffer.reset(local_trace_token)
        _progress_callback.reset(token)
