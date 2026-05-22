from __future__ import annotations

import asyncio
import contextvars
import importlib.metadata
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
from olostep import Olostep
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "local")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8011/v1")
OLOSTEP_API_KEY = os.getenv("OLOSTEP_API_KEY")
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "https://search.furyhawk.lol")
LOCAL_TRACE_DIR = os.getenv("LOCAL_TRACE_DIR", ".debug_traces")
MODEL = os.getenv("OPENAI_MODEL", "unsloth/gemma-4-E4B-it-GGUF")

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


def _is_openai_cloud_base_url() -> bool:
    return OPENAI_BASE_URL.rstrip("/") == "https://api.openai.com/v1"


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


class OlostepError(RuntimeError):
    """Raised when an Olostep SDK request fails."""


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
            "OLOSTEP_API_KEY": OLOSTEP_API_KEY,
        }.items()
        if not value
    ]
    try:
        olostep_version = importlib.metadata.version("olostep")
    except importlib.metadata.PackageNotFoundError:
        olostep_version = "not installed"
    try:
        openai_version = importlib.metadata.version("openai-agents")
    except importlib.metadata.PackageNotFoundError:
        openai_version = "not installed"
    return not missing, missing, olostep_version, openai_version


def require_olostep_key() -> str:
    if not OLOSTEP_API_KEY:
        raise OlostepError(
            "OLOSTEP_API_KEY is not set. Add it to .env and restart the app."
        )
    return OLOSTEP_API_KEY


def get_olostep_client() -> Olostep:
    return Olostep(api_key=require_olostep_key())


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
    return text[:max_chars] + "\n... [truncated]"


def compact_exception_message(exc: BaseException, max_chars: int = 400) -> str:
    message = str(exc).strip()
    if message:
        message = f"{type(exc).__name__}: {message}"
    else:
        message = type(exc).__name__
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 15].rstrip() + " ... [truncated]"


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


def _answer_query_impl(query: str) -> str:
    with custom_span("olostep.answer_query", {"query": query}):
        result = get_olostep_client().answers.create(task=query)
        return compact_json(sdk_result_to_dict(result))


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
                "raw": data,
            }
        )


def _search_with_scrape_impl(query: str, limit: int = 5) -> str:
    scrape_options = {"formats": ["markdown"], "timeout": 25}
    with custom_span(
        "olostep.search_with_scrape",
        {"query": query, "limit": limit, "scrape_options": scrape_options},
    ):
        search = get_olostep_client().searches.create(
            query=query,
            limit=limit,
            scrape_options=scrape_options,
        )
        data = sdk_result_to_dict(search)
        return compact_json(
            {
                "query": query,
                "results": normalize_search_links(data.get("links", []), limit=limit),
                "raw": data,
            },
            max_chars=6000,
        )


def _scrape_url_impl(url: str) -> str:
    with custom_span("olostep.scrape_url", {"url": url, "formats": ["markdown"]}):
        scrape = get_olostep_client().scrapes.create(url=url, formats=["markdown"])
        return compact_json(
            {"url": url, "scrape": sdk_result_to_dict(scrape)}, max_chars=10000
        )


@function_tool
async def answer_query(query: str) -> str:
    """Answer a natural-language research query using Olostep Answer API."""
    await emit_progress("Calling Olostep Answer API.")
    try:
        result = await asyncio.to_thread(_answer_query_impl, query)
    except Exception as exc:
        raise OlostepError(
            f"Olostep Answer API failed: {compact_exception_message(exc)}"
        ) from exc
    await emit_progress("Olostep Answer API returned evidence.")
    return result


@function_tool
async def search_web(query: str, limit: int = 8) -> str:
    """Search the web using local SearXNG and return normalized results."""
    await emit_progress(f"Searching the web with local SearXNG: {query}")
    try:
        result = await asyncio.to_thread(_search_web_impl, query, limit)
    except Exception as exc:
        raise OlostepError(
            f"Local SearXNG search failed: {compact_exception_message(exc)}"
        ) from exc
    await emit_progress("Local SearXNG search returned results.")
    return result


@function_tool
async def search_with_scrape(query: str, limit: int = 5) -> str:
    """Search the web and scrape each returned link using Olostep Search with Scrape."""
    await emit_progress(f"Running Olostep search with scrape: {query}")
    try:
        result = await asyncio.to_thread(_search_with_scrape_impl, query, limit)
    except Exception as exc:
        raise OlostepError(
            f"Olostep Search with Scrape failed: {compact_exception_message(exc)}"
        ) from exc
    await emit_progress("Search with scrape returned source content.")
    return result


@function_tool
async def scrape_url(url: str) -> str:
    """Scrape one URL with Olostep and return compact page content."""
    await emit_progress(f"Scraping selected source: {url}")
    try:
        result = await asyncio.to_thread(_scrape_url_impl, url)
    except Exception as exc:
        raise OlostepError(
            f"Olostep Scrape API failed: {compact_exception_message(exc)}"
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
{evidence}

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

analyst_tool = analyst_agent.as_tool(
    tool_name="write_markdown_research_report",
    tool_description="Write the final structured Markdown research report from the gathered evidence.",
)

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
        analyst_tool,
    ],
    output_type=MarkdownResearchReport,
)


async def run_research_assistant(
    query: str, progress: ProgressCallback | None = None
) -> tuple[MarkdownResearchReport, str]:
    require_olostep_key()

    token = _progress_callback.set(progress)
    local_trace_token: contextvars.Token[list[dict[str, Any]] | None] | None = None
    trace_id = gen_trace_id()
    if not _is_openai_cloud_base_url():
        local_trace_token = _local_trace_buffer.set([])
        record_local_trace_event("local_trace_started", {"query": query})
    trace_url = openai_trace_url(trace_id)

    try:
        record_local_trace_event(
            "manager_run_started",
            {"trace_id": trace_id, "workflow": "multi_agent_research_assistant_olostep"},
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
        with trace(
            workflow_name="multi_agent_research_assistant_olostep",
            trace_id=trace_id,
            metadata={"query": query, "app": "reflex_research_assistant"},
        ):
            with custom_span("manager.run", {"query": query}):
                try:
                    result = await Runner.run(manager_agent, prompt, max_turns=30)
                except Exception as exc:
                    raise RuntimeError(
                        f"Manager agent failed: {compact_exception_message(exc)}"
                    ) from exc

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
        await emit_progress("Manager run completed. Flushing OpenAI traces.")
        flush_traces()
        await emit_progress("Trace flushed. Rendering Markdown report.")
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
