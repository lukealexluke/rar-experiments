#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from query_openai_tex_mcp import (
    DEFAULT_ALLOWED_MCP_TOOLS,
    DEFAULT_APPROVAL_LIMIT,
    DEFAULT_BODY_CONTEXT_LINES,
    DEFAULT_LANGFUSE_TRACE_NAME,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MCP_LABEL,
    DEFAULT_MCP_URL,
    DEFAULT_RETRIEVAL_MODE,
    ModelPricing,
    RETRIEVAL_MODES,
    build_langfuse_generation_input,
    build_langfuse_generation_metadata,
    build_langfuse_generation_output,
    build_langfuse_generation_result_metadata,
    build_langfuse_model_parameters,
    build_langfuse_observation_input,
    build_langfuse_observation_metadata,
    build_langfuse_observation_output,
    build_langfuse_trace_metadata,
    build_system_prompt,
    build_user_prompt,
    coalesce_float,
    compute_langfuse_cost_details,
    extract_langfuse_tool_calls,
    flush_langfuse,
    load_environment_from_dotenv,
    parse_nonnegative_int,
    parse_mcp_headers,
    prepare_upload_inputs,
    read_api_key,
    read_optional_float_env,
    record_langfuse_tool_calls,
    safe_serialize_value,
    setup_langfuse,
    validate_input_path,
    write_json_output,
    write_text_output,
)


DEFAULT_API_KEY_ENV = "GOOGLE_API_KEY"
DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_GEMINI_INPUT_COST_ENV = "GEMINI_INPUT_COST_PER_1M"
DEFAULT_GEMINI_CACHED_INPUT_COST_ENV = "GEMINI_CACHED_INPUT_COST_PER_1M"
DEFAULT_GEMINI_OUTPUT_COST_ENV = "GEMINI_OUTPUT_COST_PER_1M"
DEFAULT_GEMINI_MCP_MAX_REMOTE_CALLS = 10
GEMINI_MCP_FINALIZER_TOOL_CALL_LIMIT = 10
GEMINI_MCP_FINALIZER_RESULT_LIMIT = 4
GEMINI_MCP_FINALIZER_TEXT_LIMIT = 700

DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-pro": ModelPricing(
        input_cost_per_1m=2.50,
        cached_input_cost_per_1m=0.25,
        output_cost_per_1m=15.00,
        source=(
            "Gemini Developer API pricing as of 2026-04-18 "
            "(gemini-2.5-pro, >200k prompt-token tier)"
        ),
    ),
    "gemini-2.5-flash": ModelPricing(
        input_cost_per_1m=0.30,
        cached_input_cost_per_1m=0.03,
        output_cost_per_1m=2.50,
        source="Gemini Developer API pricing as of 2026-04-18",
    ),
    "gemini-2.5-flash-lite": ModelPricing(
        input_cost_per_1m=0.10,
        cached_input_cost_per_1m=0.01,
        output_cost_per_1m=0.40,
        source="Gemini Developer API pricing as of 2026-04-18",
    ),
}


@dataclass
class GeminiClientHandle:
    api_key: str
    sync_client: Any

    @property
    def files(self):
        return self.sync_client.files

    def close(self) -> None:
        self.sync_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a .tex file and a bibliography source to the Gemini API, "
            "and let the model use a selected external retrieval mode."
        )
    )
    parser.add_argument("tex_file", type=Path, help="Path to the LaTeX source file.")
    parser.add_argument(
        "bib_file",
        type=Path,
        help="Path to the bibliography source (.txt, .bib, or .bbl).",
    )
    parser.add_argument(
        "--audit-log",
        type=Path,
        help=(
            "Optional statement audit log path. When provided, the uploaded TeX is "
            "reduced to a masked main-body excerpt around the selected logged line, "
            "and the logged bibliography entry is removed from the uploaded "
            "bibliography text."
        ),
    )
    parser.add_argument(
        "--log-entry",
        type=int,
        help=(
            "1-based audit-log entry index to sanitize against. Required when the "
            "selected audit log contains multiple entries."
        ),
    )
    parser.add_argument(
        "--tex-context-lines",
        "--body-context-lines",
        "--context-lines",
        dest="body_context_lines",
        type=parse_nonnegative_int,
        default=DEFAULT_BODY_CONTEXT_LINES,
        metavar="N",
        help=(
            "Number of main TeX body lines before and after the logged citation line to upload. "
            f"Defaults to {DEFAULT_BODY_CONTEXT_LINES}."
        ),
    )
    parser.add_argument(
        "--prompt",
        help=(
            "The question or instruction to send with the uploaded files. "
            "If omitted, the script uses a default paper-analysis prompt."
        ),
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=RETRIEVAL_MODES,
        default=DEFAULT_RETRIEVAL_MODE,
        help=(
            "External retrieval mode: mcp uses TheoremSearch, web-search uses "
            "Gemini Google Search grounding, and none disables external retrieval. "
            f"Defaults to {DEFAULT_RETRIEVAL_MODE}."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Reasoning effort hint. The script maps this onto Gemini thinking "
            f"controls. Defaults to {DEFAULT_REASONING_EFFORT}."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=(
            "Maximum output tokens for the response. "
            f"Defaults to {DEFAULT_MAX_OUTPUT_TOKENS}."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=(
            "Environment variable containing your Gemini API key. "
            f"Defaults to {DEFAULT_API_KEY_ENV}."
        ),
    )
    parser.add_argument(
        "--mcp-url",
        default=DEFAULT_MCP_URL,
        help=f"Remote MCP server URL. Defaults to {DEFAULT_MCP_URL}.",
    )
    parser.add_argument(
        "--mcp-label",
        default=DEFAULT_MCP_LABEL,
        help=f"MCP server label. Defaults to {DEFAULT_MCP_LABEL}.",
    )
    parser.add_argument(
        "--mcp-tool",
        dest="mcp_tools",
        action="append",
        help=(
            "Allowed MCP tool name. Repeat to allow multiple tools. "
            "Defaults to theorem_search."
        ),
    )
    parser.add_argument(
        "--mcp-header",
        dest="mcp_headers",
        action="append",
        metavar="KEY=VALUE",
        help="Optional HTTP header to send to the MCP server. Repeat as needed.",
    )
    parser.add_argument(
        "--disable-langfuse",
        action="store_true",
        help="Disable Langfuse tracing even if Langfuse credentials are configured.",
    )
    parser.add_argument(
        "--langfuse-trace-name",
        default=DEFAULT_LANGFUSE_TRACE_NAME,
        help=(
            "Langfuse trace name when tracing is enabled. "
            f"Defaults to {DEFAULT_LANGFUSE_TRACE_NAME}."
        ),
    )
    parser.add_argument(
        "--langfuse-user-id",
        help="Optional Langfuse user ID for the trace.",
    )
    parser.add_argument(
        "--langfuse-session-id",
        help="Optional Langfuse session ID for the trace.",
    )
    parser.add_argument(
        "--langfuse-tag",
        dest="langfuse_tags",
        action="append",
        help="Optional Langfuse trace tag. Repeat to add multiple tags.",
    )
    parser.add_argument(
        "--input-cost-per-1m",
        type=float,
        help=(
            "Override model input pricing in USD per 1M tokens for Langfuse cost "
            "tracking."
        ),
    )
    parser.add_argument(
        "--cached-input-cost-per-1m",
        type=float,
        help=(
            "Override cached-input pricing in USD per 1M tokens for Langfuse cost "
            "tracking."
        ),
    )
    parser.add_argument(
        "--output-cost-per-1m",
        type=float,
        help=(
            "Override model output pricing in USD per 1M tokens for Langfuse cost "
            "tracking."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the model's text response.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write the final response JSON.",
    )
    parser.add_argument(
        "--delete-uploads",
        action="store_true",
        help="Delete the uploaded Gemini files after the response completes.",
    )
    return parser.parse_args()


def create_upload_temp_dir() -> Path:
    base_dir = (Path.cwd() / ".gemini_upload_tmp").resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = base_dir / f"tex-mcp-upload-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir()
    return temp_dir


def cleanup_upload_temp_dir(temp_dir: Path | None) -> None:
    if temp_dir is None:
        return
    resolved = temp_dir.resolve()
    workspace_root = Path.cwd().resolve()
    try:
        resolved.relative_to(workspace_root)
    except ValueError:
        return
    for path in sorted(resolved.rglob("*"), reverse=True):
        try:
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        except OSError:
            pass
    try:
        resolved.rmdir()
    except OSError:
        pass


def resolve_model_pricing(args: argparse.Namespace) -> ModelPricing | None:
    model_key = args.model.strip().lower()
    default_pricing = DEFAULT_MODEL_PRICING.get(model_key)

    input_cost = coalesce_float(
        args.input_cost_per_1m,
        read_optional_float_env(DEFAULT_GEMINI_INPUT_COST_ENV),
        default_pricing.input_cost_per_1m if default_pricing else None,
    )
    cached_input_cost = coalesce_float(
        args.cached_input_cost_per_1m,
        read_optional_float_env(DEFAULT_GEMINI_CACHED_INPUT_COST_ENV),
        default_pricing.cached_input_cost_per_1m if default_pricing else None,
    )
    output_cost = coalesce_float(
        args.output_cost_per_1m,
        read_optional_float_env(DEFAULT_GEMINI_OUTPUT_COST_ENV),
        default_pricing.output_cost_per_1m if default_pricing else None,
    )

    if input_cost is None or cached_input_cost is None or output_cost is None:
        return None

    if (
        args.input_cost_per_1m is not None
        or args.cached_input_cost_per_1m is not None
        or args.output_cost_per_1m is not None
    ):
        source = "CLI pricing override"
    elif any(
        os.environ.get(env_name, "").strip()
        for env_name in (
            DEFAULT_GEMINI_INPUT_COST_ENV,
            DEFAULT_GEMINI_CACHED_INPUT_COST_ENV,
            DEFAULT_GEMINI_OUTPUT_COST_ENV,
        )
    ):
        source = "environment pricing override"
    elif default_pricing is not None:
        source = default_pricing.source
    else:
        source = "manual pricing override"

    return ModelPricing(
        input_cost_per_1m=input_cost,
        cached_input_cost_per_1m=cached_input_cost,
        output_cost_per_1m=output_cost,
        source=source,
    )


def ensure_runtime_dependencies() -> None:
    try:
        from google.genai import types  # noqa: F401
        import httpx  # noqa: F401
        from mcp import ClientSession  # noqa: F401
        from mcp.client.streamable_http import streamable_http_client  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The Gemini path requires the `google-genai` and `mcp` packages. "
            "Install them with `pip install google-genai mcp`."
        ) from exc


def make_client(api_key: str, *, enable_langfuse: bool):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "The `google-genai` package is not installed. Install it with "
            "`pip install google-genai` and rerun the script."
        ) from exc
    return GeminiClientHandle(
        api_key=api_key,
        sync_client=genai.Client(api_key=api_key),
    )


def upload_file(client, path: Path):
    from google.genai import types

    mime_type = "text/plain" if path.suffix.lower() in {".tex", ".txt", ".bib", ".bbl"} else None
    return client.files.upload(
        file=str(path),
        config=types.UploadFileConfig(
            mime_type=mime_type,
            display_name=path.name,
        ),
    )


def cleanup_uploaded_files(client, uploaded_file_names: list[str]) -> None:
    if client is None:
        return
    for file_name in uploaded_file_names:
        if not file_name:
            continue
        try:
            client.files.delete(name=file_name)
        except Exception:
            pass


def build_thinking_config(model: str, reasoning_effort: str):
    from google.genai import types

    model_key = model.strip().lower()
    note: str | None = None

    if model_key.startswith("gemini-3"):
        level_map = {
            "none": "minimal",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
        }
        level = level_map[reasoning_effort]
        if reasoning_effort == "xhigh":
            note = f"{model} only supports thinking levels up to high; mapping xhigh to high."
        return types.ThinkingConfig(thinking_level=level), note

    if "flash-lite" in model_key:
        budget_map = {
            "none": 0,
            "low": 512,
            "medium": -1,
            "high": 4096,
            "xhigh": 24576,
        }
    elif "flash" in model_key:
        budget_map = {
            "none": 0,
            "low": 1024,
            "medium": -1,
            "high": 8192,
            "xhigh": 24576,
        }
    else:
        budget_map = {
            "none": 128,
            "low": 1024,
            "medium": -1,
            "high": 8192,
            "xhigh": 32768,
        }
        if reasoning_effort == "none":
            note = (
                f"{model} does not support fully disabling thinking; "
                "mapping reasoning-effort=none to thinking_budget=128."
            )

    return types.ThinkingConfig(thinking_budget=budget_map[reasoning_effort]), note


def run_response(
    client: GeminiClientHandle,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    prompt: str,
    tex_upload,
    bib_upload,
    mcp_label: str,
    mcp_url: str,
    requested_tools: list[str],
    mcp_headers: dict[str, str],
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
    mcp_read_timeout: float | None = None,
):
    return asyncio.run(
        _run_response_async(
            client=client,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            prompt=prompt,
            tex_upload=tex_upload,
            bib_upload=bib_upload,
            mcp_label=mcp_label,
            mcp_url=mcp_url,
            requested_tools=requested_tools,
            mcp_headers=mcp_headers,
            retrieval_mode=retrieval_mode,
            mcp_read_timeout=mcp_read_timeout,
        )
    )


async def _run_response_async(
    client,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    prompt: str,
    tex_upload,
    bib_upload,
    mcp_label: str,
    mcp_url: str,
    requested_tools: list[str],
    mcp_headers: dict[str, str],
    retrieval_mode: str,
    mcp_read_timeout: float | None,
):
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Missing Gemini runtime dependency. Install `google-genai`."
        ) from exc

    thinking_config, thinking_note = build_thinking_config(model, reasoning_effort)
    async_client = genai.Client(api_key=client.api_key).aio
    try:
        config_kwargs: dict[str, Any] = {
            "system_instruction": build_system_prompt(retrieval_mode),
            "max_output_tokens": max_output_tokens,
            "thinking_config": thinking_config,
        }
        if retrieval_mode == "mcp":
            try:
                config_kwargs["automatic_function_calling"] = (
                    types.AutomaticFunctionCallingConfig(
                        maximum_remote_calls=DEFAULT_GEMINI_MCP_MAX_REMOTE_CALLS,
                        ignore_call_history=False,
                    )
                )
            except AttributeError:
                pass
        if retrieval_mode == "web-search":
            try:
                config_kwargs["tools"] = [
                    types.Tool(google_search=types.GoogleSearch())
                ]
            except AttributeError as exc:
                raise RuntimeError(
                    "Installed google-genai does not support Google Search grounding. "
                    "Upgrade `google-genai` or use --retrieval-mode mcp."
                ) from exc
            config = types.GenerateContentConfig(**config_kwargs)
            response = await async_client.models.generate_content(
                model=model,
                contents=[prompt, tex_upload, bib_upload],
                config=config,
            )
            return response, to_dict(response), ["google_search"], thinking_note

        if retrieval_mode == "none":
            config = types.GenerateContentConfig(**config_kwargs)
            response = await async_client.models.generate_content(
                model=model,
                contents=[prompt, tex_upload, bib_upload],
                config=config,
            )
            return response, to_dict(response), [], thinking_note

        try:
            import httpx
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise RuntimeError(
                "Missing MCP runtime dependency. Install `mcp` and `httpx`, "
                "or use --retrieval-mode web-search/none."
            ) from exc

        timeout = (
            httpx.Timeout(
                connect=30.0,
                read=mcp_read_timeout if mcp_read_timeout and mcp_read_timeout > 0 else None,
                write=30.0,
                pool=30.0,
            )
            if mcp_read_timeout is not None
            else None
        )
        http_client_kwargs: dict[str, Any] = {
            "headers": mcp_headers or None,
            "follow_redirects": True,
        }
        if timeout is not None:
            http_client_kwargs["timeout"] = timeout

        http_client = httpx.AsyncClient(**http_client_kwargs)
        async with http_client:
            async with streamable_http_client(
                mcp_url,
                http_client=http_client,
                terminate_on_close=False,
            ) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    list_tools_result = await session.list_tools()
                    available_tool_names = [
                        tool.name
                        for tool in getattr(list_tools_result, "tools", [])
                        if getattr(tool, "name", "")
                    ]
                    config_kwargs["tools"] = [session]
                    config = types.GenerateContentConfig(**config_kwargs)
                    response = await async_client.models.generate_content(
                        model=model,
                        contents=[prompt, tex_upload, bib_upload],
                        config=config,
                    )
                    response_data = to_dict(response)
                    if should_finalize_empty_mcp_response(response, response_data):
                        response, response_data = await finalize_empty_mcp_response(
                            async_client=async_client,
                            types=types,
                            model=model,
                            thinking_config=thinking_config,
                            max_output_tokens=max_output_tokens,
                            prompt=prompt,
                            tex_upload=tex_upload,
                            bib_upload=bib_upload,
                            search_response=response,
                            search_response_data=response_data,
                        )
                    return (
                        response,
                        response_data,
                        available_tool_names or requested_tools,
                        thinking_note,
                    )
    finally:
        await async_client.aclose()


def should_finalize_empty_mcp_response(response, response_data: dict[str, Any]) -> bool:
    if extract_output_text(response, response_data):
        return False
    tool_calls = extract_langfuse_tool_calls(response_data)
    if not tool_calls:
        return False
    return any(
        call.get("name") == "theorem_search"
        and call.get("metadata", {}).get("source") == "candidates"
        and not call.get("output")
        for call in tool_calls
    )


async def finalize_empty_mcp_response(
    *,
    async_client,
    types,
    model: str,
    thinking_config,
    max_output_tokens: int,
    prompt: str,
    tex_upload,
    bib_upload,
    search_response,
    search_response_data: dict[str, Any],
):
    tool_calls = extract_langfuse_tool_calls(search_response_data)
    final_prompt = build_mcp_finalizer_prompt(prompt, tool_calls)
    config = types.GenerateContentConfig(
        system_instruction=(
            "External tool use is finished. Do not call tools. Return exactly one "
            "JSON object and no extra commentary."
        ),
        max_output_tokens=max_output_tokens,
        thinking_config=thinking_config,
    )
    final_response = await async_client.models.generate_content(
        model=model,
        contents=[final_prompt, tex_upload, bib_upload],
        config=config,
    )
    final_response_data = to_dict(final_response)
    final_response_data["gemini_mcp_finalized_after_empty_tool_response"] = True
    final_response_data["gemini_mcp_search_response_id"] = read_response_id(
        search_response_data
    )
    final_response_data["gemini_mcp_search_response_status"] = read_response_status(
        search_response_data
    )
    final_response_data["langfuse_tool_calls"] = tool_calls
    final_response_data["langfuse_usage_details"] = combine_usage_details(
        extract_langfuse_usage_details(search_response, search_response_data),
        extract_langfuse_usage_details(final_response, final_response_data),
    )
    return final_response, final_response_data


def build_mcp_finalizer_prompt(
    prompt: str,
    tool_calls: list[dict[str, Any]],
) -> str:
    return (
        f"{prompt.rstrip()}\n\n"
        "The previous attempt used theorem_search but ended with another tool call "
        "instead of a final answer. Tool use is now disabled. Use the attached files "
        "and the compact theorem_search transcript below. Return JSON only with "
        'exactly {"ai_id": <string or null>, "ai_num": <string or null>}. '
        "If no exact match is proven, use the best-supported guess from the search "
        "results rather than leaving both fields null.\n\n"
        "Compact theorem_search transcript:\n"
        f"{format_mcp_tool_call_summary(tool_calls)}"
    )


def format_mcp_tool_call_summary(tool_calls: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, call in enumerate(
        tool_calls[:GEMINI_MCP_FINALIZER_TOOL_CALL_LIMIT],
        start=1,
    ):
        query = normalize_summary_text(call.get("query")) or normalize_summary_text(
            call.get("input")
        )
        lines.append(f"{index}. theorem_search query: {query or '(unknown query)'}")
        theorems = extract_theorem_results_from_tool_call(call)
        if not theorems:
            lines.append("   results: none recorded")
            continue
        for result_index, theorem in enumerate(
            theorems[:GEMINI_MCP_FINALIZER_RESULT_LIMIT],
            start=1,
        ):
            paper = theorem.get("paper") if isinstance(theorem.get("paper"), dict) else {}
            paper_id = normalize_summary_text(paper.get("paper_id"))
            title = normalize_summary_text(paper.get("title"))
            name = normalize_summary_text(theorem.get("name"))
            score = theorem.get("score", theorem.get("similarity"))
            body = normalize_summary_text(theorem.get("body"))
            lines.append(
                "   "
                f"{result_index}) {paper_id or 'unknown arXiv'} "
                f"{name or 'unknown result'}"
                + (f" score={score}" if score is not None else "")
                + (f" title={title}" if title else "")
            )
            if body:
                lines.append(
                    "      body: "
                    + truncate_summary_text(body, GEMINI_MCP_FINALIZER_TEXT_LIMIT)
                )
    return "\n".join(lines) if lines else "(no theorem_search calls recorded)"


def extract_theorem_results_from_tool_call(call: dict[str, Any]) -> list[dict[str, Any]]:
    output = call.get("output")
    if not isinstance(output, dict):
        return []
    result = output.get("result")
    if not isinstance(result, dict):
        return []
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and isinstance(structured.get("theorems"), list):
        return [
            theorem
            for theorem in structured["theorems"]
            if isinstance(theorem, dict)
        ]

    content = result.get("content")
    if not isinstance(content, list):
        return []
    for item in content:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, dict) and isinstance(text.get("theorems"), list):
            return [
                theorem
                for theorem in text["theorems"]
                if isinstance(theorem, dict)
            ]
        if not isinstance(text, str):
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("theorems"), list):
            return [
                theorem
                for theorem in data["theorems"]
                if isinstance(theorem, dict)
            ]
    return []


def normalize_summary_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return truncate_summary_text(
            json.dumps(value, ensure_ascii=False, default=str),
            GEMINI_MCP_FINALIZER_TEXT_LIMIT,
        )
    return str(value).strip()


def truncate_summary_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


def read_response_id(response_data: dict[str, Any]) -> str | None:
    value = response_data.get("id") or response_data.get("response_id")
    return str(value) if value else None


def read_response_status(response_data: dict[str, Any]) -> str | None:
    value = response_data.get("status")
    if value:
        return str(value)
    candidates = response_data.get("candidates")
    if isinstance(candidates, list):
        finish_reasons = [
            str(candidate.get("finish_reason") or candidate.get("finishReason"))
            for candidate in candidates
            if isinstance(candidate, dict)
            and (candidate.get("finish_reason") or candidate.get("finishReason"))
        ]
        if finish_reasons:
            return ",".join(finish_reasons)
    return None


def combine_usage_details(*usage_details_items: dict[str, int]) -> dict[str, int]:
    combined: dict[str, int] = {}
    for usage_details in usage_details_items:
        for key, value in usage_details.items():
            try:
                numeric_value = int(value)
            except (TypeError, ValueError):
                continue
            combined[key] = combined.get(key, 0) + numeric_value
    return combined


def extract_output_text(response, response_data: dict[str, Any]) -> str:
    text_chunks: list[str] = []
    candidates = response_data.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            parts = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("thought"):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_chunks.append(text.strip())
    return "\n".join(text_chunks)


def extract_langfuse_usage_details(response, response_data: dict[str, Any]) -> dict[str, int]:
    explicit_usage = response_data.get("langfuse_usage_details")
    if isinstance(explicit_usage, dict):
        cleaned_usage: dict[str, int] = {}
        for key, value in explicit_usage.items():
            try:
                cleaned_usage[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        if cleaned_usage:
            return cleaned_usage

    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        usage = response_data.get("usage_metadata")
    if usage is None:
        return {}

    prompt_tokens = read_usage_int(usage, "prompt_token_count")
    output_tokens = coalesce_int(
        read_usage_int(usage, "candidates_token_count"),
        read_usage_int(usage, "response_token_count"),
    )
    total_tokens = read_usage_int(usage, "total_token_count")
    cached_tokens = read_usage_int(usage, "cached_content_token_count")
    thoughts_tokens = read_usage_int(usage, "thoughts_token_count")
    tool_use_prompt_tokens = read_usage_int(usage, "tool_use_prompt_token_count")

    usage_details: dict[str, int] = {}
    if prompt_tokens is not None:
        usage_details["input"] = prompt_tokens
    if output_tokens is not None:
        usage_details["output"] = output_tokens
    if total_tokens is not None:
        usage_details["total"] = total_tokens
    if cached_tokens is not None:
        usage_details["cached_input"] = cached_tokens
        usage_details["uncached_input"] = (
            max(prompt_tokens - cached_tokens, 0)
            if prompt_tokens is not None
            else 0
        )
    if thoughts_tokens is not None:
        usage_details["reasoning"] = thoughts_tokens
    if tool_use_prompt_tokens is not None:
        usage_details["tool_use_input"] = tool_use_prompt_tokens
    return usage_details


def read_usage_object(container: Any, field_name: str) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(field_name)
    return getattr(container, field_name, None)


def read_usage_int(container: Any, field_name: str) -> int | None:
    value = read_usage_object(container, field_name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coalesce_int(*values: int | None) -> int | None:
    for value in values:
        if value is not None:
            return value
    return None


def to_dict(response) -> dict[str, Any]:
    data = safe_serialize_value(response)
    if isinstance(data, dict):
        return data
    raise RuntimeError("Could not convert the Gemini response object to a dictionary.")


def format_exception_message(exc: BaseException) -> str:
    child_exceptions = getattr(exc, "exceptions", None)
    if isinstance(child_exceptions, tuple) and child_exceptions:
        child_messages = [
            format_exception_message(child)
            for child in child_exceptions
            if child is not None
        ]
        child_messages = [message for message in child_messages if message]
        if child_messages:
            return "; ".join(child_messages)
    return str(exc)


def main() -> int:
    args = parse_args()

    try:
        dotenv_path = load_environment_from_dotenv()
        tex_path = validate_input_path(args.tex_file, expected_suffix=".tex")
        bib_path = validate_input_path(args.bib_file, expected_suffix=(".txt", ".bib", ".bbl"))
        resolved_audit_log = (
            validate_input_path(args.audit_log, expected_suffix=".log")
            if args.audit_log is not None
            else None
        )
        api_key = read_api_key(args.api_key_env)
        ensure_runtime_dependencies()
        langfuse_runtime = setup_langfuse(args)
        client = make_client(api_key, enable_langfuse=langfuse_runtime.enabled)
        mcp_headers = parse_mcp_headers(args.mcp_headers or [])
        pricing = resolve_model_pricing(args)
        mcp_tools = args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if dotenv_path:
        print(f"Loaded environment from {dotenv_path}", file=sys.stderr)
    if langfuse_runtime.status_message:
        print(langfuse_runtime.status_message, file=sys.stderr)

    uploaded_file_names: list[str] = []
    trace_url: str | None = None
    upload_temp_dir: Path | None = None
    try:
        upload_temp_dir = create_upload_temp_dir()
        upload_inputs = prepare_upload_inputs(
            tex_path=tex_path,
            bib_path=bib_path,
            audit_log_path=resolved_audit_log,
            log_entry_index=args.log_entry,
            temp_dir=upload_temp_dir,
            body_context_lines=args.body_context_lines,
        )
        for message in upload_inputs.status_messages:
            print(message, file=sys.stderr)

        trace_context = (
            langfuse_runtime.client.start_as_current_observation(
                name=args.langfuse_trace_name,
                as_type="span",
                input=build_langfuse_observation_input(
                    tex_path=tex_path,
                    bib_path=bib_path,
                    user_prompt=args.prompt,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    mcp_url=args.mcp_url,
                    mcp_label=args.mcp_label,
                    mcp_tools=mcp_tools,
                    retrieval_mode=args.retrieval_mode,
                ),
                metadata=build_langfuse_observation_metadata(
                    tex_path=tex_path,
                    bib_path=bib_path,
                    mcp_url=args.mcp_url,
                    mcp_label=args.mcp_label,
                    mcp_tools=mcp_tools,
                    retrieval_mode=args.retrieval_mode,
                ),
            )
            if langfuse_runtime.enabled
            else nullcontext(None)
        )
        with trace_context as trace_observation:
            propagation_context = (
                langfuse_runtime.propagate_attributes(
                    user_id=args.langfuse_user_id,
                    session_id=args.langfuse_session_id,
                    tags=args.langfuse_tags,
                    trace_name=args.langfuse_trace_name,
                    metadata=build_langfuse_trace_metadata(
                        tex_path=tex_path,
                        bib_path=bib_path,
                        model=args.model,
                        retrieval_mode=args.retrieval_mode,
                    ),
                )
                if langfuse_runtime.enabled and langfuse_runtime.propagate_attributes
                else nullcontext()
            )
            with propagation_context:
                tex_upload = upload_file(client, upload_inputs.tex_path)
                bib_upload = upload_file(client, upload_inputs.bib_path)

                tex_upload_name = getattr(tex_upload, "name", "")
                bib_upload_name = getattr(bib_upload, "name", "")
                uploaded_file_names.extend(
                    [name for name in (tex_upload_name, bib_upload_name) if name]
                )

                generation_context = (
                    langfuse_runtime.client.start_as_current_observation(
                        name="google.genai.models.generate_content",
                        as_type="generation",
                        input=build_langfuse_generation_input(
                            tex_path=tex_path,
                            bib_path=bib_path,
                            user_prompt=args.prompt,
                            mcp_url=args.mcp_url,
                            mcp_label=args.mcp_label,
                            mcp_tools=mcp_tools,
                            retrieval_mode=args.retrieval_mode,
                        ),
                        metadata=build_langfuse_generation_metadata(
                            tex_path=tex_path,
                            bib_path=bib_path,
                            mcp_url=args.mcp_url,
                            mcp_label=args.mcp_label,
                            mcp_tools=mcp_tools,
                            pricing=pricing,
                            retrieval_mode=args.retrieval_mode,
                        ),
                        model=args.model,
                        model_parameters=build_langfuse_model_parameters(args),
                    )
                    if langfuse_runtime.enabled
                    else nullcontext(None)
                )
                with generation_context as generation_observation:
                    response, response_data, available_tool_names, thinking_note = run_response(
                        client=client,
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        max_output_tokens=args.max_output_tokens,
                        prompt=build_user_prompt(
                            tex_path=upload_inputs.tex_path,
                            bib_path=upload_inputs.bib_path,
                            user_prompt=args.prompt,
                            context_note=upload_inputs.prompt_note,
                            retrieval_mode=args.retrieval_mode,
                        ),
                        tex_upload=tex_upload,
                        bib_upload=bib_upload,
                        mcp_label=args.mcp_label,
                        mcp_url=args.mcp_url,
                        requested_tools=mcp_tools,
                        mcp_headers=mcp_headers,
                        retrieval_mode=args.retrieval_mode,
                    )
                    if thinking_note:
                        print(thinking_note, file=sys.stderr)
                    if args.retrieval_mode == "mcp":
                        if available_tool_names == mcp_tools:
                            print(
                                "MCP session tools: " + ", ".join(available_tool_names) + ".",
                                file=sys.stderr,
                            )
                        else:
                            print(
                                "Requested MCP tools: "
                                + ", ".join(mcp_tools)
                                + "; MCP session exposes: "
                                + ", ".join(available_tool_names)
                                + " (Gemini SDK MCP sessions do not enforce client-side tool allowlists).",
                                file=sys.stderr,
                            )
                    elif args.retrieval_mode == "web-search":
                        print("Gemini retrieval tools: google_search.", file=sys.stderr)
                    else:
                        print("External retrieval disabled.", file=sys.stderr)

                    output_text = extract_output_text(response, response_data)
                    usage_details = extract_langfuse_usage_details(response, response_data)
                    cost_details = compute_langfuse_cost_details(usage_details, pricing)
                    tool_calls = extract_langfuse_tool_calls(response_data)

                    if generation_observation is not None:
                        record_langfuse_tool_calls(langfuse_runtime.client, tool_calls)
                        generation_observation.update(
                            output=build_langfuse_generation_output(
                                response_data=response_data,
                                output_text=output_text,
                                usage_details=usage_details,
                                cost_details=cost_details,
                            ),
                            metadata=build_langfuse_generation_result_metadata(
                                pricing=pricing,
                                usage_details=usage_details,
                                cost_details=cost_details,
                            ),
                            model=args.model,
                            model_parameters=build_langfuse_model_parameters(args),
                            usage_details=usage_details or None,
                            cost_details=cost_details or None,
                        )
                        trace_url = langfuse_runtime.client.get_trace_url()

                    if trace_observation is not None:
                        trace_observation.update(
                            output=build_langfuse_observation_output(
                                response_data=response_data,
                                output_text=output_text,
                                usage_details=usage_details,
                                cost_details=cost_details,
                            )
                        )

                    if output_text:
                        print(output_text)
                    else:
                        print(
                            json.dumps(response_data, ensure_ascii=False, indent=2),
                            file=sys.stdout,
                        )

                    if args.output:
                        write_text_output(args.output, output_text, response_data)
                    if args.json_output:
                        write_json_output(args.json_output, response_data)
    except Exception as exc:
        print(f"Error: {format_exception_message(exc)}", file=sys.stderr)
        return 1
    finally:
        cleanup_upload_temp_dir(upload_temp_dir)
        if args.delete_uploads:
            cleanup_uploaded_files(client if "client" in locals() else None, uploaded_file_names)
        if "langfuse_runtime" in locals() and langfuse_runtime.enabled:
            flush_langfuse(langfuse_runtime)
        if "client" in locals():
            try:
                client.close()
            except Exception:
                pass

    if trace_url:
        print(f"Langfuse trace: {trace_url}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
