#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
import json
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Sequence

from query_openai_tex_mcp import (
    DEFAULT_ALLOWED_MCP_TOOLS,
    DEFAULT_APPROVAL_LIMIT,
    DEFAULT_BODY_CONTEXT_LINES,
    DEFAULT_LANGFUSE_TRACE_NAME,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MCP_LABEL,
    DEFAULT_MCP_URL,
    ModelPricing,
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
    parse_mcp_headers,
    parse_nonnegative_int,
    prepare_upload_inputs,
    read_optional_float_env,
    record_langfuse_tool_calls,
    safe_serialize_value,
    setup_langfuse,
    validate_input_path,
    write_json_output,
    write_text_output,
)
from statement_reference_audit import read_candidate_text


DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MCP_READ_TIMEOUT = 300.0
DEFAULT_CLAUDE_INPUT_COST_ENV = "CLAUDE_INPUT_COST_PER_1M"
DEFAULT_CLAUDE_CACHED_INPUT_COST_ENV = "CLAUDE_CACHED_INPUT_COST_PER_1M"
DEFAULT_CLAUDE_OUTPUT_COST_ENV = "CLAUDE_OUTPUT_COST_PER_1M"
LEGACY_CLAUDE_INPUT_COST_ENV = "CLAUDE_BEDROCK_INPUT_COST_PER_1M"
LEGACY_CLAUDE_CACHED_INPUT_COST_ENV = "CLAUDE_BEDROCK_CACHED_INPUT_COST_PER_1M"
LEGACY_CLAUDE_OUTPUT_COST_ENV = "CLAUDE_BEDROCK_OUTPUT_COST_PER_1M"
DEFAULT_CLAUDE_JSON_FINALIZER_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CLAUDE_API_MAX_RETRIES = 5
DEFAULT_CLAUDE_API_RETRY_INITIAL_DELAY = 2.0
DEFAULT_CLAUDE_API_RETRY_MAX_DELAY = 60.0
CLAUDE_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}
CLAUDE_RETRYABLE_ERROR_TYPES = {
    "api_connection_error",
    "api_error",
    "api_timeout_error",
    "overloaded_error",
    "rate_limit_error",
}
CLAUDE_JSON_SOURCE_OUTPUT_TEXT_LIMIT = 4000
REQUIRED_AUDIT_JSON_KEYS = ("ai_id", "ai_num")
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class ClaudeClientHandle:
    messages_client: Any
    api_key_env: str = DEFAULT_API_KEY_ENV

    def close(self) -> None:
        close = getattr(self.messages_client, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class ClaudeFileRef:
    path: Path
    name: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send a .tex file and bibliography source to Claude through the "
            "Anthropic API, and let the model use the TheoremSearch MCP server."
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
            "Number of main TeX body lines before and after the logged citation line to send. "
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
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Anthropic Claude model ID to use. "
            f"Defaults to {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Claude extended-thinking effort. Use none to disable thinking. "
            f"Defaults to {DEFAULT_REASONING_EFFORT}."
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
            "Environment variable containing the Anthropic API key. "
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
        "--mcp-read-timeout",
        type=float,
        default=DEFAULT_MCP_READ_TIMEOUT,
        metavar="SECONDS",
        help=(
            "Timeout for individual MCP tool calls in seconds. "
            f"Defaults to {DEFAULT_MCP_READ_TIMEOUT:g}."
        ),
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
    parser.add_argument("--langfuse-user-id", help="Optional Langfuse user ID for the trace.")
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
        help="Override model input pricing in USD per 1M tokens for Langfuse cost tracking.",
    )
    parser.add_argument(
        "--cached-input-cost-per-1m",
        type=float,
        help="Override cached-input pricing in USD per 1M tokens for Langfuse cost tracking.",
    )
    parser.add_argument(
        "--output-cost-per-1m",
        type=float,
        help="Override model output pricing in USD per 1M tokens for Langfuse cost tracking.",
    )
    parser.add_argument("--output", type=Path, help="Optional path to write the model text response.")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write the final Claude response JSON.",
    )
    parser.add_argument(
        "--delete-uploads",
        action="store_true",
        help="Accepted for CLI parity; Claude text uploads are local-only.",
    )
    return parser.parse_args()


def create_upload_temp_dir() -> Path:
    base_dir = (Path.cwd() / ".claude_upload_tmp").resolve()
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
    shutil.rmtree(resolved, ignore_errors=True)


def resolve_model_pricing(args: argparse.Namespace) -> ModelPricing | None:
    input_cost = coalesce_float(
        args.input_cost_per_1m,
        read_optional_float_env(DEFAULT_CLAUDE_INPUT_COST_ENV),
        read_optional_float_env(LEGACY_CLAUDE_INPUT_COST_ENV),
    )
    cached_input_cost = coalesce_float(
        args.cached_input_cost_per_1m,
        read_optional_float_env(DEFAULT_CLAUDE_CACHED_INPUT_COST_ENV),
        read_optional_float_env(LEGACY_CLAUDE_CACHED_INPUT_COST_ENV),
        input_cost,
    )
    output_cost = coalesce_float(
        args.output_cost_per_1m,
        read_optional_float_env(DEFAULT_CLAUDE_OUTPUT_COST_ENV),
        read_optional_float_env(LEGACY_CLAUDE_OUTPUT_COST_ENV),
    )

    if input_cost is None or cached_input_cost is None or output_cost is None:
        return None

    if (
        args.input_cost_per_1m is not None
        or args.cached_input_cost_per_1m is not None
        or args.output_cost_per_1m is not None
    ):
        source = "CLI pricing override"
    else:
        source = "environment pricing override"

    return ModelPricing(
        input_cost_per_1m=input_cost,
        cached_input_cost_per_1m=cached_input_cost,
        output_cost_per_1m=output_cost,
        source=source,
    )


def ensure_runtime_dependencies() -> None:
    try:
        import anthropic  # noqa: F401
        import httpx  # noqa: F401
        from mcp import ClientSession  # noqa: F401
        from mcp.client.streamable_http import streamable_http_client  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The Claude API path requires `anthropic`, `httpx`, and `mcp`. "
            "Install them with `pip install anthropic httpx mcp`."
        ) from exc


def make_client_from_args(args: argparse.Namespace, *, enable_langfuse: bool):
    api_key_env = getattr(args, "api_key_env", None) or DEFAULT_API_KEY_ENV
    return make_client(
        read_api_key_from_env(api_key_env),
        enable_langfuse=enable_langfuse,
        api_key_env=api_key_env,
    )


def make_client(
    api_key: str | None = None,
    *,
    enable_langfuse: bool,
    api_key_env: str = DEFAULT_API_KEY_ENV,
):
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The `anthropic` package is not installed. Install it with "
            "`pip install anthropic` and rerun the script."
        ) from exc

    if not api_key:
        raise RuntimeError(f"Missing Anthropic API key in environment variable {api_key_env}.")

    return ClaudeClientHandle(messages_client=Anthropic(api_key=api_key), api_key_env=api_key_env)


def read_api_key_from_env(api_key_env: str) -> str:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing Anthropic API key in environment variable {api_key_env}.")
    return api_key


def upload_file(client, path: Path) -> ClaudeFileRef:
    return ClaudeFileRef(
        path=path.resolve(),
        name=path.name,
        text=read_candidate_text(path),
    )


def cleanup_uploaded_files(client, uploaded_file_names: list[str]) -> None:
    return None


def run_response(
    client: ClaudeClientHandle,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    prompt: str,
    tex_upload: ClaudeFileRef,
    bib_upload: ClaudeFileRef,
    mcp_label: str,
    mcp_url: str,
    requested_tools: list[str],
    mcp_headers: dict[str, str],
    mcp_read_timeout: float = DEFAULT_MCP_READ_TIMEOUT,
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
            mcp_read_timeout=mcp_read_timeout,
        )
    )


async def _run_response_async(
    client: ClaudeClientHandle,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    prompt: str,
    tex_upload: ClaudeFileRef,
    bib_upload: ClaudeFileRef,
    mcp_label: str,
    mcp_url: str,
    requested_tools: list[str],
    mcp_headers: dict[str, str],
    mcp_read_timeout: float,
):
    try:
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as exc:
        raise RuntimeError(
            "Missing Claude runtime dependency. Install `anthropic`, `httpx`, and `mcp`."
        ) from exc

    http_client = httpx.AsyncClient(
        headers=mcp_headers or None,
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0),
    )
    async with http_client:
        async with streamable_http_client(
            mcp_url,
            http_client=http_client,
            terminate_on_close=False,
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                list_tools_result = await session.list_tools()
                tool_specs, available_tool_names = build_claude_tool_specs(
                    getattr(list_tools_result, "tools", []),
                    requested_tools,
                )
                if not tool_specs:
                    raise RuntimeError(
                        "The MCP session did not expose any requested Claude-compatible tools."
                    )

                messages = [
                    {
                        "role": "user",
                        "content": build_initial_user_content(prompt, tex_upload, bib_upload),
                    }
                ]
                additional_fields, thinking_note = build_additional_model_request_fields(
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                )
                response = converse(
                    client=client,
                    model=model,
                    messages=messages,
                    tool_specs=tool_specs,
                    max_output_tokens=max_output_tokens,
                    additional_model_request_fields=additional_fields,
                )

                tool_rounds = 0
                mcp_tool_calls: list[dict[str, Any]] = []
                for _ in range(DEFAULT_APPROVAL_LIMIT):
                    response_data = normalize_claude_response(response)
                    response_message = response_data.get("output", {}).get("message")
                    if isinstance(response_message, dict):
                        messages.append(response_message)

                    tool_uses = extract_tool_uses(response_message)
                    if not tool_uses:
                        if prompt_requests_required_audit_json(prompt):
                            response, response_data = ensure_required_json_response(
                                client=client,
                                model=model,
                                max_output_tokens=max_output_tokens,
                                response=response,
                                response_data=response_data,
                            )
                        response_data["mcp_available_tools"] = available_tool_names
                        response_data["mcp_tool_rounds"] = tool_rounds
                        response_data["langfuse_tool_calls"] = mcp_tool_calls
                        return response, response_data, available_tool_names, thinking_note

                    tool_rounds += 1
                    tool_result_blocks = []
                    for tool_use in tool_uses:
                        tool_result = await call_mcp_tool_for_claude(
                            session=session,
                            tool_use=tool_use,
                            allowed_tool_names=set(available_tool_names),
                            mcp_read_timeout=mcp_read_timeout,
                        )
                        tool_result_blocks.append(tool_result)
                        mcp_tool_calls.append(
                            build_claude_langfuse_tool_call(
                                tool_use=tool_use,
                                tool_result_block=tool_result,
                                round_index=tool_rounds,
                            )
                        )
                    messages.append({"role": "user", "content": tool_result_blocks})
                    response = converse(
                        client=client,
                        model=model,
                        messages=messages,
                        tool_specs=tool_specs,
                        max_output_tokens=max_output_tokens,
                        additional_model_request_fields=additional_fields,
                    )

                raise RuntimeError("Too many MCP tool-use rounds. Try rerunning with a simpler prompt.")


def build_initial_user_content(
    prompt: str,
    tex_upload: ClaudeFileRef,
    bib_upload: ClaudeFileRef,
) -> list[dict[str, str]]:
    return [
        {"type": "text", "text": prompt},
        {"type": "text", "text": render_text_file_block("LaTeX source", tex_upload)},
        {"type": "text", "text": render_text_file_block("Bibliography source", bib_upload)},
    ]


def render_text_file_block(label: str, file_ref: ClaudeFileRef) -> str:
    return (
        f"\n\n===== {label}: {file_ref.name} =====\n"
        f"{file_ref.text.rstrip()}\n"
        f"===== End {label}: {file_ref.name} ====="
    )


def build_claude_tool_specs(
    mcp_tools: Sequence[Any],
    requested_tools: Sequence[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    requested = {tool_name for tool_name in requested_tools if tool_name}
    tool_specs: list[dict[str, Any]] = []
    available_tool_names: list[str] = []

    for tool in mcp_tools:
        name = normalize_tool_name(getattr(tool, "name", ""))
        if not name or name not in requested or not TOOL_NAME_RE.match(name):
            continue

        schema = read_tool_input_schema(tool)
        description = normalize_tool_description(getattr(tool, "description", ""))
        tool_specs.append({"name": name, "description": description, "input_schema": schema})
        available_tool_names.append(name)

    return tool_specs, available_tool_names


def normalize_tool_name(value: object) -> str:
    return str(value).strip() if value is not None else ""


def normalize_tool_description(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "Search theorem statements and references."


def read_tool_input_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "inputSchema", None)
    if schema is None:
        schema = getattr(tool, "input_schema", None)
    serialized = safe_serialize_value(schema)
    if isinstance(serialized, dict) and serialized:
        return serialized
    return {"type": "object", "properties": {}}


def build_additional_model_request_fields(
    *,
    reasoning_effort: str,
    max_output_tokens: int,
) -> tuple[dict[str, Any], str | None]:
    if reasoning_effort == "none":
        return {}, None

    budget_map = {
        "low": 1024,
        "medium": 4096,
        "high": 8192,
        "xhigh": 16384,
    }
    requested_budget = budget_map[reasoning_effort]
    budget = min(requested_budget, max_output_tokens - 1)
    if budget < 1024:
        return {}, (
            "Claude extended thinking disabled because max-output-tokens is too low "
            "for Anthropic's minimum thinking budget."
        )
    note = None
    if budget != requested_budget:
        note = (
            f"Claude thinking budget reduced from {requested_budget} to {budget} "
            "so it stays below max-output-tokens."
        )
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}, note


def converse(
    *,
    client: ClaudeClientHandle,
    model: str,
    messages: list[dict[str, Any]],
    tool_specs: list[dict[str, Any]] | None,
    max_output_tokens: int,
    additional_model_request_fields: dict[str, Any] | None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": normalize_anthropic_model_id(model),
        "messages": messages,
        "system": system_prompt or build_system_prompt(),
        "max_tokens": max_output_tokens,
    }
    if tool_specs:
        request["tools"] = tool_specs
    if additional_model_request_fields:
        request.update(additional_model_request_fields)
    for attempt_index in range(DEFAULT_CLAUDE_API_MAX_RETRIES + 1):
        try:
            return client.messages_client.messages.create(**request)
        except Exception as exc:
            if (
                attempt_index >= DEFAULT_CLAUDE_API_MAX_RETRIES
                or not is_retryable_anthropic_error(exc)
            ):
                raise RuntimeError(
                    "Anthropic Messages API operation failed after "
                    f"{attempt_index + 1} attempt(s): {format_exception_message(exc)}"
                ) from exc
            delay = compute_anthropic_retry_delay(attempt_index)
            print(
                "Warning: Anthropic Messages API transient failure "
                f"({summarize_anthropic_error(exc)}); retrying in {delay:g}s "
                f"[attempt {attempt_index + 2}/{DEFAULT_CLAUDE_API_MAX_RETRIES + 1}].",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError("Anthropic Messages API operation failed unexpectedly.")


def is_retryable_anthropic_error(exc: BaseException) -> bool:
    status_code = read_exception_status_code(exc)
    if status_code in CLAUDE_RETRYABLE_STATUS_CODES:
        return True

    error_type = read_exception_error_type(exc)
    if error_type in CLAUDE_RETRYABLE_ERROR_TYPES:
        return True

    class_name = exc.__class__.__name__.lower()
    return any(
        marker in class_name
        for marker in ("overloaded", "ratelimit", "rate_limit", "timeout", "connection")
    )


def compute_anthropic_retry_delay(attempt_index: int) -> float:
    delay = DEFAULT_CLAUDE_API_RETRY_INITIAL_DELAY * (2 ** attempt_index)
    return min(delay, DEFAULT_CLAUDE_API_RETRY_MAX_DELAY)


def summarize_anthropic_error(exc: BaseException) -> str:
    status_code = read_exception_status_code(exc)
    error_type = read_exception_error_type(exc)
    parts = []
    if status_code is not None:
        parts.append(str(status_code))
    if error_type:
        parts.append(error_type)
    if parts:
        return " ".join(parts)
    return exc.__class__.__name__


def read_exception_status_code(exc: BaseException) -> int | None:
    for field_name in ("status_code", "status"):
        value = getattr(exc, field_name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return read_status_code_from_mapping(getattr(exc, "body", None))


def read_status_code_from_mapping(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    for field_name in ("status_code", "status"):
        raw_value = value.get(field_name)
        if raw_value is None:
            continue
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            pass
    return None


def read_exception_error_type(exc: BaseException) -> str:
    error_type = read_error_type_from_mapping(getattr(exc, "body", None))
    if error_type:
        return error_type
    error = getattr(exc, "error", None)
    error_type = read_error_type_from_mapping(safe_serialize_value(error))
    if error_type:
        return error_type
    return ""


def read_error_type_from_mapping(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    nested_error = value.get("error")
    if isinstance(nested_error, dict):
        nested_type = nested_error.get("type")
        if isinstance(nested_type, str) and nested_type.strip():
            return nested_type.strip().lower()
    direct_type = value.get("type")
    if isinstance(direct_type, str) and direct_type.strip():
        return direct_type.strip().lower()
    return ""


def normalize_anthropic_model_id(model: str) -> str:
    model = str(model or "").strip()
    match = re.fullmatch(r"(?:[a-z]{2}\.)?anthropic\.(.+?)-v\d+:\d+", model)
    if match:
        return match.group(1)
    return model


def normalize_claude_response(response: dict[str, Any]) -> dict[str, Any]:
    raw_response = to_dict(response)
    stop_reason = str(raw_response.get("stop_reason") or raw_response.get("stopReason") or "").strip()
    content = raw_response.get("content")
    if not isinstance(content, list):
        content = []
    response_data: dict[str, Any] = {
        "provider": "claude",
        "id": raw_response.get("id"),
        "model": raw_response.get("model"),
        "role": raw_response.get("role"),
        "type": raw_response.get("type"),
        "stop_reason": stop_reason or None,
        "stopReason": stop_reason or None,
        "usage": raw_response.get("usage"),
        "output": {
            "message": {
                "role": "assistant",
                "content": [normalize_claude_content_block(block) for block in content],
            }
        },
        "raw_response": raw_response,
    }
    if stop_reason == "end_turn":
        response_data["status"] = "completed"
    elif stop_reason:
        response_data["status"] = stop_reason
    return response_data


def normalize_claude_content_block(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        block_type = block.get("type")
        if block_type == "text":
            return {"type": "text", "text": block.get("text", "")}
        if block_type == "tool_use":
            return {
                "type": "tool_use",
                "id": block.get("id"),
                "name": block.get("name"),
                "input": block.get("input") if isinstance(block.get("input"), dict) else {},
            }
        return block

    block_type = getattr(block, "type", None)
    if block_type == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if block_type == "tool_use":
        tool_input = getattr(block, "input", None)
        return {
            "type": "tool_use",
            "id": getattr(block, "id", None),
            "name": getattr(block, "name", None),
            "input": tool_input if isinstance(tool_input, dict) else safe_serialize_value(tool_input),
        }
    serialized = safe_serialize_value(block)
    return serialized if isinstance(serialized, dict) else {"type": str(block_type or "unknown")}


def prompt_requests_required_audit_json(prompt: str) -> bool:
    return all(f'"{key}"' in prompt for key in REQUIRED_AUDIT_JSON_KEYS)


def ensure_required_json_response(
    *,
    client: ClaudeClientHandle,
    model: str,
    max_output_tokens: int,
    response: dict[str, Any],
    response_data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_text = extract_output_text(response, response_data)
    normalized_text = normalize_required_audit_json_text(output_text)
    if normalized_text is not None:
        if (
            not is_bare_required_audit_json_output(output_text)
            or normalized_text != output_text.strip()
        ):
            rewrite_response_output_text(response_data, normalized_text)
            annotate_json_rewrite(
                response_data=response_data,
                source_output_text=output_text,
                flag_name="claude_json_normalized_after_non_json_text",
            )
        return response, response_data

    return finalize_required_json_response(
        client=client,
        model=model,
        max_output_tokens=max_output_tokens,
        source_response=response,
        source_response_data=response_data,
        source_output_text=output_text,
    )


def finalize_required_json_response(
    *,
    client: ClaudeClientHandle,
    model: str,
    max_output_tokens: int,
    source_response: dict[str, Any],
    source_response_data: dict[str, Any],
    source_output_text: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_usage = extract_langfuse_usage_details(source_response, source_response_data)
    final_response = converse(
        client=client,
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_required_json_finalizer_prompt(source_output_text),
                    }
                ],
            }
        ],
        tool_specs=None,
        max_output_tokens=min(
            max_output_tokens,
            DEFAULT_CLAUDE_JSON_FINALIZER_MAX_OUTPUT_TOKENS,
        ),
        additional_model_request_fields=None,
        system_prompt=build_required_json_finalizer_system_prompt(),
    )
    final_response_data = normalize_claude_response(final_response)
    final_output_text = extract_output_text(final_response, final_response_data)
    normalized_text = normalize_required_audit_json_text(final_output_text)
    finalizer_parse_failed = normalized_text is None
    if normalized_text is None:
        normalized_text = render_required_audit_json_object({"ai_id": None, "ai_num": None})

    rewrite_response_output_text(final_response_data, normalized_text)
    annotate_json_rewrite(
        response_data=final_response_data,
        source_output_text=source_output_text,
        flag_name="claude_json_finalized_after_non_json_text",
    )
    final_response_data["claude_json_source_response_status"] = source_response_data.get(
        "status"
    )
    if finalizer_parse_failed:
        final_response_data["claude_json_finalizer_parse_failed"] = True
        final_response_data["claude_json_finalizer_output_text"] = truncate_text(
            final_output_text,
            CLAUDE_JSON_SOURCE_OUTPUT_TEXT_LIMIT,
        )

    finalizer_usage = extract_langfuse_usage_details(final_response, final_response_data)
    combined_usage = combine_langfuse_usage_details(source_usage, finalizer_usage)
    if combined_usage:
        final_response_data["langfuse_usage_details"] = combined_usage
    return final_response, final_response_data


def build_required_json_finalizer_system_prompt() -> str:
    return (
        "You are a strict JSON finalizer. Return exactly one JSON object. "
        "Do not include markdown, code fences, prose, analysis, or explanations."
    )


def build_required_json_finalizer_prompt(source_output_text: str) -> str:
    previous_answer = source_output_text.strip() or "(empty response)"
    return (
        "Rewrite the previous assistant answer as exactly one JSON object with this schema:\n"
        '{"ai_id": <string or null>, "ai_num": <string or null>}\n\n'
        "Rules:\n"
        "- Preserve the answer values already given by the previous assistant answer.\n"
        "- Use null for a value that is unknown, unavailable, or already null.\n"
        "- Return no markdown, no code fence, and no surrounding text.\n\n"
        "Previous assistant answer:\n"
        f"{truncate_text(previous_answer, CLAUDE_JSON_SOURCE_OUTPUT_TEXT_LIMIT)}"
    )


def normalize_required_audit_json_text(output_text: str) -> str | None:
    value = parse_bare_json_value(output_text)
    if has_required_audit_json_fields(value):
        return render_required_audit_json_object(value)

    decoder = json.JSONDecoder()
    for index, char in enumerate(output_text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(output_text[index:])
        except json.JSONDecodeError:
            continue
        if has_required_audit_json_fields(value):
            return render_required_audit_json_object(value)
    return None


def is_bare_required_audit_json_output(output_text: str) -> bool:
    stripped = output_text.strip()
    if not stripped:
        return False
    decoder = json.JSONDecoder()
    try:
        value, end_index = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return False
    if stripped[end_index:].strip():
        return False
    return has_exact_required_audit_json_shape(value)


def parse_bare_json_value(output_text: str) -> Any:
    stripped = output_text.strip()
    if not stripped:
        return None
    decoder = json.JSONDecoder()
    try:
        value, end_index = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    if stripped[end_index:].strip():
        return None
    return value


def has_required_audit_json_fields(value: Any) -> bool:
    return isinstance(value, dict) and all(key in value for key in REQUIRED_AUDIT_JSON_KEYS)


def has_exact_required_audit_json_shape(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if set(value) != set(REQUIRED_AUDIT_JSON_KEYS):
        return False
    return all(
        value[key] is None or isinstance(value[key], str)
        for key in REQUIRED_AUDIT_JSON_KEYS
    )


def render_required_audit_json_object(value: dict[str, Any]) -> str:
    rendered = {
        key: coerce_required_json_value(value.get(key))
        for key in REQUIRED_AUDIT_JSON_KEYS
    }
    return json.dumps(rendered, ensure_ascii=False, separators=(", ", ": "))


def coerce_required_json_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def rewrite_response_output_text(response_data: dict[str, Any], output_text: str) -> None:
    output = response_data.get("output")
    if not isinstance(output, dict):
        output = {}
        response_data["output"] = output
    message = output.get("message")
    if not isinstance(message, dict):
        message = {"role": "assistant"}
        output["message"] = message
    message["content"] = [{"type": "text", "text": output_text}]


def annotate_json_rewrite(
    *,
    response_data: dict[str, Any],
    source_output_text: str,
    flag_name: str,
) -> None:
    response_data[flag_name] = True
    response_data["claude_json_source_output_text"] = truncate_text(
        source_output_text,
        CLAUDE_JSON_SOURCE_OUTPUT_TEXT_LIMIT,
    )


def truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(limit - 20, 0)].rstrip() + "\n... [truncated]"


def extract_tool_uses(response_message: Any) -> list[dict[str, Any]]:
    if not isinstance(response_message, dict):
        return []
    content = response_message.get("content")
    if not isinstance(content, list):
        return []
    tool_uses: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("toolUse"), dict):
            tool_uses.append(block["toolUse"])
            continue
        if block.get("type") == "tool_use":
            tool_uses.append(block)
    return tool_uses


async def call_mcp_tool_for_claude(
    *,
    session: Any,
    tool_use: dict[str, Any],
    allowed_tool_names: set[str],
    mcp_read_timeout: float,
) -> dict[str, Any]:
    tool_use_id = str(tool_use.get("id") or tool_use.get("toolUseId") or "")
    tool_name = str(tool_use.get("name") or "")
    tool_input = tool_use.get("input")
    if not isinstance(tool_input, dict):
        tool_input = {}

    if not tool_use_id:
        return build_tool_result_block(
            tool_use_id="missing-tool-use-id",
            payload={"error": "Claude did not provide a tool use id."},
            status="error",
        )
    if tool_name not in allowed_tool_names:
        return build_tool_result_block(
            tool_use_id=tool_use_id,
            payload={"error": f"Tool {tool_name!r} is not in the allowed MCP tool list."},
            status="error",
        )

    try:
        read_timeout = (
            timedelta(seconds=mcp_read_timeout)
            if mcp_read_timeout and mcp_read_timeout > 0
            else None
        )
        result = await session.call_tool(
            tool_name,
            arguments=tool_input,
            read_timeout_seconds=read_timeout,
        )
        return build_tool_result_block(
            tool_use_id=tool_use_id,
            payload=safe_serialize_value(result),
            status="success",
        )
    except Exception as exc:
        return build_tool_result_block(
            tool_use_id=tool_use_id,
            payload={"error": format_exception_message(exc)},
            status="error",
        )


def build_tool_result_block(
    *,
    tool_use_id: str,
    payload: Any,
    status: str,
) -> dict[str, Any]:
    content = json.dumps(payload, ensure_ascii=False)
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": status == "error",
    }


def build_claude_langfuse_tool_call(
    *,
    tool_use: dict[str, Any],
    tool_result_block: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    return {
        "provider": "claude",
        "tool_type": "mcp",
        "name": tool_use.get("name"),
        "id": tool_use.get("id") or tool_use.get("toolUseId"),
        "status": "error" if tool_result_block.get("is_error") else "success",
        "input": tool_use.get("input"),
        "output": extract_claude_tool_result_payload(tool_result_block),
        "raw_type": "tool_use",
        "round": round_index,
    }


def extract_claude_tool_result_payload(tool_result: dict[str, Any]) -> Any:
    content = tool_result.get("content")
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def extract_output_text(response, response_data: dict[str, Any]) -> str:
    message = response_data.get("output", {}).get("message")
    if not isinstance(message, dict):
        return ""
    text_chunks: list[str] = []
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            text_chunks.append(text.strip())
    return "\n".join(text_chunks)


def extract_langfuse_usage_details(response, response_data: dict[str, Any]) -> dict[str, int]:
    explicit_usage = response_data.get("langfuse_usage_details")
    if isinstance(explicit_usage, dict):
        usage_details: dict[str, int] = {}
        for key, value in explicit_usage.items():
            if value is None:
                continue
            try:
                usage_details[str(key)] = int(value)
            except (TypeError, ValueError):
                continue
        return usage_details

    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        return {}

    input_tokens = read_first_usage_int(usage, ("input_tokens", "inputTokens"))
    output_tokens = read_first_usage_int(usage, ("output_tokens", "outputTokens"))
    total_tokens = read_first_usage_int(usage, ("total_tokens", "totalTokens"))
    cache_read_tokens = read_first_usage_int(
        usage,
        ("cache_read_input_tokens", "cacheReadInputTokens"),
    )
    cache_write_tokens = read_first_usage_int(
        usage,
        ("cache_creation_input_tokens", "cacheWriteInputTokens"),
    )

    usage_details: dict[str, int] = {}
    if input_tokens is not None:
        usage_details["input"] = input_tokens
    if output_tokens is not None:
        usage_details["output"] = output_tokens
    if total_tokens is not None:
        usage_details["total"] = total_tokens
    cached_tokens = (cache_read_tokens or 0) + (cache_write_tokens or 0)
    if cached_tokens:
        usage_details["cached_input"] = cached_tokens
        usage_details["uncached_input"] = max(input_tokens - cached_tokens, 0) if input_tokens else 0
    return usage_details


def read_first_usage_int(container: dict[str, Any], field_names: Sequence[str]) -> int | None:
    for field_name in field_names:
        value = read_usage_int(container, field_name)
        if value is not None:
            return value
    return None


def combine_langfuse_usage_details(*usage_items: dict[str, int]) -> dict[str, int]:
    combined: dict[str, int] = {}
    for usage in usage_items:
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if value is None:
                continue
            try:
                numeric_value = int(value)
            except (TypeError, ValueError):
                continue
            combined[key] = combined.get(key, 0) + numeric_value
    if "total" not in combined and ("input" in combined or "output" in combined):
        combined["total"] = combined.get("input", 0) + combined.get("output", 0)
    return combined


def read_usage_int(container: dict[str, Any], field_name: str) -> int | None:
    value = container.get(field_name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_dict(response) -> dict[str, Any]:
    data = safe_serialize_value(response)
    if isinstance(data, dict):
        return data
    raise RuntimeError("Could not convert the Claude response object to a dictionary.")


def format_exception_message(exc: BaseException, seen: set[int] | None = None) -> str:
    if seen is None:
        seen = set()
    exc_id = id(exc)
    if exc_id in seen:
        return ""
    seen.add(exc_id)

    child_exceptions = getattr(exc, "exceptions", None)
    if isinstance(child_exceptions, tuple) and child_exceptions:
        child_messages = [
            format_exception_message(child, seen)
            for child in child_exceptions
            if child is not None
        ]
        child_messages = [message for message in child_messages if message]
        if child_messages:
            return "; ".join(child_messages)

    message = str(exc).strip()
    class_name = exc.__class__.__name__
    rendered = f"{class_name}: {message}" if message else class_name

    cause = getattr(exc, "__cause__", None)
    if cause is None and not getattr(exc, "__suppress_context__", False):
        cause = getattr(exc, "__context__", None)
    if cause is not None:
        cause_message = format_exception_message(cause, seen)
        if cause_message and cause_message != rendered:
            return f"{rendered}; caused by {cause_message}"
    return rendered


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
        ensure_runtime_dependencies()
        langfuse_runtime = setup_langfuse(args)
        client = make_client_from_args(args, enable_langfuse=langfuse_runtime.enabled)
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
    print(f"Using Anthropic API key from {client.api_key_env}.", file=sys.stderr)

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
                ),
                metadata=build_langfuse_observation_metadata(
                    tex_path=tex_path,
                    bib_path=bib_path,
                    mcp_url=args.mcp_url,
                    mcp_label=args.mcp_label,
                    mcp_tools=mcp_tools,
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
                    ),
                )
                if langfuse_runtime.enabled and langfuse_runtime.propagate_attributes
                else nullcontext()
            )
            with propagation_context:
                tex_upload = upload_file(client, upload_inputs.tex_path)
                bib_upload = upload_file(client, upload_inputs.bib_path)

                generation_context = (
                    langfuse_runtime.client.start_as_current_observation(
                        name="anthropic.messages.create",
                        as_type="generation",
                        input=build_langfuse_generation_input(
                            tex_path=tex_path,
                            bib_path=bib_path,
                            user_prompt=args.prompt,
                            mcp_url=args.mcp_url,
                            mcp_label=args.mcp_label,
                            mcp_tools=mcp_tools,
                        ),
                        metadata=build_langfuse_generation_metadata(
                            tex_path=tex_path,
                            bib_path=bib_path,
                            mcp_url=args.mcp_url,
                            mcp_label=args.mcp_label,
                            mcp_tools=mcp_tools,
                            pricing=pricing,
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
                        ),
                        tex_upload=tex_upload,
                        bib_upload=bib_upload,
                        mcp_label=args.mcp_label,
                        mcp_url=args.mcp_url,
                        requested_tools=mcp_tools,
                        mcp_headers=mcp_headers,
                        mcp_read_timeout=args.mcp_read_timeout,
                    )
                    if thinking_note:
                        print(thinking_note, file=sys.stderr)
                    print(
                        "MCP session tools exposed to Claude: "
                        + ", ".join(available_tool_names)
                        + ".",
                        file=sys.stderr,
                    )

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
