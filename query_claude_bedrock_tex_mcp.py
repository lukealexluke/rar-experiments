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


DEFAULT_API_KEY_ENV = ""
DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_BEDROCK_CONNECT_TIMEOUT = 30.0
DEFAULT_BEDROCK_READ_TIMEOUT = 600.0
DEFAULT_MCP_READ_TIMEOUT = 300.0
DEFAULT_CLAUDE_INPUT_COST_ENV = "CLAUDE_BEDROCK_INPUT_COST_PER_1M"
DEFAULT_CLAUDE_CACHED_INPUT_COST_ENV = "CLAUDE_BEDROCK_CACHED_INPUT_COST_PER_1M"
DEFAULT_CLAUDE_OUTPUT_COST_ENV = "CLAUDE_BEDROCK_OUTPUT_COST_PER_1M"
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class BedrockClientHandle:
    runtime_client: Any
    region_name: str
    profile_name: str | None = None
    endpoint_url: str | None = None

    def close(self) -> None:
        close = getattr(self.runtime_client, "close", None)
        if callable(close):
            close()


@dataclass(frozen=True)
class BedrockFileRef:
    path: Path
    name: str
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send a .tex file and bibliography source to Claude through Amazon "
            "Bedrock, and let the model use the TheoremSearch MCP server."
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
            "Bedrock model or inference profile ID to use. Claude Sonnet 4.5 "
            f"requires an inference profile for on-demand use. Defaults to {DEFAULT_MODEL}."
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
        "--aws-region",
        default=default_aws_region(),
        help=(
            "AWS region for the bedrock-runtime client. Defaults to AWS_REGION, "
            f"AWS_DEFAULT_REGION, or {DEFAULT_AWS_REGION}."
        ),
    )
    parser.add_argument(
        "--aws-profile",
        help="Optional AWS profile name. Defaults to AWS_PROFILE or the SDK default chain.",
    )
    parser.add_argument(
        "--bedrock-endpoint-url",
        help="Optional custom bedrock-runtime endpoint URL.",
    )
    parser.add_argument(
        "--bedrock-connect-timeout",
        type=float,
        default=DEFAULT_BEDROCK_CONNECT_TIMEOUT,
        metavar="SECONDS",
        help=(
            "Bedrock client connect timeout in seconds. "
            f"Defaults to {DEFAULT_BEDROCK_CONNECT_TIMEOUT:g}."
        ),
    )
    parser.add_argument(
        "--bedrock-read-timeout",
        type=float,
        default=DEFAULT_BEDROCK_READ_TIMEOUT,
        metavar="SECONDS",
        help=(
            "Bedrock client read timeout in seconds. "
            f"Defaults to {DEFAULT_BEDROCK_READ_TIMEOUT:g}."
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
        help="Optional path to write the final Bedrock response JSON.",
    )
    parser.add_argument(
        "--delete-uploads",
        action="store_true",
        help="Accepted for CLI parity; Bedrock Converse does not create uploaded files.",
    )
    return parser.parse_args()


def default_aws_region() -> str:
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_AWS_REGION
    )


def create_upload_temp_dir() -> Path:
    base_dir = (Path.cwd() / ".bedrock_upload_tmp").resolve()
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
    )
    cached_input_cost = coalesce_float(
        args.cached_input_cost_per_1m,
        read_optional_float_env(DEFAULT_CLAUDE_CACHED_INPUT_COST_ENV),
        input_cost,
    )
    output_cost = coalesce_float(
        args.output_cost_per_1m,
        read_optional_float_env(DEFAULT_CLAUDE_OUTPUT_COST_ENV),
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
        import boto3  # noqa: F401
        import httpx  # noqa: F401
        from mcp import ClientSession  # noqa: F401
        from mcp.client.streamable_http import streamable_http_client  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The Claude Bedrock path requires `boto3`, `httpx`, and `mcp`. "
            "Install them with `pip install boto3 httpx mcp`."
        ) from exc


def make_client_from_args(args: argparse.Namespace, *, enable_langfuse: bool):
    return make_client(
        None,
        enable_langfuse=enable_langfuse,
        region_name=args.aws_region,
        profile_name=args.aws_profile,
        endpoint_url=args.bedrock_endpoint_url,
        connect_timeout=args.bedrock_connect_timeout,
        read_timeout=args.bedrock_read_timeout,
    )


def make_client(
    api_key: str | None = None,
    *,
    enable_langfuse: bool,
    region_name: str | None = None,
    profile_name: str | None = None,
    endpoint_url: str | None = None,
    connect_timeout: float = DEFAULT_BEDROCK_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_BEDROCK_READ_TIMEOUT,
):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "The `boto3`/`botocore` packages are not installed. Install them with "
            "`pip install boto3` and rerun the script."
        ) from exc

    region = region_name or default_aws_region()
    profile = profile_name or os.environ.get("AWS_PROFILE") or None
    try:
        session = (
            boto3.Session(profile_name=profile, region_name=region)
            if profile
            else boto3.Session(region_name=region)
        )
        client_kwargs: dict[str, Any] = {
            "config": Config(
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                retries={"max_attempts": 3, "mode": "standard"},
            )
        }
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url
        runtime_client = session.client("bedrock-runtime", **client_kwargs)
    except Exception as exc:
        raise RuntimeError(f"Could not create a Bedrock Runtime client: {exc}") from exc

    return BedrockClientHandle(
        runtime_client=runtime_client,
        region_name=region,
        profile_name=profile,
        endpoint_url=endpoint_url,
    )


def upload_file(client, path: Path) -> BedrockFileRef:
    return BedrockFileRef(
        path=path.resolve(),
        name=path.name,
        text=read_candidate_text(path),
    )


def cleanup_uploaded_files(client, uploaded_file_names: list[str]) -> None:
    return None


def run_response(
    client: BedrockClientHandle,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    prompt: str,
    tex_upload: BedrockFileRef,
    bib_upload: BedrockFileRef,
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
    client: BedrockClientHandle,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    prompt: str,
    tex_upload: BedrockFileRef,
    bib_upload: BedrockFileRef,
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
            "Missing Claude Bedrock runtime dependency. Install `boto3`, `httpx`, and `mcp`."
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
                tool_specs, available_tool_names = build_bedrock_tool_specs(
                    getattr(list_tools_result, "tools", []),
                    requested_tools,
                )
                if not tool_specs:
                    raise RuntimeError(
                        "The MCP session did not expose any requested Bedrock-compatible tools."
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
                    response_data = normalize_bedrock_response(response)
                    response_message = response_data.get("output", {}).get("message")
                    if isinstance(response_message, dict):
                        messages.append(response_message)

                    tool_uses = extract_tool_uses(response_message)
                    if not tool_uses:
                        response_data["mcp_available_tools"] = available_tool_names
                        response_data["mcp_tool_rounds"] = tool_rounds
                        response_data["langfuse_tool_calls"] = mcp_tool_calls
                        return response, response_data, available_tool_names, thinking_note

                    tool_rounds += 1
                    tool_result_blocks = []
                    for tool_use in tool_uses:
                        tool_result = await call_mcp_tool_for_bedrock(
                            session=session,
                            tool_use=tool_use,
                            allowed_tool_names=set(available_tool_names),
                            mcp_read_timeout=mcp_read_timeout,
                        )
                        tool_result_blocks.append(tool_result)
                        mcp_tool_calls.append(
                            build_bedrock_langfuse_tool_call(
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
    tex_upload: BedrockFileRef,
    bib_upload: BedrockFileRef,
) -> list[dict[str, str]]:
    return [
        {"text": prompt},
        {"text": render_text_file_block("LaTeX source", tex_upload)},
        {"text": render_text_file_block("Bibliography source", bib_upload)},
    ]


def render_text_file_block(label: str, file_ref: BedrockFileRef) -> str:
    return (
        f"\n\n===== {label}: {file_ref.name} =====\n"
        f"{file_ref.text.rstrip()}\n"
        f"===== End {label}: {file_ref.name} ====="
    )


def build_bedrock_tool_specs(
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
        tool_specs.append(
            {
                "toolSpec": {
                    "name": name,
                    "description": description,
                    "inputSchema": {"json": schema},
                }
            }
        )
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
            "for Bedrock's minimum thinking budget."
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
    client: BedrockClientHandle,
    model: str,
    messages: list[dict[str, Any]],
    tool_specs: list[dict[str, Any]],
    max_output_tokens: int,
    additional_model_request_fields: dict[str, Any],
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "modelId": model,
        "messages": messages,
        "system": [{"text": build_system_prompt()}],
        "inferenceConfig": {"maxTokens": max_output_tokens},
        "toolConfig": {"tools": tool_specs},
    }
    if additional_model_request_fields:
        request["additionalModelRequestFields"] = additional_model_request_fields
    try:
        return client.runtime_client.converse(**request)
    except Exception as exc:
        raise RuntimeError(
            "Bedrock Converse operation failed: " + format_exception_message(exc)
        ) from exc


def normalize_bedrock_response(response: dict[str, Any]) -> dict[str, Any]:
    response_data = to_dict(response)
    stop_reason = str(response_data.get("stopReason") or "").strip()
    if stop_reason == "end_turn":
        response_data["status"] = "completed"
    elif stop_reason:
        response_data["status"] = stop_reason
    return response_data


def extract_tool_uses(response_message: Any) -> list[dict[str, Any]]:
    if not isinstance(response_message, dict):
        return []
    content = response_message.get("content")
    if not isinstance(content, list):
        return []
    tool_uses: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("toolUse"), dict):
            tool_uses.append(block["toolUse"])
    return tool_uses


async def call_mcp_tool_for_bedrock(
    *,
    session: Any,
    tool_use: dict[str, Any],
    allowed_tool_names: set[str],
    mcp_read_timeout: float,
) -> dict[str, Any]:
    tool_use_id = str(tool_use.get("toolUseId") or "")
    tool_name = str(tool_use.get("name") or "")
    tool_input = tool_use.get("input")
    if not isinstance(tool_input, dict):
        tool_input = {}

    if not tool_use_id:
        return build_tool_result_block(
            tool_use_id="missing-tool-use-id",
            payload={"error": "Bedrock did not provide a toolUseId."},
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
    return {
        "toolResult": {
            "toolUseId": tool_use_id,
            "content": [{"json": payload}],
            "status": status,
        }
    }


def build_bedrock_langfuse_tool_call(
    *,
    tool_use: dict[str, Any],
    tool_result_block: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    tool_result = tool_result_block.get("toolResult")
    if not isinstance(tool_result, dict):
        tool_result = {}
    return {
        "provider": "claude_bedrock",
        "tool_type": "mcp",
        "name": tool_use.get("name"),
        "id": tool_use.get("toolUseId"),
        "status": tool_result.get("status"),
        "input": tool_use.get("input"),
        "output": extract_bedrock_tool_result_payload(tool_result),
        "raw_type": "toolUse",
        "round": round_index,
    }


def extract_bedrock_tool_result_payload(tool_result: dict[str, Any]) -> Any:
    content = tool_result.get("content")
    if not isinstance(content, list) or not content:
        return None
    if len(content) == 1 and isinstance(content[0], dict):
        block = content[0]
        if "json" in block:
            return block["json"]
        if "text" in block:
            return block["text"]
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
    usage = response_data.get("usage")
    if not isinstance(usage, dict):
        return {}

    input_tokens = read_usage_int(usage, "inputTokens")
    output_tokens = read_usage_int(usage, "outputTokens")
    total_tokens = read_usage_int(usage, "totalTokens")
    cache_read_tokens = read_usage_int(usage, "cacheReadInputTokens")
    cache_write_tokens = read_usage_int(usage, "cacheWriteInputTokens")

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
    raise RuntimeError("Could not convert the Bedrock response object to a dictionary.")


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
    print(
        f"Using Amazon Bedrock region {client.region_name}"
        + (f" with profile {client.profile_name}" if client.profile_name else "")
        + ".",
        file=sys.stderr,
    )

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
                        name="bedrock-runtime.converse",
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
