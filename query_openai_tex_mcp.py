#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_API_KEY_ENV = "openai_key"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 4000
DEFAULT_MCP_LABEL = "theoremsearch"
DEFAULT_MCP_URL = "https://api.theoremsearch.com/mcp"
DEFAULT_ALLOWED_MCP_TOOLS = ("theorem_search",)
DEFAULT_APPROVAL_LIMIT = 8
DEFAULT_DOTENV_FILENAME = ".env"
DEFAULT_LANGFUSE_TRACE_NAME = "tex-bib-mcp-query"
DEFAULT_LANGFUSE_PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
DEFAULT_LANGFUSE_SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
DEFAULT_OPENAI_INPUT_COST_ENV = "OPENAI_INPUT_COST_PER_1M"
DEFAULT_OPENAI_CACHED_INPUT_COST_ENV = "OPENAI_CACHED_INPUT_COST_PER_1M"
DEFAULT_OPENAI_OUTPUT_COST_ENV = "OPENAI_OUTPUT_COST_PER_1M"


@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_1m: float
    cached_input_cost_per_1m: float
    output_cost_per_1m: float
    source: str


DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-5.4": ModelPricing(
        input_cost_per_1m=2.50,
        cached_input_cost_per_1m=0.25,
        output_cost_per_1m=15.00,
        source="OpenAI API pricing as of 2026-04-13",
    ),
    "gpt-5.4-mini": ModelPricing(
        input_cost_per_1m=0.750,
        cached_input_cost_per_1m=0.075,
        output_cost_per_1m=4.500,
        source="OpenAI API pricing as of 2026-04-13",
    ),
    "gpt-5.4-nano": ModelPricing(
        input_cost_per_1m=0.20,
        cached_input_cost_per_1m=0.02,
        output_cost_per_1m=1.25,
        source="OpenAI API pricing as of 2026-04-13",
    ),
}


@dataclass
class LangfuseRuntime:
    enabled: bool
    client: Any | None = None
    propagate_attributes: Any | None = None
    status_message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a .tex file and a .txt bib file to the OpenAI Responses API, "
            "and let the model use the TheoremSearch MCP server."
        )
    )
    parser.add_argument("tex_file", type=Path, help="Path to the LaTeX source file.")
    parser.add_argument("bib_file", type=Path, help="Path to the BibTeX file.")
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
        help=f"OpenAI model to use. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        default=DEFAULT_REASONING_EFFORT,
        help=(
            "Reasoning effort for supported models. "
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
            "Environment variable containing your OpenAI API key. "
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
        help="Delete the uploaded OpenAI files after the response completes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        dotenv_path = load_environment_from_dotenv()
        tex_path = validate_input_path(args.tex_file, expected_suffix=".tex")
        bib_path = validate_input_path(args.bib_file, expected_suffix=".txt")
        api_key = read_api_key(args.api_key_env)
        langfuse_runtime = setup_langfuse(args)
        client = make_client(api_key, enable_langfuse=langfuse_runtime.enabled)
        mcp_headers = parse_mcp_headers(args.mcp_headers or [])
        pricing = resolve_model_pricing(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if dotenv_path:
        print(f"Loaded environment from {dotenv_path}", file=sys.stderr)
    if langfuse_runtime.status_message:
        print(langfuse_runtime.status_message, file=sys.stderr)

    uploaded_file_ids: list[str] = []
    trace_url: str | None = None
    try:
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
                    mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
                ),
                metadata=build_langfuse_observation_metadata(
                    tex_path=tex_path,
                    bib_path=bib_path,
                    mcp_url=args.mcp_url,
                    mcp_label=args.mcp_label,
                    mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
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
                tex_upload = upload_file(client, tex_path)
                bib_upload = upload_file(client, bib_path)
                uploaded_file_ids.extend(
                    [
                        getattr(tex_upload, "id", ""),
                        getattr(bib_upload, "id", ""),
                    ]
                )

                generation_context = (
                    langfuse_runtime.client.start_as_current_observation(
                        name="openai.responses.create",
                        as_type="generation",
                        input=build_langfuse_generation_input(
                            tex_path=tex_path,
                            bib_path=bib_path,
                            user_prompt=args.prompt,
                            mcp_url=args.mcp_url,
                            mcp_label=args.mcp_label,
                            mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
                        ),
                        metadata=build_langfuse_generation_metadata(
                            tex_path=tex_path,
                            bib_path=bib_path,
                            mcp_url=args.mcp_url,
                            mcp_label=args.mcp_label,
                            mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
                            pricing=pricing,
                        ),
                        model=args.model,
                        model_parameters=build_langfuse_model_parameters(args),
                    )
                    if langfuse_runtime.enabled
                    else nullcontext(None)
                )
                with generation_context as generation_observation:
                    response, response_data = run_response(
                        client=client,
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        max_output_tokens=args.max_output_tokens,
                        prompt=build_user_prompt(
                            tex_path=tex_path,
                            bib_path=bib_path,
                            user_prompt=args.prompt,
                        ),
                        tex_upload_id=required_file_id(tex_upload, tex_path),
                        bib_upload_id=required_file_id(bib_upload, bib_path),
                        mcp_tool=build_mcp_tool(
                            mcp_label=args.mcp_label,
                            mcp_url=args.mcp_url,
                            mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
                            mcp_headers=mcp_headers,
                        ),
                    )

                    output_text = extract_output_text(response, response_data)
                    usage_details = extract_langfuse_usage_details(response, response_data)
                    cost_details = compute_langfuse_cost_details(usage_details, pricing)

                    if generation_observation is not None:
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
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.delete_uploads:
            cleanup_uploaded_files(client if "client" in locals() else None, uploaded_file_ids)
        if "langfuse_runtime" in locals() and langfuse_runtime.enabled:
            flush_langfuse(langfuse_runtime)

    if trace_url:
        print(f"Langfuse trace: {trace_url}", file=sys.stderr)

    return 0


def validate_input_path(path: Path, expected_suffix: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise RuntimeError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise RuntimeError(f"Expected a file, but found: {resolved}")
    if resolved.suffix.lower() != expected_suffix:
        raise RuntimeError(
            f"Expected a {expected_suffix} file, but got: {resolved.name}"
        )
    return resolved


def read_api_key(api_key_env: str) -> str:
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Environment variable {api_key_env} is not set or is empty."
        )
    return api_key


def load_environment_from_dotenv() -> Path | None:
    dotenv_path = find_local_dotenv()
    if dotenv_path is None:
        return None

    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "A .env file was found, but `python-dotenv` is not installed. "
            "Install it with `pip install python-dotenv`."
        ) from exc

    load_dotenv(dotenv_path=dotenv_path, override=False)
    return dotenv_path


def find_local_dotenv() -> Path | None:
    candidates = [
        Path.cwd() / DEFAULT_DOTENV_FILENAME,
        Path(__file__).resolve().with_name(DEFAULT_DOTENV_FILENAME),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def setup_langfuse(args: argparse.Namespace) -> LangfuseRuntime:
    if args.disable_langfuse:
        return LangfuseRuntime(
            enabled=False,
            status_message="Langfuse tracing disabled via --disable-langfuse.",
        )

    public_key = os.environ.get(DEFAULT_LANGFUSE_PUBLIC_KEY_ENV, "").strip()
    secret_key = os.environ.get(DEFAULT_LANGFUSE_SECRET_KEY_ENV, "").strip()
    if not public_key or not secret_key:
        return LangfuseRuntime(
            enabled=False,
            status_message=(
                "Langfuse tracing is off. Set LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY to enable it."
            ),
        )

    try:
        from langfuse import get_client, propagate_attributes
    except ImportError:
        return LangfuseRuntime(
            enabled=False,
            status_message=(
                "Langfuse tracing is off because the `langfuse` package is not "
                "installed. Install it with `pip install langfuse`."
            ),
        )

    client = get_client()
    return LangfuseRuntime(
        enabled=True,
        client=client,
        propagate_attributes=propagate_attributes,
        status_message="Langfuse tracing is enabled in manual trace mode.",
    )


def make_client(api_key: str, *, enable_langfuse: bool):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The `openai` package is not installed. Install it with "
            "`pip install openai` and rerun the script."
        ) from exc
    return OpenAI(api_key=api_key)


def parse_mcp_headers(items: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise RuntimeError(
                f"Invalid --mcp-header value {item!r}. Use KEY=VALUE."
            )
        headers[key.strip()] = value.strip()
    return headers


def upload_file(client, path: Path):
    with path.open("rb") as handle:
        return client.files.create(file=handle, purpose="user_data")


def required_file_id(uploaded_file, path: Path) -> str:
    file_id = getattr(uploaded_file, "id", "")
    if not file_id:
        raise RuntimeError(f"OpenAI did not return a file ID for {path.name}.")
    return file_id


def build_user_prompt(tex_path: Path, bib_path: Path, user_prompt: str | None) -> str:
    instruction = user_prompt.strip() if user_prompt else (
        "Read the attached LaTeX source and bibliography, then answer "
        "helpfully. Use theorem_search when it would improve factual or "
        "literature-grounded answers."
    )
    return (
        "You are given two uploaded files.\n"
        f"- {tex_path.name}: the LaTeX source file\n"
        f"- {bib_path.name}: the bibliography file\n\n"
        "Treat the uploaded files as the primary project context.\n"
        "Your job is to search the database for citations via the search tool. You can call it through the attached MCP server.\n\n"
        "User request:\n"
        f"{instruction}"
    )


def build_system_prompt() -> str:
    return (
        "You do not have access to web search in this session. "
        "The only external tool available is the attached MCP server, "
        "and the only allowed MCP tool is theorem_search. "
        "If you need external retrieval, use theorem_search only."
    )


def build_langfuse_observation_input(
    tex_path: Path,
    bib_path: Path,
    user_prompt: str | None,
    model: str,
    reasoning_effort: str,
    mcp_url: str,
    mcp_label: str,
    mcp_tools: list[str],
) -> dict[str, Any]:
    return {
        "tex_file": str(tex_path),
        "bib_file": str(bib_path),
        "user_prompt": user_prompt or "(default prompt)",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "web_search_enabled": False,
        "mcp": {
            "label": mcp_label,
            "url": mcp_url,
            "allowed_tools": mcp_tools,
        },
    }


def build_langfuse_observation_metadata(
    tex_path: Path,
    bib_path: Path,
    mcp_url: str,
    mcp_label: str,
    mcp_tools: list[str],
) -> dict[str, Any]:
    return {
        "tex_filename": tex_path.name,
        "bib_filename": bib_path.name,
        "tool_policy": {
            "web_search_enabled": False,
            "mcp_only": True,
        },
        "mcp": {
            "label": mcp_label,
            "url": mcp_url,
            "allowed_tools": mcp_tools,
        },
    }


def build_langfuse_trace_metadata(
    tex_path: Path,
    bib_path: Path,
    model: str,
) -> dict[str, str]:
    return {
        "tex_file": tex_path.name,
        "bib_file": bib_path.name,
        "model": model,
        "web_search": "disabled",
        "external_tool": "theorem_search",
    }


def build_langfuse_observation_output(
    response_data: dict[str, Any],
    output_text: str,
    usage_details: dict[str, int],
    cost_details: dict[str, float],
) -> dict[str, Any]:
    return {
        "response_id": response_data.get("id"),
        "has_text_output": bool(output_text),
        "output_text": output_text or None,
        "mcp_approval_requests": len(extract_mcp_approval_requests(response_data)),
        "usage_details": usage_details or None,
        "cost_details": cost_details or None,
    }


def build_langfuse_model_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "tool_mode": "mcp_only",
    }


def build_langfuse_generation_input(
    tex_path: Path,
    bib_path: Path,
    user_prompt: str | None,
    mcp_url: str,
    mcp_label: str,
    mcp_tools: list[str],
) -> dict[str, Any]:
    return {
        "tex_file": str(tex_path),
        "bib_file": str(bib_path),
        "user_prompt": user_prompt or "(default prompt)",
        "tools": {
            "web_search_enabled": False,
            "mcp_label": mcp_label,
            "mcp_url": mcp_url,
            "allowed_tools": mcp_tools,
        },
    }


def build_langfuse_generation_metadata(
    tex_path: Path,
    bib_path: Path,
    mcp_url: str,
    mcp_label: str,
    mcp_tools: list[str],
    pricing: ModelPricing | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "tex_filename": tex_path.name,
        "bib_filename": bib_path.name,
        "tool_policy": {
            "web_search_enabled": False,
            "mcp_only": True,
        },
        "mcp": {
            "label": mcp_label,
            "url": mcp_url,
            "allowed_tools": mcp_tools,
        },
    }
    if pricing is not None:
        metadata["pricing"] = {
            "source": pricing.source,
            "input_cost_per_1m": pricing.input_cost_per_1m,
            "cached_input_cost_per_1m": pricing.cached_input_cost_per_1m,
            "output_cost_per_1m": pricing.output_cost_per_1m,
        }
    return metadata


def build_langfuse_generation_output(
    response_data: dict[str, Any],
    output_text: str,
    usage_details: dict[str, int],
    cost_details: dict[str, float],
) -> dict[str, Any]:
    return {
        "response_id": response_data.get("id"),
        "status": response_data.get("status"),
        "output_text": output_text or None,
        "usage_details": usage_details or None,
        "cost_details": cost_details or None,
    }


def build_langfuse_generation_result_metadata(
    pricing: ModelPricing | None,
    usage_details: dict[str, int],
    cost_details: dict[str, float],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "usage_tracked": bool(usage_details),
        "cost_tracked": bool(cost_details),
    }
    if pricing is not None:
        metadata["pricing_source"] = pricing.source
    return metadata


def resolve_model_pricing(args: argparse.Namespace) -> ModelPricing | None:
    model_key = args.model.strip().lower()
    default_pricing = DEFAULT_MODEL_PRICING.get(model_key)

    input_cost = coalesce_float(
        args.input_cost_per_1m,
        read_optional_float_env(DEFAULT_OPENAI_INPUT_COST_ENV),
        default_pricing.input_cost_per_1m if default_pricing else None,
    )
    cached_input_cost = coalesce_float(
        args.cached_input_cost_per_1m,
        read_optional_float_env(DEFAULT_OPENAI_CACHED_INPUT_COST_ENV),
        default_pricing.cached_input_cost_per_1m if default_pricing else None,
    )
    output_cost = coalesce_float(
        args.output_cost_per_1m,
        read_optional_float_env(DEFAULT_OPENAI_OUTPUT_COST_ENV),
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
            DEFAULT_OPENAI_INPUT_COST_ENV,
            DEFAULT_OPENAI_CACHED_INPUT_COST_ENV,
            DEFAULT_OPENAI_OUTPUT_COST_ENV,
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


def coalesce_float(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def read_optional_float_env(env_name: str) -> float | None:
    raw_value = os.environ.get(env_name, "").strip()
    if not raw_value:
        return None
    try:
        return float(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"Environment variable {env_name} must be a number, got {raw_value!r}."
        ) from exc


def extract_langfuse_usage_details(response, response_data: dict[str, Any]) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        usage = response_data.get("usage")
    if usage is None:
        return {}

    input_tokens = read_usage_int(usage, "input_tokens")
    output_tokens = read_usage_int(usage, "output_tokens")
    total_tokens = read_usage_int(usage, "total_tokens")

    input_tokens_details = read_usage_object(usage, "input_tokens_details")
    output_tokens_details = read_usage_object(usage, "output_tokens_details")

    cached_tokens = read_usage_int(input_tokens_details, "cached_tokens")
    reasoning_tokens = read_usage_int(output_tokens_details, "reasoning_tokens")

    usage_details: dict[str, int] = {}
    if input_tokens is not None:
        usage_details["input"] = input_tokens
    if output_tokens is not None:
        usage_details["output"] = output_tokens
    if total_tokens is not None:
        usage_details["total"] = total_tokens
    if cached_tokens is not None:
        usage_details["cached_input"] = cached_tokens
        usage_details["uncached_input"] = max(input_tokens - cached_tokens, 0) if input_tokens is not None else 0
    if reasoning_tokens is not None:
        usage_details["reasoning"] = reasoning_tokens
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


def compute_langfuse_cost_details(
    usage_details: dict[str, int],
    pricing: ModelPricing | None,
) -> dict[str, float]:
    if pricing is None or not usage_details:
        return {}

    input_tokens = usage_details.get("input", 0)
    cached_tokens = usage_details.get("cached_input", 0)
    uncached_input_tokens = max(input_tokens - cached_tokens, 0)
    output_tokens = usage_details.get("output", 0)

    input_cost = (uncached_input_tokens / 1_000_000) * pricing.input_cost_per_1m
    cached_input_cost = (cached_tokens / 1_000_000) * pricing.cached_input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_1m
    total_cost = input_cost + cached_input_cost + output_cost

    return {
        "input": round(input_cost, 10),
        "cached_input": round(cached_input_cost, 10),
        "output": round(output_cost, 10),
        "total": round(total_cost, 10),
    }


def build_mcp_tool(
    mcp_label: str,
    mcp_url: str,
    mcp_tools: list[str],
    mcp_headers: dict[str, str],
) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "type": "mcp",
        "server_label": mcp_label,
        "server_url": mcp_url,
        "server_description": "TheoremSearch MCP for semantic theorem retrieval.",
        "allowed_tools": mcp_tools,
        "require_approval": "never",
    }
    if mcp_headers:
        tool["headers"] = mcp_headers
    return tool


def run_response(
    client,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    prompt: str,
    tex_upload_id: str,
    bib_upload_id: str,
    mcp_tool: dict[str, Any],
):
    response = client.responses.create(
        model=model,
        reasoning={"effort": reasoning_effort},
        max_output_tokens=max_output_tokens,
        tools=[mcp_tool],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_system_prompt(),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_file",
                        "file_id": tex_upload_id,
                    },
                    {
                        "type": "input_file",
                        "file_id": bib_upload_id,
                    },
                ],
            }
        ],
    )

    response_data = to_dict(response)
    previous_response_id = response_data.get("id")
    if not previous_response_id:
        return response, response_data

    for _ in range(DEFAULT_APPROVAL_LIMIT):
        approval_requests = extract_mcp_approval_requests(response_data)
        if not approval_requests:
            return response, response_data

        response = client.responses.create(
            model=model,
            reasoning={"effort": reasoning_effort},
            max_output_tokens=max_output_tokens,
            tools=[mcp_tool],
            previous_response_id=previous_response_id,
            input=[
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": request["id"],
                    "approve": True,
                    "reason": "Approved by local CLI script.",
                }
                for request in approval_requests
            ],
        )
        response_data = to_dict(response)
        previous_response_id = response_data.get("id", previous_response_id)

    raise RuntimeError(
        "Too many MCP approval rounds. Try rerunning with a simpler prompt."
    )


def extract_mcp_approval_requests(response_data: dict[str, Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for item in response_data.get("output", []):
        if isinstance(item, dict) and item.get("type") == "mcp_approval_request":
            requests.append(item)
    return requests


def extract_output_text(response, response_data: dict[str, Any]) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text_chunks: list[str] = []
    for item in response_data.get("output", []):
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for content_item in item.get("content", []):
            if (
                isinstance(content_item, dict)
                and content_item.get("type") == "output_text"
                and isinstance(content_item.get("text"), str)
            ):
                text_chunks.append(content_item["text"])
    return "\n".join(chunk.strip() for chunk in text_chunks if chunk.strip())


def to_dict(response) -> dict[str, Any]:
    data = safe_serialize_value(response)
    if isinstance(data, dict):
        return data
    raise RuntimeError("Could not convert the OpenAI response object to a dictionary.")


def safe_serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {
            str(key): safe_serialize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [safe_serialize_value(item) for item in value]

    field_map = getattr(value.__class__, "model_fields", None)
    if isinstance(field_map, dict) and field_map:
        serialized: dict[str, Any] = {}
        for field_name in field_map:
            try:
                field_value = getattr(value, field_name)
            except Exception:
                continue
            serialized[str(field_name)] = safe_serialize_value(field_value)
        if serialized:
            return serialized

    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        visible_items = {
            str(key): safe_serialize_value(item)
            for key, item in data.items()
            if not str(key).startswith("_")
        }
        if visible_items:
            return visible_items

    return repr(value)


def write_text_output(output_path: Path, output_text: str, response_data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_text:
        output_path.write_text(output_text + "\n", encoding="utf-8")
        return
    output_path.write_text(
        json.dumps(response_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_json_output(output_path: Path, response_data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(response_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def cleanup_uploaded_files(client, uploaded_file_ids: list[str]) -> None:
    if client is None:
        return
    for file_id in uploaded_file_ids:
        if not file_id:
            continue
        try:
            client.files.delete(file_id)
        except Exception:
            pass


def flush_langfuse(langfuse_runtime: LangfuseRuntime) -> None:
    try:
        langfuse_runtime.client.flush()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
