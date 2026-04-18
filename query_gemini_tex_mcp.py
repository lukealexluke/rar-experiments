#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from query_openai_tex_mcp import (
    DEFAULT_ALLOWED_MCP_TOOLS,
    DEFAULT_APPROVAL_LIMIT,
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
    flush_langfuse,
    load_environment_from_dotenv,
    parse_mcp_headers,
    prepare_upload_inputs,
    read_api_key,
    read_optional_float_env,
    safe_serialize_value,
    setup_langfuse,
    validate_input_path,
    write_json_output,
    write_text_output,
)


DEFAULT_API_KEY_ENV = "GOOGLE_API_KEY"
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_GEMINI_INPUT_COST_ENV = "GEMINI_INPUT_COST_PER_1M"
DEFAULT_GEMINI_CACHED_INPUT_COST_ENV = "GEMINI_CACHED_INPUT_COST_PER_1M"
DEFAULT_GEMINI_OUTPUT_COST_ENV = "GEMINI_OUTPUT_COST_PER_1M"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a .tex file and a bibliography source to the Gemini API, "
            "and let the model use the TheoremSearch MCP server."
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
            "truncated at the selected logged line, the logged citation is masked as "
            "[Citation Needed], and the logged bibliography entry is removed from the "
            "uploaded bibliography text."
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
        "--prompt",
        help=(
            "The question or instruction to send with the uploaded files. "
            "If omitted, the script uses a default paper-analysis prompt."
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
    except ImportError as exc:
        raise RuntimeError(
            "The Gemini path requires the `google-genai` package. "
            "Install it with `pip install google-genai`."
        ) from exc


def make_client(api_key: str, *, enable_langfuse: bool):
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "The `google-genai` package is not installed. Install it with "
            "`pip install google-genai` and rerun the script."
        ) from exc
    return genai.Client(api_key=api_key)


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
):
    try:
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Missing Gemini runtime dependency. Install `google-genai`."
        ) from exc

    thinking_config, thinking_note = build_thinking_config(model, reasoning_effort)
    config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(),
        max_output_tokens=max_output_tokens,
        thinking_config=thinking_config,
        tools=[
            types.Tool(
                mcp_servers=[
                    types.McpServer(
                        name=mcp_label,
                        streamable_http_transport=types.StreamableHttpTransport(
                            url=mcp_url,
                            headers=mcp_headers or None,
                        ),
                    )
                ]
            )
        ],
    )
    response = client.models.generate_content(
        model=model,
        contents=[prompt, tex_upload, bib_upload],
        config=config,
    )
    return response, to_dict(response), requested_tools, thinking_note


def extract_output_text(response, response_data: dict[str, Any]) -> str:
    output_text = getattr(response, "text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text_chunks: list[str] = []
    for candidate in response_data.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        content = candidate.get("content")
        if not isinstance(content, dict):
            continue
        for part in content.get("parts", []):
            if not isinstance(part, dict):
                continue
            if part.get("thought"):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                text_chunks.append(text.strip())
    return "\n".join(text_chunks)


def extract_langfuse_usage_details(response, response_data: dict[str, Any]) -> dict[str, int]:
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
                    )
                    if thinking_note:
                        print(thinking_note, file=sys.stderr)
                    print(
                        "Requested MCP tools: "
                        + ", ".join(available_tool_names)
                        + " (Gemini server-side MCP does not enforce client-side tool allowlists).",
                        file=sys.stderr,
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
