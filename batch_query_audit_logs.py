#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import re
import sys
from pathlib import Path
from typing import Any

from query_openai_tex_mcp import (
    DEFAULT_ALLOWED_MCP_TOOLS,
    DEFAULT_API_KEY_ENV,
    DEFAULT_LANGFUSE_TRACE_NAME,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MCP_LABEL,
    DEFAULT_MCP_URL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    build_mcp_tool,
    build_langfuse_generation_output,
    build_langfuse_generation_result_metadata,
    build_langfuse_model_parameters,
    cleanup_uploaded_files,
    cleanup_upload_temp_dir,
    compute_langfuse_cost_details,
    create_upload_temp_dir,
    extract_output_text,
    extract_langfuse_usage_details,
    flush_langfuse,
    load_environment_from_dotenv,
    make_client,
    parse_audit_log,
    parse_mcp_headers,
    prepare_upload_inputs,
    read_api_key,
    required_file_id,
    resolve_model_pricing,
    run_response,
    setup_langfuse,
    upload_file,
)

CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
DEFAULT_LOGS_DIR = Path("statement_reference_audit_logs")
FALLBACK_LOGS_DIRS = (
    Path("random_arxiv_source_samples") / "statement_reference_audit_logs",
)
DEFAULT_BATCH_LANGFUSE_TRACE_NAME = f"{DEFAULT_LANGFUSE_TRACE_NAME}-batch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run query_openai_tex_mcp-style GPT queries over every audit-log entry in a "
            "statement_reference_audit_logs directory and write one JSON result per line."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help=(
            "Directory containing statement audit logs. Defaults to "
            "statement_reference_audit_logs and falls back to "
            "random_arxiv_source_samples/statement_reference_audit_logs when present."
        ),
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        help=(
            "Root directory used to relocate TeX/bibliography files when absolute paths "
            "stored in the logs no longer exist. Defaults to the parent of --logs-dir."
        ),
    )
    parser.add_argument(
        "--output-log",
        type=Path,
        help="Output JSONL log path. Defaults to <logs-dir>/openai_query_results.jsonl.",
    )
    parser.add_argument(
        "--raw-response-dir",
        type=Path,
        help=(
            "Optional directory where raw OpenAI response JSON files are written for "
            "items that do not complete cleanly."
        ),
    )
    parser.add_argument(
        "--prompt",
        help=(
            "Optional custom task instruction appended after the audit-log context. "
            "If omitted, the script asks for the cited theorem-like result name/number "
            "and the external arXiv identifier."
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
        help=f"Reasoning effort. Defaults to {DEFAULT_REASONING_EFFORT}.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Maximum output tokens. Defaults to {DEFAULT_MAX_OUTPUT_TOKENS}.",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=f"Environment variable containing the OpenAI API key. Defaults to {DEFAULT_API_KEY_ENV}.",
    )
    parser.add_argument(
        "--mcp-url",
        default=DEFAULT_MCP_URL,
        help=f"MCP server URL. Defaults to {DEFAULT_MCP_URL}.",
    )
    parser.add_argument(
        "--mcp-label",
        default=DEFAULT_MCP_LABEL,
        help=f"MCP label. Defaults to {DEFAULT_MCP_LABEL}.",
    )
    parser.add_argument(
        "--mcp-tool",
        dest="mcp_tools",
        action="append",
        help="Allowed MCP tool name. Repeat to allow multiple tools.",
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
        default=DEFAULT_BATCH_LANGFUSE_TRACE_NAME,
        help=(
            "Langfuse trace name when tracing is enabled. "
            f"Defaults to {DEFAULT_BATCH_LANGFUSE_TRACE_NAME}."
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
    parser.add_argument(
        "--delete-uploads",
        action="store_true",
        help="Delete uploaded OpenAI files after each query finishes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output JSONL if it already exists.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Optional cap on the total number of audit-log entries to query.",
    )
    return parser.parse_args()


def sorted_log_paths(logs_dir: Path) -> list[Path]:
    return sorted(logs_dir.glob("*.log"), key=lambda path: path.name)


def resolve_logs_dir(requested_logs_dir: Path) -> Path:
    candidates = [requested_logs_dir]
    if requested_logs_dir == DEFAULT_LOGS_DIR:
        candidates.extend(FALLBACK_LOGS_DIRS)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists() and resolved.is_dir():
            return resolved

    rendered = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise RuntimeError(f"logs directory does not exist. Tried: {rendered}")


def resolve_current_tex_path(logged_tex_path: Path, search_root: Path) -> Path:
    if logged_tex_path.exists():
        return logged_tex_path.resolve()

    direct_candidate = (search_root / logged_tex_path.parent.name / logged_tex_path.name).resolve()
    if direct_candidate.exists():
        return direct_candidate

    matches = [
        candidate.resolve()
        for candidate in search_root.rglob(logged_tex_path.name)
        if candidate.parent.name == logged_tex_path.parent.name
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple TeX files match logged path {logged_tex_path.name}: {rendered}"
        )
    raise RuntimeError(f"Could not relocate logged TeX file: {logged_tex_path}")


def resolve_fallback_bib_path(logged_bib_paths: tuple[Path, ...], tex_path: Path) -> Path:
    for logged_bib_path in logged_bib_paths:
        if logged_bib_path.exists():
            return logged_bib_path.resolve()
        same_name = (tex_path.parent / logged_bib_path.name).resolve()
        if same_name.exists():
            return same_name

    for pattern in ("*.bib", "*.bbl", "*.txt"):
        matches = sorted(path.resolve() for path in tex_path.parent.glob(pattern))
        if matches:
            return matches[0]

    raise RuntimeError(f"Could not find a bibliography file near {tex_path}")


def build_item_prompt(log_name: str, record_index: int, line_number: int, user_prompt: str | None) -> str:
    default_instruction = (
        "Task:\n"
        '1. Identify the cited theorem-like result name/number, for example "Theorem 1.2" or "Proposition 3.4".\n'
        "2. Identify the external arXiv identifier of the cited source paper containing that result.\n"
        "Use theorem_search if it helps.\n\n"
        "Return JSON only, with exactly this shape:\n"
        '{"gpt_name": <string or null>, "gpt_ext_source": <string or null>}'
    )
    instruction = user_prompt.strip() if user_prompt else default_instruction
    return (
        "Analyze one logged theorem-style citation occurrence in the attached LaTeX source and bibliography.\n\n"
        f"Log file: {log_name}\n"
        f"Log item: {record_index}\n"
        f"Logged source line number: {line_number}\n\n"
        "The attached LaTeX source has been truncated at the logged line.\n"
        "The relevant citation on that line has been masked as [Citation Needed].\n"
        "The matching bibliography entry has been removed from the uploaded bibliography text.\n\n"
        f"{instruction}"
    )


def parse_gpt_response(output_text: str) -> tuple[str, str]:
    text = output_text.strip()
    if not text:
        return "", ""

    fenced_match = CODE_FENCE_RE.match(text)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return "", ""
    if not isinstance(data, dict):
        return "", ""

    gpt_name = normalize_optional_string(data.get("gpt_name") or data.get("name"))
    gpt_ext_source = normalize_optional_string(
        data.get("gpt_ext_source") or data.get("ext_source")
    )
    return gpt_name, gpt_ext_source


def normalize_optional_string(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"null", "none", "unknown"}:
        return ""
    return cleaned


def make_raw_response_path(
    raw_response_dir: Path,
    log_path: Path,
    record_index: int,
) -> Path:
    return raw_response_dir / f"{log_path.stem}_item_{record_index}.json"


def build_langfuse_batch_span_input(
    log_path: Path,
    record_index: int,
    line_number: int,
    tex_path: Path,
    prompt: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    mcp_url: str,
    mcp_label: str,
    mcp_tools: list[str],
) -> dict[str, Any]:
    return {
        "log_file": str(log_path),
        "record_index": record_index,
        "line_number": line_number,
        "tex_file": str(tex_path),
        "user_prompt": prompt,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "mcp": {
            "label": mcp_label,
            "url": mcp_url,
            "allowed_tools": mcp_tools,
        },
    }


def build_langfuse_batch_span_metadata(
    log_path: Path,
    record_index: int,
    line_number: int,
    output_log: Path,
) -> dict[str, Any]:
    return {
        "log_file": str(log_path),
        "record_index": record_index,
        "line_number": line_number,
        "output_log": str(output_log),
        "tool_policy": {
            "web_search_enabled": False,
            "mcp_only": True,
        },
    }


def build_langfuse_batch_span_output(
    response_data: dict[str, Any],
    output_text: str,
    usage_details: dict[str, int],
    cost_details: dict[str, float],
    result_status: str,
) -> dict[str, Any]:
    return {
        "response_id": response_data.get("id"),
        "status": result_status,
        "response_status": response_data.get("status"),
        "output_text": output_text or None,
        "usage_details": usage_details or None,
        "cost_details": cost_details or None,
    }


def main() -> int:
    args = parse_args()
    try:
        logs_dir = resolve_logs_dir(args.logs_dir)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.max_items is not None and args.max_items <= 0:
        print("Error: max-items must be positive when provided.", file=sys.stderr)
        return 1

    search_root = (args.search_root or logs_dir.parent).resolve()
    output_log = (args.output_log or (logs_dir / "openai_query_results.jsonl")).resolve()
    if output_log.exists() and not args.overwrite:
        print(
            f"Error: output log already exists: {output_log}. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 1
    raw_response_dir = (
        (args.raw_response_dir or (output_log.parent / "raw_openai_responses")).resolve()
    )

    try:
        dotenv_path = load_environment_from_dotenv()
        api_key = read_api_key(args.api_key_env)
        langfuse_runtime = setup_langfuse(args)
        pricing = resolve_model_pricing(args)
        client = make_client(api_key, enable_langfuse=langfuse_runtime.enabled)
        mcp_headers = parse_mcp_headers(args.mcp_headers or [])
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if dotenv_path:
        print(f"Loaded environment from {dotenv_path}", file=sys.stderr)
    if langfuse_runtime.status_message:
        print(langfuse_runtime.status_message, file=sys.stderr)

    mcp_tool = build_mcp_tool(
        mcp_label=args.mcp_label,
        mcp_url=args.mcp_url,
        mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
        mcp_headers=mcp_headers,
    )

    log_paths = sorted_log_paths(logs_dir)
    if not log_paths:
        print(f"Error: no log files found in {logs_dir}", file=sys.stderr)
        return 1

    output_log.parent.mkdir(parents=True, exist_ok=True)
    raw_response_dir.mkdir(parents=True, exist_ok=True)
    processed_items = 0
    try:
        with output_log.open("w", encoding="utf-8") as handle:
            for log_path in log_paths:
                if args.max_items is not None and processed_items >= args.max_items:
                    break

                try:
                    audit_context = parse_audit_log(log_path)
                    tex_path = resolve_current_tex_path(audit_context.tex_path, search_root)
                    fallback_bib_path = resolve_fallback_bib_path(
                        audit_context.bibliography_paths,
                        tex_path,
                    )
                except RuntimeError as exc:
                    print(f"Warning: skipping {log_path.name}: {exc}", file=sys.stderr)
                    continue

                if not audit_context.records:
                    print(f"Skipping {log_path.name}: no audit records found.", file=sys.stderr)
                    continue

                print(
                    f"Processing {log_path.name} with {len(audit_context.records)} item(s)...",
                    file=sys.stderr,
                )

                for record in audit_context.records:
                    if args.max_items is not None and processed_items >= args.max_items:
                        break

                    upload_temp_dir: Path | None = None
                    uploaded_file_ids: list[str] = []
                    result: dict[str, object] = {
                        "log_file": str(log_path),
                        "record_index": record.record_index,
                        "line_number": record.line_number,
                        "tex_file": str(tex_path),
                        "status": "error",
                        "output_text": "",
                        "gpt_name": "",
                        "gpt_ext_source": "",
                        "response_id": None,
                        "response_status": None,
                        "incomplete_details": None,
                        "usage_details": None,
                        "cost_details": None,
                        "langfuse_trace_url": None,
                        "raw_response_file": None,
                        "error": None,
                    }

                    try:
                        upload_temp_dir = create_upload_temp_dir()
                        upload_inputs = prepare_upload_inputs(
                            tex_path=tex_path,
                            bib_path=fallback_bib_path,
                            audit_log_path=log_path,
                            log_entry_index=record.record_index,
                            temp_dir=upload_temp_dir,
                        )
                        prompt = build_item_prompt(
                            log_name=log_path.name,
                            record_index=record.record_index,
                            line_number=record.line_number,
                            user_prompt=args.prompt,
                        )

                        trace_context = (
                            langfuse_runtime.client.start_as_current_observation(
                                name=args.langfuse_trace_name,
                                as_type="span",
                                input=build_langfuse_batch_span_input(
                                    log_path=log_path,
                                    record_index=record.record_index,
                                    line_number=record.line_number,
                                    tex_path=tex_path,
                                    prompt=prompt,
                                    model=args.model,
                                    reasoning_effort=args.reasoning_effort,
                                    max_output_tokens=args.max_output_tokens,
                                    mcp_url=args.mcp_url,
                                    mcp_label=args.mcp_label,
                                    mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
                                ),
                                metadata=build_langfuse_batch_span_metadata(
                                    log_path=log_path,
                                    record_index=record.record_index,
                                    line_number=record.line_number,
                                    output_log=output_log,
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
                                    metadata={
                                        "log_file": str(log_path),
                                        "record_index": record.record_index,
                                        "line_number": record.line_number,
                                    },
                                )
                                if langfuse_runtime.enabled and langfuse_runtime.propagate_attributes
                                else nullcontext()
                            )
                            with propagation_context:
                                generation_context = (
                                    langfuse_runtime.client.start_as_current_observation(
                                        name="openai.responses.create",
                                        as_type="generation",
                                        input={
                                            "log_file": str(log_path),
                                            "record_index": record.record_index,
                                            "line_number": record.line_number,
                                            "tex_file": str(upload_inputs.tex_path),
                                            "bib_file": str(upload_inputs.bib_path),
                                            "user_prompt": prompt,
                                            "tools": {
                                                "web_search_enabled": False,
                                                "mcp_label": args.mcp_label,
                                                "mcp_url": args.mcp_url,
                                                "allowed_tools": args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
                                            },
                                        },
                                        metadata={
                                            "log_file": str(log_path),
                                            "record_index": record.record_index,
                                            "line_number": record.line_number,
                                            "tex_filename": upload_inputs.tex_path.name,
                                            "bib_filename": upload_inputs.bib_path.name,
                                            "tool_policy": {
                                                "web_search_enabled": False,
                                                "mcp_only": True,
                                            },
                                        },
                                        model=args.model,
                                        model_parameters=build_langfuse_model_parameters(args),
                                    )
                                    if langfuse_runtime.enabled
                                    else nullcontext(None)
                                )
                                with generation_context as generation_observation:
                                    tex_upload = upload_file(client, upload_inputs.tex_path)
                                    bib_upload = upload_file(client, upload_inputs.bib_path)
                                    uploaded_file_ids.extend(
                                        [
                                            getattr(tex_upload, "id", ""),
                                            getattr(bib_upload, "id", ""),
                                        ]
                                    )
                                    response, response_data = run_response(
                                        client=client,
                                        model=args.model,
                                        reasoning_effort=args.reasoning_effort,
                                        max_output_tokens=args.max_output_tokens,
                                        prompt=prompt,
                                        tex_upload_id=required_file_id(tex_upload, upload_inputs.tex_path),
                                        bib_upload_id=required_file_id(bib_upload, upload_inputs.bib_path),
                                        mcp_tool=mcp_tool,
                                    )
                                    output_text = extract_output_text(response, response_data)
                                    usage_details = extract_langfuse_usage_details(response, response_data)
                                    cost_details = compute_langfuse_cost_details(usage_details, pricing)
                                    gpt_name, gpt_ext_source = parse_gpt_response(output_text)
                                    response_status = str(response_data.get("status") or "")
                                    incomplete_details = response_data.get("incomplete_details")
                                    if response_status == "completed":
                                        status = "ok"
                                    elif response_status:
                                        status = response_status
                                    else:
                                        status = "ok" if output_text else "empty"
                                    result.update(
                                        {
                                            "status": status,
                                            "output_text": output_text,
                                            "gpt_name": gpt_name,
                                            "gpt_ext_source": gpt_ext_source,
                                            "response_id": response_data.get("id"),
                                            "response_status": response_status or None,
                                            "incomplete_details": incomplete_details,
                                            "usage_details": usage_details or None,
                                            "cost_details": cost_details or None,
                                        }
                                    )
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
                                        result["langfuse_trace_url"] = langfuse_runtime.client.get_trace_url()
                                    if trace_observation is not None:
                                        trace_observation.update(
                                            output=build_langfuse_batch_span_output(
                                                response_data=response_data,
                                                output_text=output_text,
                                                usage_details=usage_details,
                                                cost_details=cost_details,
                                                result_status=status,
                                            )
                                        )
                                    if response_status != "completed":
                                        raw_response_path = make_raw_response_path(
                                            raw_response_dir=raw_response_dir,
                                            log_path=log_path,
                                            record_index=record.record_index,
                                        )
                                        raw_response_path.write_text(
                                            json.dumps(response_data, ensure_ascii=False, indent=2),
                                            encoding="utf-8",
                                        )
                                        result["raw_response_file"] = str(raw_response_path)
                                    if response_status and response_status != "completed":
                                        print(
                                            f"Warning: {log_path.name} item {record.record_index} returned "
                                            f"response_status={response_status} with no final completion.",
                                            file=sys.stderr,
                                        )
                    except Exception as exc:
                        result["error"] = str(exc)
                        if result["raw_response_file"] is None:
                            result["raw_response_file"] = None
                        print(
                            f"Warning: GPT query failed for {log_path.name} item {record.record_index}: {exc}",
                            file=sys.stderr,
                        )
                    finally:
                        cleanup_upload_temp_dir(upload_temp_dir)
                        if args.delete_uploads:
                            cleanup_uploaded_files(client, uploaded_file_ids)

                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    processed_items += 1
    finally:
        if "langfuse_runtime" in locals() and langfuse_runtime.enabled:
            flush_langfuse(langfuse_runtime)

    print(f"Wrote {processed_items} result(s) to {output_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
