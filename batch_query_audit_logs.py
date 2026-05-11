#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from dataclasses import dataclass
import json
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import query_claude_bedrock_tex_mcp as claude_query
import query_gemini_tex_mcp as gemini_query
import query_openai_tex_mcp as openai_query
from query_openai_tex_mcp import (
    DEFAULT_ALLOWED_MCP_TOOLS,
    DEFAULT_BODY_CONTEXT_LINES,
    DEFAULT_LANGFUSE_TRACE_NAME,
    DEFAULT_MCP_LABEL,
    DEFAULT_MCP_URL,
    build_langfuse_generation_result_metadata,
    build_langfuse_model_parameters,
    compute_langfuse_cost_details,
    flush_langfuse,
    load_environment_from_dotenv,
    parse_audit_log,
    parse_mcp_headers,
    parse_nonnegative_int,
    prepare_upload_inputs,
    read_api_key,
    setup_langfuse,
)
from statement_reference_audit_wholebody import normalize_arxiv_id

CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
RESULT_CSV_FIELDS = ("baseline_id", "baseline_num", "ai_id", "ai_num")
DEFAULT_LOGS_DIR = Path("statement_reference_audit_logs")
FALLBACK_LOGS_DIRS = (
    Path("random_arxiv_source_samples") / "statement_reference_audit_logs",
)
DEFAULT_BATCH_LANGFUSE_TRACE_NAME = f"{DEFAULT_LANGFUSE_TRACE_NAME}-batch"
DEFAULT_PROVIDER = "openai"


@dataclass(frozen=True)
class ProviderRuntime:
    name: str
    label: str
    module: Any
    generation_observation_name: str
    default_output_log_name: str
    default_raw_response_dir_name: str


PROVIDER_RUNTIMES: dict[str, ProviderRuntime] = {
    "openai": ProviderRuntime(
        name="openai",
        label="OpenAI",
        module=openai_query,
        generation_observation_name="openai.responses.create",
        default_output_log_name="openai_query_results.csv",
        default_raw_response_dir_name="raw_openai_responses",
    ),
    "gemini": ProviderRuntime(
        name="gemini",
        label="Gemini",
        module=gemini_query,
        generation_observation_name="google.genai.models.generate_content",
        default_output_log_name="gemini_query_results.csv",
        default_raw_response_dir_name="raw_gemini_responses",
    ),
    "claude": ProviderRuntime(
        name="claude",
        label="Claude Bedrock",
        module=claude_query,
        generation_observation_name="bedrock-runtime.converse",
        default_output_log_name="claude_bedrock_query_results.csv",
        default_raw_response_dir_name="raw_claude_bedrock_responses",
    ),
}


@dataclass(frozen=True)
class RetryManifest:
    source_path: Path
    failed_only: bool
    selected_count: int
    by_path: dict[tuple[str, int], dict[str, Any]]
    by_name: dict[tuple[str, int], dict[str, Any]]
    by_baseline: dict[tuple[str, str], list[dict[str, Any]]]


@dataclass(frozen=True)
class MergeOutcome:
    rows: list[dict[str, Any]]
    replaced_count: int
    appended_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run theorem-citation queries over every audit-log entry in a "
            "statement_reference_audit_logs directory and write CSV results."
        )
    )
    parser.add_argument(
        "--provider",
        choices=tuple(PROVIDER_RUNTIMES),
        default=DEFAULT_PROVIDER,
        help=f"Model provider to use. Defaults to {DEFAULT_PROVIDER}.",
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
        help="Output CSV path. Defaults to a provider-specific file under <logs-dir>.",
    )
    parser.add_argument(
        "--retry-from",
        type=Path,
        help=(
            "Existing detailed JSONL results log or four-column CSV results log to use "
            "as a retry manifest. When provided, only items present in that file are "
            "eligible to run."
        ),
    )
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help=(
            "Retry only items whose previous detailed JSONL status was not ok, or whose "
            "CSV ai_id/ai_num fields are blank or incomplete. When --retry-from is "
            "omitted, the script reads --output-log or the provider's default CSV."
        ),
    )
    parser.add_argument(
        "--raw-response-dir",
        type=Path,
        help=(
            "Optional directory where raw provider response JSON files are written for "
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
        "--tex-context-lines",
        "--body-context-lines",
        "--context-lines",
        dest="body_context_lines",
        type=parse_nonnegative_int,
        default=DEFAULT_BODY_CONTEXT_LINES,
        metavar="N",
        help=(
            "Number of main TeX body lines before and after each logged citation line to upload. "
            f"Defaults to {DEFAULT_BODY_CONTEXT_LINES}."
        ),
    )
    parser.add_argument(
        "--model",
        help="Model to use. Defaults to the selected provider's default model.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh"),
        help="Reasoning effort. Defaults to the selected provider's default value.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        help="Maximum output tokens. Defaults to the selected provider's default value.",
    )
    parser.add_argument(
        "--api-key-env",
        help=(
            "Environment variable containing the provider API key. Defaults to the "
            "selected provider's default value."
        ),
    )
    parser.add_argument(
        "--aws-region",
        help=(
            "AWS region for --provider claude. Defaults to AWS_REGION, "
            "AWS_DEFAULT_REGION, or us-east-1."
        ),
    )
    parser.add_argument(
        "--aws-profile",
        help="Optional AWS profile name for --provider claude.",
    )
    parser.add_argument(
        "--bedrock-endpoint-url",
        help="Optional custom bedrock-runtime endpoint URL for --provider claude.",
    )
    parser.add_argument(
        "--bedrock-connect-timeout",
        type=float,
        default=claude_query.DEFAULT_BEDROCK_CONNECT_TIMEOUT,
        metavar="SECONDS",
        help=(
            "Bedrock connect timeout for --provider claude. Defaults to "
            f"{claude_query.DEFAULT_BEDROCK_CONNECT_TIMEOUT:g} seconds."
        ),
    )
    parser.add_argument(
        "--bedrock-read-timeout",
        type=float,
        default=claude_query.DEFAULT_BEDROCK_READ_TIMEOUT,
        metavar="SECONDS",
        help=(
            "Bedrock read timeout for --provider claude. Defaults to "
            f"{claude_query.DEFAULT_BEDROCK_READ_TIMEOUT:g} seconds."
        ),
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
        "--mcp-read-timeout",
        type=float,
        default=claude_query.DEFAULT_MCP_READ_TIMEOUT,
        metavar="SECONDS",
        help=(
            "MCP tool-call timeout for --provider claude. Defaults to "
            f"{claude_query.DEFAULT_MCP_READ_TIMEOUT:g} seconds."
        ),
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
        help="Delete uploaded provider files after each query finishes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    parser.add_argument(
        "--update-existing-log",
        action="store_true",
        help=(
            "Update an existing CSV results log in place by replacing rows that match "
            "the current run's results. This is intended for retry runs and preserves "
            "untouched rows instead of replacing the whole file."
        ),
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Optional cap on the total number of audit-log entries to query.",
    )
    return parser.parse_args()


def resolve_provider_runtime(provider_name: str) -> ProviderRuntime:
    return PROVIDER_RUNTIMES[provider_name]


def normalize_cli_path(path: Path | None) -> Path | None:
    if path is None or sys.platform.startswith("win"):
        return path

    rendered = str(path)
    if "\\" not in rendered:
        return path

    normalized = rendered.replace("\\", "/")
    if normalized.startswith("./") or normalized.startswith("../") or normalized.startswith("/"):
        return Path(normalized)
    if normalized.startswith("."):
        return Path(normalized)
    return Path(normalized)


def normalize_cli_path_args(args: argparse.Namespace) -> argparse.Namespace:
    for field_name in ("logs_dir", "search_root", "output_log", "retry_from", "raw_response_dir"):
        setattr(args, field_name, normalize_cli_path(getattr(args, field_name, None)))
    return args


def compact_path_string(value: str) -> str:
    return value.replace("\\", "").replace("/", "").strip().lower()


def build_retry_output_log_name(default_output_log_name: str, failed_only: bool) -> str:
    path = Path(default_output_log_name)
    suffix = "_retry_failed" if failed_only else "_retry"
    extension = path.suffix or ".csv"
    return f"{path.stem}{suffix}{extension}"


def build_retry_raw_response_dir_name(default_raw_response_dir_name: str, failed_only: bool) -> str:
    suffix = "_retry_failed" if failed_only else "_retry"
    return f"{default_raw_response_dir_name}{suffix}"


def apply_provider_defaults(
    args: argparse.Namespace,
    provider_runtime: ProviderRuntime,
) -> argparse.Namespace:
    if args.model is None:
        args.model = provider_runtime.module.DEFAULT_MODEL
    if args.reasoning_effort is None:
        args.reasoning_effort = provider_runtime.module.DEFAULT_REASONING_EFFORT
    if args.max_output_tokens is None:
        args.max_output_tokens = provider_runtime.module.DEFAULT_MAX_OUTPUT_TOKENS
    if args.api_key_env is None:
        args.api_key_env = provider_runtime.module.DEFAULT_API_KEY_ENV
    return args


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


def build_item_prompt(
    log_name: str,
    record_index: int,
    line_number: int,
    user_prompt: str | None,
    body_context_lines: int,
) -> str:
    default_instruction = (
        "Task:\n"
        "1. Identify the external arXiv identifier of the cited source paper containing the missing result.\n"
        '2. Identify the cited theorem-like result name/number, for example "Theorem 1.2" or "Proposition 3.4".\n'
        "Use theorem_search if it helps.\n\n"
        "Return JSON only, with exactly this shape:\n"
        '{"ai_id": <string or null>, "ai_num": <string or null>}'
    )
    instruction = user_prompt.strip() if user_prompt else default_instruction
    return (
        "Analyze one logged theorem-style citation occurrence in the attached LaTeX source and bibliography.\n\n"
        f"Log file: {log_name}\n"
        f"Log item: {record_index}\n"
        f"Logged source line number: {line_number}\n\n"
        "The attached LaTeX source is a main-body excerpt containing up to "
        f"{body_context_lines} line(s) before and after the logged line.\n"
        "The relevant citation on that line has been masked as [Citation Needed].\n"
        "The bibliography upload is not line-windowed; the matching bibliography entry "
        "has been removed from the uploaded bibliography text when it could be identified.\n\n"
        f"{instruction}"
    )


def parse_ai_response(output_text: str) -> tuple[str, str]:
    text = output_text.strip()
    if not text:
        return "", ""

    fenced_match = CODE_FENCE_RE.match(text)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start_index = text.find("{")
        end_index = text.rfind("}")
        if start_index < 0 or end_index <= start_index:
            return "", ""
        try:
            data = json.loads(text[start_index : end_index + 1])
        except json.JSONDecodeError:
            return "", ""
    if not isinstance(data, dict):
        return "", ""

    ai_id = normalize_optional_string(
        data.get("ai_id")
        or data.get("arxiv_id")
        or data.get("located_arxiv_id")
        or data.get("gpt_ext_source")
        or data.get("ext_source")
    )
    ai_num = normalize_optional_string(
        data.get("ai_num")
        or data.get("theorem_name")
        or data.get("theorem")
        or data.get("result_name")
        or data.get("located_theorem_name")
        or data.get("gpt_name")
        or data.get("name")
    )
    return ai_id, ai_num


def normalize_optional_string(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"null", "none", "unknown"}:
        return ""
    return cleaned


def normalize_result_id(value: object) -> str:
    text = normalize_optional_string(value)
    return normalize_arxiv_id(text) if text else ""


def baseline_fields_from_record(record: object) -> tuple[str, str]:
    baseline_id = normalize_result_id(getattr(record, "baseline_id", ""))
    baseline_num = normalize_optional_string(getattr(record, "baseline_num", ""))
    return baseline_id, baseline_num


def result_to_csv_row(result: dict[str, object]) -> dict[str, str]:
    return {
        "baseline_id": normalize_result_id(result.get("baseline_id")),
        "baseline_num": normalize_optional_string(result.get("baseline_num")),
        "ai_id": normalize_result_id(result.get("ai_id")),
        "ai_num": normalize_optional_string(result.get("ai_num")),
    }


def write_result_csv_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(result_to_csv_row(row))


def coerce_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_retry_path_key(value: object) -> str:
    text = normalize_optional_string(value)
    if not text:
        return ""
    normalized = text.replace("\\", "/")
    return normalized.lower()


def normalize_retry_name_key(value: object) -> str:
    text = normalize_optional_string(value)
    if not text:
        return ""
    candidates = (
        text.replace("\\", "/").rsplit("/", 1)[-1],
        PurePosixPath(text).name,
        PureWindowsPath(text).name,
    )
    for candidate in candidates:
        normalized = candidate.strip().lower()
        if normalized:
            return normalized
    return ""


def build_retry_baseline_key(row: dict[str, Any]) -> tuple[str, str] | None:
    baseline_id = normalize_result_id(row.get("baseline_id"))
    baseline_num = normalize_optional_string(row.get("baseline_num")).lower()
    if not baseline_id and not baseline_num:
        return None
    return baseline_id, baseline_num


def row_has_complete_ai_output(row: dict[str, Any]) -> bool:
    return bool(
        normalize_result_id(row.get("ai_id"))
        and normalize_optional_string(row.get("ai_num"))
    )


def row_has_any_ai_output(row: dict[str, Any]) -> bool:
    return bool(
        normalize_result_id(row.get("ai_id"))
        or normalize_optional_string(row.get("ai_num"))
    )


def replacement_should_update_existing(
    existing_row: dict[str, Any],
    replacement_row: dict[str, Any],
) -> bool:
    if row_has_complete_ai_output(replacement_row):
        return True
    if not row_has_any_ai_output(existing_row):
        return True
    if row_has_any_ai_output(replacement_row) and not row_has_complete_ai_output(existing_row):
        return True
    return False


def retry_manifest_row_is_failed(row: dict[str, Any]) -> bool:
    previous_status = normalize_optional_string(row.get("status")).lower()
    if previous_status:
        return previous_status != "ok"
    return not row_has_complete_ai_output(row)


def load_retry_manifest(
    source_path: Path,
    *,
    provider_name: str,
    failed_only: bool,
) -> RetryManifest:
    resolved_source = source_path.resolve()
    if not resolved_source.exists():
        raise RuntimeError(f"retry manifest does not exist: {resolved_source}")
    if not resolved_source.is_file():
        raise RuntimeError(f"retry manifest is not a file: {resolved_source}")

    by_path: dict[tuple[str, int], dict[str, Any]] = {}
    by_name: dict[tuple[str, int], dict[str, Any]] = {}
    by_baseline: dict[tuple[str, str], list[dict[str, Any]]] = {}
    selected_count = 0

    def consider_row(row: dict[str, Any]) -> None:
        nonlocal selected_count

        row_provider = normalize_optional_string(row.get("provider")).lower()
        if row_provider and row_provider != provider_name:
            return

        if failed_only and not retry_manifest_row_is_failed(row):
            return

        record_index = coerce_optional_int(row.get("record_index"))
        path_key = normalize_retry_path_key(row.get("log_file"))
        name_key = normalize_retry_name_key(row.get("log_file"))
        baseline_key = build_retry_baseline_key(row)
        if record_index is None and baseline_key is None:
            return

        row_selected = False
        if record_index is not None:
            if path_key:
                by_path[(path_key, record_index)] = row
                row_selected = True
            if name_key:
                by_name[(name_key, record_index)] = row
                row_selected = True

        if baseline_key is not None:
            by_baseline.setdefault(baseline_key, []).append(row)
            row_selected = True

        if row_selected:
            selected_count += 1

    if resolved_source.suffix.lower() == ".csv":
        with resolved_source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_fields = [
                field for field in RESULT_CSV_FIELDS if field not in (reader.fieldnames or [])
            ]
            if missing_fields:
                raise RuntimeError(
                    f"retry CSV is missing required column(s): {', '.join(missing_fields)}"
                )
            for row in reader:
                consider_row(dict(row))
    else:
        with resolved_source.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"retry manifest {resolved_source} contains invalid JSON on line "
                        f"{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise RuntimeError(
                        f"retry manifest {resolved_source} contains a non-object JSON value "
                        f"on line {line_number}."
                    )
                consider_row(row)

    return RetryManifest(
        source_path=resolved_source,
        failed_only=failed_only,
        selected_count=selected_count,
        by_path=by_path,
        by_name=by_name,
        by_baseline=by_baseline,
    )


def resolve_retry_manifest_path(
    args: argparse.Namespace,
    logs_dir: Path,
    provider_runtime: ProviderRuntime,
) -> Path | None:
    if args.retry_from is not None:
        return args.retry_from
    if args.retry_failed_only:
        return (args.output_log or (logs_dir / provider_runtime.default_output_log_name)).resolve()
    return None


def recover_retry_manifest_path(path: Path, logs_dir: Path) -> Path:
    if path.exists() or sys.platform.startswith("win"):
        return path

    rendered = normalize_optional_string(str(path))
    if not rendered or "/" in rendered or "\\" in rendered or not rendered.endswith(".jsonl"):
        return path

    search_root = logs_dir.parent.resolve()
    target = compact_path_string(rendered)
    matches: list[Path] = []

    for candidate in search_root.rglob("*.jsonl"):
        resolved_candidate = candidate.resolve()
        candidate_variants = [resolved_candidate.name, str(resolved_candidate)]
        try:
            relative_candidate = resolved_candidate.relative_to(search_root)
            candidate_variants.append(str(relative_candidate))
            candidate_variants.append(f".{Path(relative_candidate).as_posix()}")
        except ValueError:
            pass

        if any(compact_path_string(variant) == target for variant in candidate_variants):
            matches.append(resolved_candidate)

    if len(matches) == 1:
        return matches[0]
    return path


def load_result_log_rows(source_path: Path) -> list[dict[str, Any]]:
    resolved_source = source_path.resolve()
    if not resolved_source.exists():
        raise RuntimeError(f"results log does not exist: {resolved_source}")
    if not resolved_source.is_file():
        raise RuntimeError(f"results log is not a file: {resolved_source}")

    if resolved_source.suffix.lower() == ".csv":
        with resolved_source.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    rows: list[dict[str, Any]] = []
    with resolved_source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"results log {resolved_source} contains invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    f"results log {resolved_source} contains a non-object JSON value "
                    f"on line {line_number}."
                )
            rows.append(row)
    return rows


def build_result_row_match_keys(
    row: dict[str, Any],
    *,
    default_provider: str | None = None,
) -> list[tuple[str, str, str]]:
    baseline_id = normalize_result_id(row.get("baseline_id"))
    baseline_num = normalize_optional_string(row.get("baseline_num")).lower()
    if baseline_id or baseline_num:
        return [("baseline", baseline_id, baseline_num)]

    provider = normalize_optional_string(row.get("provider")).lower() or (default_provider or "")
    if not provider:
        return []

    record_index = coerce_optional_int(row.get("record_index"))
    if record_index is None:
        return []

    keys: list[tuple[str, str, str]] = []
    path_key = normalize_retry_path_key(row.get("log_file"))
    if path_key:
        keys.append((provider, f"path:{path_key}", str(record_index)))

    name_key = normalize_retry_name_key(row.get("log_file"))
    if name_key:
        keys.append((provider, f"name:{name_key}", str(record_index)))

    return keys


def merge_result_log_rows(
    existing_rows: list[dict[str, Any]],
    updated_rows: list[dict[str, Any]],
    *,
    default_provider: str,
) -> MergeOutcome:
    updated_rows_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    ordered_updated_rows: list[dict[str, Any]] = []

    for row in updated_rows:
        ordered_updated_rows.append(row)
        for key in build_result_row_match_keys(row, default_provider=default_provider):
            updated_rows_by_key.setdefault(key, []).append(row)

    merged_rows: list[dict[str, Any]] = []
    consumed_update_ids: set[int] = set()
    replaced_count = 0

    for row in existing_rows:
        replacement: dict[str, Any] | None = None
        for key in build_result_row_match_keys(row, default_provider=default_provider):
            for candidate in updated_rows_by_key.get(key, []):
                if id(candidate) in consumed_update_ids:
                    continue
                replacement = candidate
                break
            if replacement is not None:
                break

        if replacement is not None and id(replacement) not in consumed_update_ids:
            consumed_update_ids.add(id(replacement))
            if replacement_should_update_existing(row, replacement):
                merged_rows.append(replacement)
                replaced_count += 1
            else:
                merged_rows.append(row)
        else:
            merged_rows.append(row)

    appended_count = 0
    for row in ordered_updated_rows:
        if id(row) in consumed_update_ids:
            continue
        merged_rows.append(row)
        consumed_update_ids.add(id(row))
        appended_count += 1

    return MergeOutcome(
        rows=merged_rows,
        replaced_count=replaced_count,
        appended_count=appended_count,
    )


def lookup_retry_manifest_row(
    retry_manifest: RetryManifest,
    log_path: Path,
    record_index: int,
    record: object | None = None,
) -> dict[str, Any] | None:
    path_key = normalize_retry_path_key(str(log_path.resolve()))
    if path_key:
        row = retry_manifest.by_path.get((path_key, record_index))
        if row is not None:
            return row

    name_key = normalize_retry_name_key(log_path.name)
    if name_key:
        row = retry_manifest.by_name.get((name_key, record_index))
        if row is not None:
            return row

    if record is not None:
        baseline_id, baseline_num = baseline_fields_from_record(record)
        baseline_key = (baseline_id, baseline_num.lower())
        rows = retry_manifest.by_baseline.get(baseline_key)
        if rows:
            return rows[0]

    return None


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


def make_raw_response_path(
    raw_response_dir: Path,
    log_path: Path,
    record_index: int,
) -> Path:
    return raw_response_dir / f"{log_path.stem}_item_{record_index}.json"


def build_langfuse_batch_span_input(
    provider_name: str,
    log_path: Path,
    record_index: int,
    line_number: int,
    body_context_lines: int,
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
        "provider": provider_name,
        "log_file": str(log_path),
        "record_index": record_index,
        "line_number": line_number,
        "body_context_lines": body_context_lines,
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
    provider_name: str,
    log_path: Path,
    record_index: int,
    line_number: int,
    output_log: Path,
) -> dict[str, Any]:
    return {
        "provider": provider_name,
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
        "response_id": extract_response_id(response_data),
        "status": result_status,
        "response_status": extract_response_status(response_data),
        "output_text": output_text or None,
        "usage_details": usage_details or None,
        "cost_details": cost_details or None,
    }


def build_langfuse_generation_output(
    response_data: dict[str, Any],
    output_text: str,
    usage_details: dict[str, int],
    cost_details: dict[str, float],
) -> dict[str, Any]:
    return {
        "response_id": extract_response_id(response_data),
        "status": extract_response_status(response_data) or None,
        "output_text": output_text or None,
        "usage_details": usage_details or None,
        "cost_details": cost_details or None,
    }


def extract_response_id(response_data: dict[str, Any]) -> str | None:
    response_id = normalize_optional_string(
        response_data.get("id") or response_data.get("response_id")
    )
    return response_id or None


def extract_candidate_finish_reasons(response_data: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    candidates = response_data.get("candidates")
    if not isinstance(candidates, list):
        candidates = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        reason = normalize_optional_string(candidate.get("finish_reason"))
        if reason and reason not in reasons:
            reasons.append(reason)
    return reasons


def extract_response_status(response_data: dict[str, Any]) -> str:
    direct_status = normalize_optional_string(response_data.get("status"))
    if direct_status:
        return direct_status

    prompt_feedback = response_data.get("prompt_feedback")
    if isinstance(prompt_feedback, dict):
        block_reason = normalize_optional_string(prompt_feedback.get("block_reason"))
        if block_reason:
            return f"blocked:{block_reason}"

    finish_reasons = extract_candidate_finish_reasons(response_data)
    if not finish_reasons:
        return ""

    normalized = {reason.lower() for reason in finish_reasons}
    if normalized == {"stop"}:
        return "completed"
    return ",".join(finish_reasons)


def derive_result_status(response_status: str, output_text: str) -> str:
    normalized = response_status.strip().lower()
    if normalized == "completed":
        return "ok"
    if normalized.startswith("blocked"):
        return "blocked"
    if normalized:
        return normalized
    return "ok" if output_text else "empty"


def maybe_ensure_provider_dependencies(provider_runtime: ProviderRuntime) -> None:
    ensure_runtime_dependencies = getattr(provider_runtime.module, "ensure_runtime_dependencies", None)
    if callable(ensure_runtime_dependencies):
        ensure_runtime_dependencies()


def upload_file_refs(provider_runtime: ProviderRuntime, tex_upload, bib_upload) -> list[str]:
    if provider_runtime.name == "claude":
        return []
    ref_attribute = "id" if provider_runtime.name == "openai" else "name"
    return [
        ref
        for ref in (
            getattr(tex_upload, ref_attribute, ""),
            getattr(bib_upload, ref_attribute, ""),
        )
        if ref
    ]


def make_provider_client(
    provider_runtime: ProviderRuntime,
    args: argparse.Namespace,
    *,
    enable_langfuse: bool,
):
    make_client_from_args = getattr(provider_runtime.module, "make_client_from_args", None)
    if callable(make_client_from_args):
        return make_client_from_args(args, enable_langfuse=enable_langfuse)

    api_key = read_api_key(args.api_key_env)
    return provider_runtime.module.make_client(
        api_key,
        enable_langfuse=enable_langfuse,
    )


def run_provider_response(
    provider_runtime: ProviderRuntime,
    client,
    args: argparse.Namespace,
    prompt: str,
    tex_upload,
    bib_upload,
    tex_upload_path: Path,
    bib_upload_path: Path,
    mcp_headers: dict[str, str],
):
    if provider_runtime.name == "openai":
        mcp_tool = provider_runtime.module.build_mcp_tool(
            mcp_label=args.mcp_label,
            mcp_url=args.mcp_url,
            mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
            mcp_headers=mcp_headers,
        )
        return provider_runtime.module.run_response(
            client=client,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            prompt=prompt,
            tex_upload_id=provider_runtime.module.required_file_id(tex_upload, tex_upload_path),
            bib_upload_id=provider_runtime.module.required_file_id(bib_upload, bib_upload_path),
            mcp_tool=mcp_tool,
        )

    if provider_runtime.name == "gemini":
        response, response_data, available_tool_names, thinking_note = provider_runtime.module.run_response(
            client=client,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            prompt=prompt,
            tex_upload=tex_upload,
            bib_upload=bib_upload,
            mcp_label=args.mcp_label,
            mcp_url=args.mcp_url,
            requested_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
            mcp_headers=mcp_headers,
            mcp_read_timeout=args.mcp_read_timeout,
        )
        if thinking_note:
            print(thinking_note, file=sys.stderr)
        requested_tools = args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS)
        if available_tool_names == requested_tools:
            print(
                "MCP session tools: " + ", ".join(available_tool_names) + ".",
                file=sys.stderr,
            )
        else:
            print(
                "Requested MCP tools: "
                + ", ".join(requested_tools)
                + "; MCP session exposes: "
                + ", ".join(available_tool_names)
                + " (Gemini SDK MCP sessions do not enforce client-side tool allowlists).",
                file=sys.stderr,
            )
        return response, response_data

    if provider_runtime.name == "claude":
        response, response_data, available_tool_names, thinking_note = provider_runtime.module.run_response(
            client=client,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            prompt=prompt,
            tex_upload=tex_upload,
            bib_upload=bib_upload,
            mcp_label=args.mcp_label,
            mcp_url=args.mcp_url,
            requested_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
            mcp_headers=mcp_headers,
        )
        if thinking_note:
            print(thinking_note, file=sys.stderr)
        print(
            "MCP session tools exposed to Claude: "
            + ", ".join(available_tool_names)
            + ".",
            file=sys.stderr,
        )
        return response, response_data

    raise RuntimeError(f"Unsupported provider: {provider_runtime.name}")


def main() -> int:
    args = parse_args()
    args = normalize_cli_path_args(args)
    provider_runtime = resolve_provider_runtime(args.provider)
    args = apply_provider_defaults(args, provider_runtime)

    try:
        logs_dir = resolve_logs_dir(args.logs_dir)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if args.max_items is not None and args.max_items <= 0:
        print("Error: max-items must be positive when provided.", file=sys.stderr)
        return 1
    search_root = (args.search_root or logs_dir.parent).resolve()
    retry_manifest_path = resolve_retry_manifest_path(args, logs_dir, provider_runtime)
    retry_manifest: RetryManifest | None = None
    if retry_manifest_path is not None:
        recovered_retry_manifest_path = recover_retry_manifest_path(retry_manifest_path, logs_dir)
        if recovered_retry_manifest_path != retry_manifest_path:
            print(
                f"Recovered retry manifest path {retry_manifest_path} -> "
                f"{recovered_retry_manifest_path}",
                file=sys.stderr,
            )
            retry_manifest_path = recovered_retry_manifest_path
        try:
            retry_manifest = load_retry_manifest(
                retry_manifest_path,
                provider_name=provider_runtime.name,
                failed_only=args.retry_failed_only,
            )
        except RuntimeError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    if args.update_existing_log and retry_manifest is None:
        print(
            "Error: --update-existing-log requires --retry-from or --retry-failed-only.",
            file=sys.stderr,
        )
        return 1

    default_output_log = logs_dir / provider_runtime.default_output_log_name
    if args.output_log is not None:
        output_log = args.output_log.resolve()
    elif args.update_existing_log:
        output_log = default_output_log.resolve()
    elif retry_manifest is not None:
        output_log = (
            logs_dir
            / build_retry_output_log_name(
                provider_runtime.default_output_log_name,
                failed_only=args.retry_failed_only,
            )
        ).resolve()
    else:
        output_log = default_output_log.resolve()
    if args.update_existing_log and not output_log.exists():
        print(
            f"Error: output log to update does not exist: {output_log}",
            file=sys.stderr,
        )
        return 1
    if not args.update_existing_log and output_log.exists() and not args.overwrite:
        print(
            f"Error: output log already exists: {output_log}. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 1
    raw_response_dir = (
        args.raw_response_dir
        or (
            output_log.parent
            / (
                build_retry_raw_response_dir_name(
                    provider_runtime.default_raw_response_dir_name,
                    failed_only=args.retry_failed_only,
                )
                if retry_manifest is not None
                else provider_runtime.default_raw_response_dir_name
            )
        )
    ).resolve()

    try:
        dotenv_path = load_environment_from_dotenv()
        langfuse_runtime = setup_langfuse(args)
        maybe_ensure_provider_dependencies(provider_runtime)
        pricing = provider_runtime.module.resolve_model_pricing(args)
        client = make_provider_client(
            provider_runtime,
            args,
            enable_langfuse=langfuse_runtime.enabled,
        )
        mcp_headers = parse_mcp_headers(args.mcp_headers or [])
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if dotenv_path:
        print(f"Loaded environment from {dotenv_path}", file=sys.stderr)
    if langfuse_runtime.status_message:
        print(langfuse_runtime.status_message, file=sys.stderr)
    if retry_manifest is not None:
        retry_scope = "failed" if retry_manifest.failed_only else "previously logged"
        print(
            f"Loaded retry manifest from {retry_manifest.source_path} with "
            f"{retry_manifest.selected_count} {retry_scope} item(s).",
            file=sys.stderr,
        )

    mcp_tools = args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS)
    log_paths = sorted_log_paths(logs_dir)
    if not log_paths:
        print(f"Error: no log files found in {logs_dir}", file=sys.stderr)
        return 1

    output_log.parent.mkdir(parents=True, exist_ok=True)
    raw_response_dir.mkdir(parents=True, exist_ok=True)
    processed_items = 0
    result_rows: list[dict[str, Any]] = []
    try:
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

            records_to_process = audit_context.records
            if retry_manifest is not None:
                records_to_process = [
                    record
                    for record in audit_context.records
                    if lookup_retry_manifest_row(
                        retry_manifest,
                        log_path=log_path,
                        record_index=record.record_index,
                        record=record,
                    )
                    is not None
                ]
                if not records_to_process:
                    print(
                        f"Skipping {log_path.name}: no retry-selected items.",
                        file=sys.stderr,
                    )
                    continue

            print(
                f"Processing {log_path.name} with {len(records_to_process)} item(s)...",
                file=sys.stderr,
            )

            for record in records_to_process:
                if args.max_items is not None and processed_items >= args.max_items:
                    break

                upload_temp_dir: Path | None = None
                uploaded_file_refs: list[str] = []
                baseline_id, baseline_num = baseline_fields_from_record(record)
                result: dict[str, object] = {
                    "baseline_id": baseline_id,
                    "baseline_num": baseline_num,
                    "ai_id": "",
                    "ai_num": "",
                    "provider": provider_runtime.name,
                    "log_file": str(log_path),
                    "record_index": record.record_index,
                    "line_number": record.line_number,
                    "body_context_lines": args.body_context_lines,
                    "tex_file": str(tex_path),
                    "status": "error",
                    "output_text": "",
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
                    upload_temp_dir = provider_runtime.module.create_upload_temp_dir()
                    upload_inputs = prepare_upload_inputs(
                        tex_path=tex_path,
                        bib_path=fallback_bib_path,
                        audit_log_path=log_path,
                        log_entry_index=record.record_index,
                        temp_dir=upload_temp_dir,
                        body_context_lines=args.body_context_lines,
                    )
                    prompt = build_item_prompt(
                        log_name=log_path.name,
                        record_index=record.record_index,
                        line_number=record.line_number,
                        user_prompt=args.prompt,
                        body_context_lines=args.body_context_lines,
                    )

                    trace_context = (
                        langfuse_runtime.client.start_as_current_observation(
                            name=args.langfuse_trace_name,
                            as_type="span",
                            input=build_langfuse_batch_span_input(
                                provider_name=provider_runtime.name,
                                log_path=log_path,
                                record_index=record.record_index,
                                line_number=record.line_number,
                                body_context_lines=args.body_context_lines,
                                tex_path=tex_path,
                                prompt=prompt,
                                model=args.model,
                                reasoning_effort=args.reasoning_effort,
                                max_output_tokens=args.max_output_tokens,
                                mcp_url=args.mcp_url,
                                mcp_label=args.mcp_label,
                                mcp_tools=mcp_tools,
                            ),
                            metadata=build_langfuse_batch_span_metadata(
                                provider_name=provider_runtime.name,
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
                                    "provider": provider_runtime.name,
                                    "log_file": str(log_path),
                                    "record_index": str(record.record_index),
                                    "line_number": str(record.line_number),
                                },
                            )
                            if langfuse_runtime.enabled and langfuse_runtime.propagate_attributes
                            else nullcontext()
                        )
                        with propagation_context:
                            generation_context = (
                                langfuse_runtime.client.start_as_current_observation(
                                    name=provider_runtime.generation_observation_name,
                                    as_type="generation",
                                    input={
                                        "provider": provider_runtime.name,
                                        "log_file": str(log_path),
                                        "record_index": record.record_index,
                                        "line_number": record.line_number,
                                        "body_context_lines": args.body_context_lines,
                                        "tex_file": str(upload_inputs.tex_path),
                                        "bib_file": str(upload_inputs.bib_path),
                                        "user_prompt": prompt,
                                        "tools": {
                                            "web_search_enabled": False,
                                            "mcp_label": args.mcp_label,
                                            "mcp_url": args.mcp_url,
                                            "allowed_tools": mcp_tools,
                                        },
                                    },
                                    metadata={
                                        "provider": provider_runtime.name,
                                        "log_file": str(log_path),
                                        "record_index": record.record_index,
                                        "line_number": record.line_number,
                                        "body_context_lines": args.body_context_lines,
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
                                tex_upload = provider_runtime.module.upload_file(
                                    client,
                                    upload_inputs.tex_path,
                                )
                                bib_upload = provider_runtime.module.upload_file(
                                    client,
                                    upload_inputs.bib_path,
                                )
                                uploaded_file_refs.extend(
                                    upload_file_refs(provider_runtime, tex_upload, bib_upload)
                                )
                                response, response_data = run_provider_response(
                                    provider_runtime=provider_runtime,
                                    client=client,
                                    args=args,
                                    prompt=prompt,
                                    tex_upload=tex_upload,
                                    bib_upload=bib_upload,
                                    tex_upload_path=upload_inputs.tex_path,
                                    bib_upload_path=upload_inputs.bib_path,
                                    mcp_headers=mcp_headers,
                                )
                                output_text = provider_runtime.module.extract_output_text(
                                    response,
                                    response_data,
                                )
                                usage_details = provider_runtime.module.extract_langfuse_usage_details(
                                    response,
                                    response_data,
                                )
                                cost_details = compute_langfuse_cost_details(usage_details, pricing)
                                ai_id, ai_num = parse_ai_response(output_text)
                                response_status = extract_response_status(response_data)
                                incomplete_details = response_data.get("incomplete_details")
                                status = derive_result_status(response_status, output_text)
                                result.update(
                                    {
                                        "status": status,
                                        "output_text": output_text,
                                        "ai_id": ai_id,
                                        "ai_num": ai_num,
                                        "response_id": extract_response_id(response_data),
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
                                if status != "ok":
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
                                    print(
                                        f"Warning: {log_path.name} item {record.record_index} returned "
                                        f"status={status}"
                                        + (
                                            f" (response_status={response_status})."
                                            if response_status
                                            else "."
                                        ),
                                        file=sys.stderr,
                                    )
                except Exception as exc:
                    result["error"] = format_exception_message(exc)
                    print(
                        f"Warning: {provider_runtime.label} query failed for "
                        f"{log_path.name} item {record.record_index}: "
                        f"{format_exception_message(exc)}",
                        file=sys.stderr,
                    )
                finally:
                    provider_runtime.module.cleanup_upload_temp_dir(upload_temp_dir)
                    if args.delete_uploads:
                        provider_runtime.module.cleanup_uploaded_files(
                            client,
                            uploaded_file_refs,
                        )

                result_rows.append(dict(result))
                processed_items += 1

        if args.update_existing_log:
            try:
                existing_rows = load_result_log_rows(output_log)
            except RuntimeError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return 1

            merge_outcome = merge_result_log_rows(
                existing_rows,
                result_rows,
                default_provider=provider_runtime.name,
            )
            write_result_csv_rows(output_log, merge_outcome.rows)
        else:
            write_result_csv_rows(output_log, result_rows)
    finally:
        if "langfuse_runtime" in locals() and langfuse_runtime.enabled:
            flush_langfuse(langfuse_runtime)
        if "client" in locals():
            try:
                client.close()
            except Exception:
                pass

    if args.update_existing_log:
        print(
            f"Updated {output_log} with {processed_items} {provider_runtime.label} retry "
            f"result(s) ({merge_outcome.replaced_count} replaced, "
            f"{merge_outcome.appended_count} appended)."
        )
    else:
        print(f"Wrote {processed_items} {provider_runtime.label} result(s) to {output_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
