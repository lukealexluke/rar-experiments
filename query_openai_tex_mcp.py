#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
import re
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

from statement_reference_audit import (
    DEFAULT_CITE_COMMANDS,
    find_macro_occurrences,
    iter_bbl_entries,
    iter_bib_entries,
    normalize_arxiv_id,
    read_candidate_text,
)


DEFAULT_API_KEY_ENV = "openai_key"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_MAX_OUTPUT_TOKENS = 12000
DEFAULT_MCP_LABEL = "theoremsearch"
DEFAULT_MCP_URL = "https://api.theoremsearch.com/mcp"
DEFAULT_ALLOWED_MCP_TOOLS = ("theorem_search",)
DEFAULT_APPROVAL_LIMIT = 8
DEFAULT_DOTENV_FILENAME = ".env"
DEFAULT_BODY_CONTEXT_LINES = 20
DEFAULT_RETRIEVAL_MODE = "mcp"
RETRIEVAL_MODES = ("mcp", "web-search", "none")
DEFAULT_LANGFUSE_TRACE_NAME = "tex-bib-mcp-query"
DEFAULT_LANGFUSE_PUBLIC_KEY_ENV = "LANGFUSE_PUBLIC_KEY"
DEFAULT_LANGFUSE_SECRET_KEY_ENV = "LANGFUSE_SECRET_KEY"
DEFAULT_OPENAI_INPUT_COST_ENV = "OPENAI_INPUT_COST_PER_1M"
DEFAULT_OPENAI_CACHED_INPUT_COST_ENV = "OPENAI_CACHED_INPUT_COST_PER_1M"
DEFAULT_OPENAI_OUTPUT_COST_ENV = "OPENAI_OUTPUT_COST_PER_1M"
AUDIT_LOG_ENTRY_RE = re.compile(r"^\[(\d+)\]\s+")
WHOLEBODY_LOG_ENTRY_RE = re.compile(r"^\[(\d+)\]\s+Line:\s+(\d+)$")
STATEMENT_LOG_LINE_RE = re.compile(r"^Line:\s+(\d+)$")
ARXIV_CITATION_ENTRY_RE = re.compile(r"([^,\[\]]+?)\s*\[([^\]]+)\]")
CITATION_LOCATOR_RE = re.compile(
    r"\b(?:theorems?|thms?\.?|thm\.?|lemmas?|lems?\.?|lem\.?|"
    r"corollaries|cors?\.?|cor\.?|propositions?|props?\.?|prop\.?|"
    r"claims?|conjectures?|definitions?|defs?\.?|defn\.?)\s*"
    r"[A-Za-z0-9][A-Za-z0-9().:-]*",
    re.IGNORECASE,
)
LOCATOR_PREFIX_RE = re.compile(
    r"^(theorems?|thms?\.?|thm\.?|lemmas?|lems?\.?|lem\.?|"
    r"corollaries|cors?\.?|cor\.?|propositions?|props?\.?|prop\.?|"
    r"claims?|conjectures?|definitions?|defs?\.?|defn\.?)\s*(.+)$",
    re.IGNORECASE,
)
BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{\s*document\s*\}")
END_DOCUMENT_RE = re.compile(r"\\end\s*\{\s*document\s*\}")
BIBLIOGRAPHY_START_RE = re.compile(
    r"^\s*\\(?:bibliography\b|printbibliography\b|begin\s*\{\s*thebibliography\s*\})"
)
LOG_SEPARATOR_LINE = "-" * 80


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


@dataclass(frozen=True)
class AuditLogRecord:
    record_index: int
    line_number: int
    bib_keys: tuple[str, ...]
    line_text: str = ""
    original_statement: str = ""
    baseline_id: str = ""
    baseline_num: str = ""


@dataclass(frozen=True)
class AuditLogContext:
    log_path: Path
    tex_path: Path
    bibliography_paths: tuple[Path, ...]
    records: tuple[AuditLogRecord, ...]


@dataclass(frozen=True)
class PreparedUploadInputs:
    tex_path: Path
    bib_path: Path
    status_messages: tuple[str, ...]
    prompt_note: str | None = None
    audit_log_path: Path | None = None
    audit_record_index: int | None = None


def parse_nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a .tex file and a bibliography source to the OpenAI Responses API, "
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
            "External retrieval mode: mcp uses TheoremSearch, web-search uses the "
            "provider's web search tool, and none disables external retrieval. "
            f"Defaults to {DEFAULT_RETRIEVAL_MODE}."
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


def prepare_upload_inputs(
    tex_path: Path,
    bib_path: Path,
    audit_log_path: Path | None,
    log_entry_index: int | None,
    temp_dir: Path,
    body_context_lines: int = DEFAULT_BODY_CONTEXT_LINES,
) -> PreparedUploadInputs:
    status_messages: list[str] = []
    prompt_note: str | None = None
    effective_log_path = audit_log_path or find_associated_audit_log(tex_path)
    selected_record: AuditLogRecord | None = None
    bibliography_sources: list[Path] = [bib_path]

    upload_tex_path = tex_path
    if effective_log_path is not None:
        audit_context = parse_audit_log(effective_log_path)
        if not audit_log_matches_tex(audit_context.tex_path, tex_path):
            raise RuntimeError(
                f"Audit log {effective_log_path} points to {audit_context.tex_path}, not {tex_path}."
            )
        selected_record = select_audit_log_record(audit_context, log_entry_index)
        if audit_context.bibliography_paths:
            bibliography_sources = resolve_logged_bibliography_paths(
                logged_paths=audit_context.bibliography_paths,
                tex_path=tex_path,
                fallback_bib_path=bib_path,
            )
        upload_tex_path = write_sanitized_tex_upload(
            tex_path=tex_path,
            record=selected_record,
            temp_dir=temp_dir,
            body_context_lines=body_context_lines,
        )
        status_messages.append(
            f"Using audit log {effective_log_path.name} entry {selected_record.record_index} "
            f"at line {selected_record.line_number}; uploading masked TeX excerpt with "
            f"up to {body_context_lines} main-body line(s) on each side."
        )
        if selected_record.bib_keys:
            status_messages.append(
                "Removing logged bibliography key(s) from upload: "
                + ", ".join(selected_record.bib_keys)
            )
        prompt_note = (
            "The uploaded LaTeX source is a main-body excerpt from the associated "
            f"audit log containing up to {body_context_lines} line(s) before and after "
            f"line {selected_record.line_number}. The logged citation on that line has "
            "been replaced with [Citation Needed]. The bibliography upload is not "
            "line-windowed; the matching bibliography entry has been removed from the "
            "uploaded bibliography text when it could be identified."
        )
    elif bib_path.suffix.lower() != ".txt":
        status_messages.append(
            f"Converting {bib_path.name} to a temporary .txt bibliography upload for OpenAI."
        )

    upload_bib_path = build_bibliography_upload_file(
        paper_source=tex_path.stem,
        bib_paths=bibliography_sources,
        temp_dir=temp_dir,
        excluded_keys=selected_record.bib_keys if selected_record is not None else (),
    )
    return PreparedUploadInputs(
        tex_path=upload_tex_path,
        bib_path=upload_bib_path,
        status_messages=tuple(status_messages),
        prompt_note=prompt_note,
        audit_log_path=effective_log_path,
        audit_record_index=selected_record.record_index if selected_record is not None else None,
    )


def find_associated_audit_log(tex_path: Path) -> Path | None:
    candidate_dirs: list[Path] = []
    seen_dirs: set[Path] = set()
    search_roots = [tex_path.parent, *tex_path.parents, Path.cwd()]
    for root in search_roots:
        candidate_dir = root / "statement_reference_audit_logs"
        if candidate_dir in seen_dirs:
            continue
        seen_dirs.add(candidate_dir)
        if candidate_dir.exists() and candidate_dir.is_dir():
            candidate_dirs.append(candidate_dir)

    matches: list[Path] = []
    for candidate_dir in candidate_dirs:
        for log_path in sorted(candidate_dir.glob("*.log")):
            try:
                for line in read_candidate_text(log_path).splitlines():
                    if not line.startswith("Input: "):
                        continue
                    logged_tex_path = Path(line[len("Input: ") :].strip()).resolve()
                    if audit_log_matches_tex(logged_tex_path, tex_path):
                        matches.append(log_path.resolve())
                        break
            except OSError:
                continue

    if not matches:
        return None
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple associated audit logs matched {tex_path}. Pass --audit-log explicitly. Matches: {rendered}"
        )
    return matches[0]


def parse_audit_log(log_path: Path) -> AuditLogContext:
    lines = read_candidate_text(log_path).splitlines()
    input_line = next((line for line in lines if line.startswith("Input: ")), None)
    bibliography_line = next((line for line in lines if line.startswith("Bibliography files: ")), None)
    if input_line is None:
        raise RuntimeError(f"Missing Input header in {log_path}")
    if bibliography_line is None:
        raise RuntimeError(f"Missing Bibliography files header in {log_path}")

    tex_path = Path(input_line[len("Input: ") :].strip()).resolve()
    bibliography_value = bibliography_line[len("Bibliography files: ") :].strip()
    bibliography_paths: list[Path] = []
    if bibliography_value != "(none found)":
        bibliography_paths = [
            Path(part.strip()).resolve()
            for part in bibliography_value.split(", ")
            if part.strip()
        ]

    if lines and lines[0].startswith("Whole-body citation audit"):
        records = parse_wholebody_audit_log_records(lines)
    else:
        records = parse_statement_audit_log_records(lines)
    return AuditLogContext(
        log_path=log_path.resolve(),
        tex_path=tex_path,
        bibliography_paths=tuple(bibliography_paths),
        records=tuple(records),
    )


def audit_log_matches_tex(logged_tex_path: Path, tex_path: Path) -> bool:
    if logged_tex_path == tex_path:
        return True
    return (
        logged_tex_path.name == tex_path.name
        and logged_tex_path.parent.name == tex_path.parent.name
    )


def resolve_logged_bibliography_paths(
    logged_paths: Sequence[Path],
    tex_path: Path,
    fallback_bib_path: Path,
) -> list[Path]:
    resolved_paths: list[Path] = []
    for logged_path in logged_paths:
        if logged_path.exists():
            resolved_paths.append(logged_path)
            continue
        same_name_in_tex_dir = (tex_path.parent / logged_path.name).resolve()
        if same_name_in_tex_dir.exists():
            resolved_paths.append(same_name_in_tex_dir)
            continue
        if fallback_bib_path.name == logged_path.name and fallback_bib_path.exists():
            resolved_paths.append(fallback_bib_path)

    if not resolved_paths:
        resolved_paths.append(fallback_bib_path)
    return [Path(path).resolve() for path in dedupe_preserving_order([str(path) for path in resolved_paths])]


def parse_statement_audit_log_records(lines: Sequence[str]) -> list[AuditLogRecord]:
    records: list[AuditLogRecord] = []
    index = 0
    while index < len(lines):
        header_match = AUDIT_LOG_ENTRY_RE.match(lines[index].strip())
        if not header_match:
            index += 1
            continue

        record_index = int(header_match.group(1))
        index += 1
        line_number: int | None = None
        bib_keys: list[str] = []
        arxiv_ids: list[str] = []
        arxiv_ids_by_key: dict[str, list[str]] = {}
        original_statement_lines: list[str] = []

        while index < len(lines) and lines[index].strip() != LOG_SEPARATOR_LINE:
            line = lines[index]
            line_match = STATEMENT_LOG_LINE_RE.match(line.strip())
            if line_match:
                line_number = int(line_match.group(1))
                index += 1
                continue
            if line == "ArXiv citations:":
                index += 1
                while index < len(lines) and lines[index].startswith("  - "):
                    for key, entry_arxiv_ids in extract_arxiv_citation_entries_from_log_line(
                        lines[index]
                    ):
                        bib_keys.append(key)
                        arxiv_ids.extend(entry_arxiv_ids)
                        arxiv_ids_by_key.setdefault(key, []).extend(entry_arxiv_ids)
                    index += 1
                continue
            if line == "Original statement:":
                index += 1
                while index < len(lines) and lines[index].strip() != LOG_SEPARATOR_LINE:
                    original_statement_lines.append(lines[index])
                    index += 1
                continue
            index += 1

        if line_number is None:
            raise RuntimeError(f"Missing line number in audit log entry [{record_index}]")

        original_statement = "\n".join(original_statement_lines).strip()
        baseline_num, locator_keys = extract_baseline_locator_and_keys_from_statement(
            original_statement,
            bib_keys,
        )
        baseline_id = first_arxiv_id_for_keys(locator_keys, arxiv_ids_by_key)
        if not baseline_id:
            baseline_id = first_or_empty(dedupe_preserving_order(arxiv_ids))

        records.append(
            AuditLogRecord(
                record_index=record_index,
                line_number=line_number,
                bib_keys=tuple(dedupe_preserving_order(bib_keys)),
                original_statement=original_statement,
                baseline_id=baseline_id,
                baseline_num=baseline_num,
            )
        )
        index += 1
    return records


def parse_wholebody_audit_log_records(lines: Sequence[str]) -> list[AuditLogRecord]:
    records: list[AuditLogRecord] = []
    index = 0
    while index < len(lines):
        match = WHOLEBODY_LOG_ENTRY_RE.match(lines[index].strip())
        if not match:
            index += 1
            continue

        record_index = int(match.group(1))
        line_number = int(match.group(2))
        cursor = index + 1
        baseline_num = ""
        baseline_id = ""
        if cursor < len(lines) and lines[cursor].startswith("Name: "):
            baseline_num = lines[cursor][len("Name: ") :].strip()
            cursor += 1
        if cursor < len(lines) and lines[cursor].startswith("External Source: "):
            baseline_id = lines[cursor][len("External Source: ") :].strip()
            cursor += 1
        line_text = lines[cursor].rstrip() if cursor < len(lines) else ""
        bib_keys = extract_citation_keys_from_text(line_text)
        records.append(
            AuditLogRecord(
                record_index=record_index,
                line_number=line_number,
                bib_keys=tuple(bib_keys),
                line_text=line_text,
                baseline_id=baseline_id,
                baseline_num=baseline_num,
            )
        )
        index += 1
        while index < len(lines) and lines[index].strip() != LOG_SEPARATOR_LINE:
            index += 1
        index += 1
    return records


def select_audit_log_record(
    audit_context: AuditLogContext,
    log_entry_index: int | None,
) -> AuditLogRecord:
    if not audit_context.records:
        raise RuntimeError(f"No audit log entries found in {audit_context.log_path}")
    if log_entry_index is None:
        if len(audit_context.records) == 1:
            return audit_context.records[0]
        available = ", ".join(str(record.record_index) for record in audit_context.records)
        raise RuntimeError(
            f"Audit log {audit_context.log_path.name} contains multiple entries. "
            f"Pass --log-entry to pick one of: {available}"
        )

    for record in audit_context.records:
        if record.record_index == log_entry_index:
            return record
    available = ", ".join(str(record.record_index) for record in audit_context.records)
    raise RuntimeError(
        f"Audit log entry {log_entry_index} was not found in {audit_context.log_path.name}. "
        f"Available entries: {available}"
    )


def write_sanitized_tex_upload(
    tex_path: Path,
    record: AuditLogRecord,
    temp_dir: Path,
    body_context_lines: int = DEFAULT_BODY_CONTEXT_LINES,
) -> Path:
    if body_context_lines < 0:
        raise RuntimeError(
            f"body_context_lines must be non-negative, got {body_context_lines}"
        )
    tex_lines = read_candidate_text(tex_path).splitlines(keepends=True)
    if record.line_number < 1:
        raise RuntimeError(f"Invalid audit log line number {record.line_number} for {tex_path}")
    if record.line_number > len(tex_lines):
        raise RuntimeError(
            f"Audit log line {record.line_number} exceeds the TeX length {len(tex_lines)} for {tex_path}"
        )

    target_index = record.line_number - 1
    body_start_index, body_end_index = find_main_body_line_bounds(tex_lines)
    start_index = max(0, target_index - body_context_lines)
    end_index = min(len(tex_lines), target_index + body_context_lines + 1)
    if body_start_index <= target_index < body_end_index:
        start_index = max(start_index, body_start_index)
        end_index = min(end_index, body_end_index)

    window_lines = list(tex_lines[start_index:end_index])
    target_window_index = target_index - start_index
    window_lines[target_window_index] = mask_logged_citations_in_line(
        window_lines[target_window_index],
        record.bib_keys,
    )
    start_line = start_index + 1
    end_line = end_index
    header_lines = [
        f"% Excerpt from {tex_path.name}; original lines {start_line}-{end_line}.\n",
        (
            f"% Target citation line: {record.line_number}; "
            f"{body_context_lines} context line(s) requested on each side.\n\n"
        ),
    ]
    output_path = (
        temp_dir
        / f"{tex_path.stem}_log_{record.record_index}_lines_{start_line}_{end_line}.tex"
    )
    output_path.write_text("".join(header_lines + window_lines), encoding="utf-8")
    return output_path


def find_main_body_line_bounds(tex_lines: Sequence[str]) -> tuple[int, int]:
    body_start_index = 0
    for index, line in enumerate(tex_lines):
        if BEGIN_DOCUMENT_RE.search(line):
            body_start_index = index + 1
            break

    body_end_index = len(tex_lines)
    for index in range(body_start_index, len(tex_lines)):
        line = tex_lines[index]
        if END_DOCUMENT_RE.search(line) or BIBLIOGRAPHY_START_RE.search(line):
            body_end_index = index
            break
    return body_start_index, body_end_index


def mask_logged_citations_in_line(line_text: str, bib_keys: Sequence[str]) -> str:
    occurrences = list(find_macro_occurrences(line_text, set(DEFAULT_CITE_COMMANDS)))
    if not occurrences:
        return line_text

    key_set = {key for key in bib_keys if key}
    target_occurrences = [
        occurrence
        for occurrence in occurrences
        if not key_set or any(key in key_set for key in occurrence.keys)
    ]
    if not target_occurrences:
        return line_text

    chunks: list[str] = []
    cursor = 0
    for occurrence in target_occurrences:
        chunks.append(line_text[cursor : occurrence.start])
        chunks.append("[Citation Needed]")
        cursor = occurrence.end
    chunks.append(line_text[cursor:])
    return "".join(chunks)


def build_bibliography_upload_file(
    paper_source: str,
    bib_paths: Sequence[Path],
    temp_dir: Path,
    excluded_keys: Sequence[str] = (),
) -> Path:
    output_path = temp_dir / f"{sanitize_for_filename(paper_source)}_bibliography.txt"
    parts: list[str] = []
    if not bib_paths:
        parts.append("(no bibliography files provided)\n")
    for bib_path in bib_paths:
        parts.append(f"===== {bib_path.name} =====\n")
        content = read_candidate_text(bib_path)
        filtered_content = filter_bibliography_text(content, excluded_keys)
        if filtered_content:
            parts.append(filtered_content)
            if not filtered_content.endswith("\n"):
                parts.append("\n")
        parts.append("\n")
    output_path.write_text("".join(parts), encoding="utf-8")
    return output_path


def filter_bibliography_text(text: str, excluded_keys: Sequence[str]) -> str:
    key_set = {key.strip() for key in excluded_keys if key.strip()}
    if not key_set:
        return text

    bib_entries = list(iter_bib_entries(text))
    if bib_entries:
        kept_entries = [
            raw.rstrip()
            for key, _, raw in bib_entries
            if key not in key_set
        ]
        return ("\n\n".join(kept_entries) + "\n") if kept_entries else ""

    bbl_entries = list(iter_bbl_entries(text))
    if bbl_entries:
        kept_entries = [
            raw.rstrip()
            for key, _, raw in bbl_entries
            if key not in key_set
        ]
        return ("\n\n".join(kept_entries) + "\n") if kept_entries else ""

    return text


def extract_bib_keys_from_log_line(line: str) -> list[str]:
    return [
        key
        for key, _arxiv_ids in extract_arxiv_citation_entries_from_log_line(line)
    ]


def extract_arxiv_ids_from_log_line(line: str) -> list[str]:
    arxiv_ids: list[str] = []
    for _key, entry_arxiv_ids in extract_arxiv_citation_entries_from_log_line(line):
        arxiv_ids.extend(entry_arxiv_ids)
    return dedupe_preserving_order(arxiv_ids)


def extract_arxiv_citation_entries_from_log_line(line: str) -> list[tuple[str, list[str]]]:
    _, _, payload = line.partition("->")
    if not payload:
        return []

    entries: list[tuple[str, list[str]]] = []
    for match in ARXIV_CITATION_ENTRY_RE.finditer(payload):
        key = match.group(1).strip()
        if not key:
            continue
        arxiv_ids: list[str] = []
        for raw_id in match.group(2).split(","):
            cleaned = raw_id.strip()
            if cleaned:
                arxiv_ids.append(normalize_arxiv_id(cleaned))
        entries.append((key, dedupe_preserving_order(arxiv_ids)))
    return entries


def extract_baseline_locator_from_statement(statement: str, bib_keys: Sequence[str]) -> str:
    locator, _keys = extract_baseline_locator_and_keys_from_statement(statement, bib_keys)
    return locator


def extract_baseline_locator_and_keys_from_statement(
    statement: str,
    bib_keys: Sequence[str],
) -> tuple[str, list[str]]:
    if not statement.strip():
        return "", []

    key_set = {key for key in bib_keys if key}
    occurrences = list(find_macro_occurrences(statement, set(DEFAULT_CITE_COMMANDS)))
    matching_occurrences = [
        occurrence
        for occurrence in occurrences
        if not key_set or any(key in key_set for key in occurrence.keys)
    ]
    if not matching_occurrences:
        matching_occurrences = occurrences

    for occurrence in matching_occurrences:
        locator = extract_locator_from_citation_occurrence(occurrence)
        if locator:
            return locator, list(occurrence.keys)
    return "", []


def first_arxiv_id_for_keys(
    keys: Sequence[str],
    arxiv_ids_by_key: dict[str, list[str]],
) -> str:
    for key in keys:
        for arxiv_id in arxiv_ids_by_key.get(key, []):
            if arxiv_id:
                return arxiv_id
    return ""


def extract_locator_from_citation_occurrence(occurrence) -> str:
    candidates: list[str] = []
    if occurrence.postnote:
        candidates.append(occurrence.postnote)
    candidates.extend(reversed(occurrence.optional_args))

    for candidate in candidates:
        locator = extract_locator_from_text(candidate)
        if locator:
            return locator
    return ""


def extract_locator_from_text(text: str) -> str:
    cleaned = clean_tex_locator_text(text)
    match = CITATION_LOCATOR_RE.search(cleaned)
    if not match:
        return ""
    return canonicalize_statement_locator(match.group(0))


def clean_tex_locator_text(text: str) -> str:
    cleaned = text.replace("~", " ")
    cleaned = re.sub(r"\\[,;:! ]", " ", cleaned)
    cleaned = cleaned.replace("{", " ").replace("}", " ").replace("$", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def canonicalize_statement_locator(text: str) -> str:
    cleaned = clean_tex_locator_text(text)
    cleaned = re.sub(r"[,:;]+$", "", cleaned).strip()
    match = LOCATOR_PREFIX_RE.match(cleaned)
    if not match:
        return cleaned

    raw_prefix = match.group(1).rstrip(".").lower()
    suffix = match.group(2).strip()
    display_prefix = {
        "theorem": "Theorem",
        "theorems": "Theorem",
        "thm": "Theorem",
        "thms": "Theorem",
        "lemma": "Lemma",
        "lemmas": "Lemma",
        "lem": "Lemma",
        "lems": "Lemma",
        "corollary": "Corollary",
        "corollaries": "Corollary",
        "cor": "Corollary",
        "cors": "Corollary",
        "proposition": "Proposition",
        "propositions": "Proposition",
        "prop": "Proposition",
        "props": "Proposition",
        "claim": "Claim",
        "claims": "Claim",
        "conjecture": "Conjecture",
        "conjectures": "Conjecture",
        "definition": "Definition",
        "definitions": "Definition",
        "def": "Definition",
        "defs": "Definition",
        "defn": "Definition",
    }.get(raw_prefix, match.group(1).rstrip("."))
    return f"{display_prefix} {suffix}".strip()


def extract_citation_keys_from_text(text: str) -> list[str]:
    keys: list[str] = []
    for occurrence in find_macro_occurrences(text, set(DEFAULT_CITE_COMMANDS)):
        keys.extend(occurrence.keys)
    return dedupe_preserving_order(keys)


def dedupe_preserving_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def first_or_empty(values: Sequence[str]) -> str:
    return values[0] if values else ""


def sanitize_for_filename(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in {".", "-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned) or "paper"


def create_upload_temp_dir() -> Path:
    base_dir = (Path.cwd() / ".openai_upload_tmp").resolve()
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
                    mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
                    retrieval_mode=args.retrieval_mode,
                ),
                metadata=build_langfuse_observation_metadata(
                    tex_path=tex_path,
                    bib_path=bib_path,
                    mcp_url=args.mcp_url,
                    mcp_label=args.mcp_label,
                    mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
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
                            retrieval_mode=args.retrieval_mode,
                        ),
                        metadata=build_langfuse_generation_metadata(
                            tex_path=tex_path,
                            bib_path=bib_path,
                            mcp_url=args.mcp_url,
                            mcp_label=args.mcp_label,
                            mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
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
                    response, response_data = run_response(
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
                        tex_upload_id=required_file_id(tex_upload, upload_inputs.tex_path),
                        bib_upload_id=required_file_id(bib_upload, upload_inputs.bib_path),
                        mcp_tool=(
                            build_mcp_tool(
                                mcp_label=args.mcp_label,
                                mcp_url=args.mcp_url,
                                mcp_tools=args.mcp_tools or list(DEFAULT_ALLOWED_MCP_TOOLS),
                                mcp_headers=mcp_headers,
                            )
                            if args.retrieval_mode == "mcp"
                            else None
                        ),
                        retrieval_mode=args.retrieval_mode,
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
        cleanup_upload_temp_dir(upload_temp_dir)
        if args.delete_uploads:
            cleanup_uploaded_files(client if "client" in locals() else None, uploaded_file_ids)
        if "langfuse_runtime" in locals() and langfuse_runtime.enabled:
            flush_langfuse(langfuse_runtime)

    if trace_url:
        print(f"Langfuse trace: {trace_url}", file=sys.stderr)

    return 0


def validate_input_path(
    path: Path,
    expected_suffix: str | Sequence[str],
) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise RuntimeError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise RuntimeError(f"Expected a file, but found: {resolved}")
    expected_suffixes = (
        tuple(item.lower() for item in expected_suffix)
        if isinstance(expected_suffix, (list, tuple))
        else (expected_suffix.lower(),)
    )
    if resolved.suffix.lower() not in expected_suffixes:
        expected_text = (
            expected_suffixes[0]
            if len(expected_suffixes) == 1
            else ", ".join(expected_suffixes)
        )
        raise RuntimeError(
            f"Expected one of {expected_text}, but got: {resolved.name}"
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


def build_user_prompt(
    tex_path: Path,
    bib_path: Path,
    user_prompt: str | None,
    context_note: str | None = None,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> str:
    retrieval_instruction = build_retrieval_instruction(retrieval_mode)
    instruction = user_prompt.strip() if user_prompt else (
        "Read the attached LaTeX source and bibliography, identify the missing "
        "citation, and return JSON only with exactly these keys: "
        '{"ai_id": <external arXiv identifier or null>, '
        '"ai_num": <cited theorem-like result name/number or null>}.'
    )
    context_block = f"{context_note.strip()}\n\n" if context_note and context_note.strip() else ""
    return (
        "You are given two uploaded files.\n"
        f"- {tex_path.name}: the LaTeX source file\n"
        f"- {bib_path.name}: the bibliography file\n\n"
        "Treat the uploaded files as the primary project context.\n"
        f"{context_block}"
        f"{retrieval_instruction}\n\n"
        "User request:\n"
        f"{instruction}"
    )


def build_retrieval_instruction(retrieval_mode: str) -> str:
    if retrieval_mode == "mcp":
        return (
            "Your job is to search the database for citations via the search tool. "
            "You can call it through the attached MCP server. Use theorem_search "
            "when it would improve factual or literature-grounded answers."
        )
    if retrieval_mode == "web-search":
        return (
            "Use web search when it would improve factual or literature-grounded "
            "answers. Do not use TheoremSearch or MCP tools in this mode."
        )
    return (
        "Do not use external retrieval tools. Answer only from the attached files "
        "and your model knowledge."
    )


def build_system_prompt(retrieval_mode: str = DEFAULT_RETRIEVAL_MODE) -> str:
    if retrieval_mode == "mcp":
        return (
            "You do not have access to web search in this session. "
            "The only external tool available is the attached MCP server, "
            "and the only allowed MCP tool is theorem_search. "
            "If you need external retrieval, use theorem_search only."
        )
    if retrieval_mode == "web-search":
        return (
            "You may use the provider web search tool for external retrieval. "
            "Do not use MCP or TheoremSearch tools in this session."
        )
    return (
        "You do not have access to external retrieval tools in this session. "
        "Use only the attached files and your model knowledge."
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
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> dict[str, Any]:
    return {
        "tex_file": str(tex_path),
        "bib_file": str(bib_path),
        "user_prompt": user_prompt or "(default prompt)",
        "model": model,
        "reasoning_effort": reasoning_effort,
        "retrieval_mode": retrieval_mode,
        "web_search_enabled": retrieval_mode == "web-search",
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
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> dict[str, Any]:
    return {
        "tex_filename": tex_path.name,
        "bib_filename": bib_path.name,
        "tool_policy": {
            "retrieval_mode": retrieval_mode,
            "web_search_enabled": retrieval_mode == "web-search",
            "mcp_enabled": retrieval_mode == "mcp",
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
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> dict[str, str]:
    return {
        "tex_file": tex_path.name,
        "bib_file": bib_path.name,
        "model": model,
        "retrieval_mode": retrieval_mode,
        "web_search": "enabled" if retrieval_mode == "web-search" else "disabled",
        "external_tool": "theorem_search" if retrieval_mode == "mcp" else retrieval_mode,
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
    retrieval_mode = getattr(args, "retrieval_mode", DEFAULT_RETRIEVAL_MODE)
    return {
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "tool_mode": retrieval_mode,
    }


def build_langfuse_generation_input(
    tex_path: Path,
    bib_path: Path,
    user_prompt: str | None,
    mcp_url: str,
    mcp_label: str,
    mcp_tools: list[str],
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> dict[str, Any]:
    return {
        "tex_file": str(tex_path),
        "bib_file": str(bib_path),
        "user_prompt": user_prompt or "(default prompt)",
        "tools": {
            "retrieval_mode": retrieval_mode,
            "web_search_enabled": retrieval_mode == "web-search",
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
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "tex_filename": tex_path.name,
        "bib_filename": bib_path.name,
        "tool_policy": {
            "retrieval_mode": retrieval_mode,
            "web_search_enabled": retrieval_mode == "web-search",
            "mcp_enabled": retrieval_mode == "mcp",
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


def build_web_search_tool() -> dict[str, Any]:
    return {
        "type": "web_search",
    }


def build_openai_tools(
    *,
    retrieval_mode: str,
    mcp_tool: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if retrieval_mode == "mcp":
        if mcp_tool is None:
            raise RuntimeError("retrieval-mode=mcp requires an MCP tool configuration.")
        return [mcp_tool]
    if retrieval_mode == "web-search":
        return [build_web_search_tool()]
    return []


def run_response(
    client,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    prompt: str,
    tex_upload_id: str,
    bib_upload_id: str,
    mcp_tool: dict[str, Any] | None,
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE,
):
    tools = build_openai_tools(retrieval_mode=retrieval_mode, mcp_tool=mcp_tool)
    request: dict[str, Any] = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": max_output_tokens,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_system_prompt(retrieval_mode),
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
    }
    if tools:
        request["tools"] = tools
    response = client.responses.create(**request)

    response_data = to_dict(response)
    previous_response_id = response_data.get("id")
    if retrieval_mode != "mcp" or not previous_response_id:
        return response, response_data

    for _ in range(DEFAULT_APPROVAL_LIMIT):
        approval_requests = extract_mcp_approval_requests(response_data)
        if not approval_requests:
            return response, response_data

        followup_request: dict[str, Any] = {
            "model": model,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "tools": tools,
            "previous_response_id": previous_response_id,
            "input": [
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": request["id"],
                    "approve": True,
                    "reason": "Approved by local CLI script.",
                }
                for request in approval_requests
            ],
        }
        response = client.responses.create(**followup_request)
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
