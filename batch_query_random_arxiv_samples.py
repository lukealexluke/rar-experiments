#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from query_openai_tex_mcp import (
    DEFAULT_ALLOWED_MCP_TOOLS,
    DEFAULT_API_KEY_ENV,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MCP_LABEL,
    DEFAULT_MCP_URL,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    build_mcp_tool,
    cleanup_uploaded_files,
    extract_output_text,
    load_environment_from_dotenv,
    make_client,
    parse_mcp_headers,
    read_api_key,
    required_file_id,
    run_response,
    upload_file,
)
from statement_reference_audit_wholebody import (
    DEFAULT_CITE_COMMANDS,
    STATEMENT_LOCATOR_RE,
    citation_mentions_statement_locator,
    find_macro_occurrences,
    load_bibliography,
    normalize_arxiv_id,
    read_candidate_text,
)

CSV_COLUMNS = ("paper_source", "name", "ext_source", "gpt_name", "gpt_ext_source")
LOG_ENTRY_RE = re.compile(r"^\[(\d+)\] Line: (\d+)$")
CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


@dataclass
class LogRecord:
    record_index: int
    line_number: int
    name: str
    ext_source: str
    line_text: str


@dataclass
class PaperContext:
    paper_source: str
    paper_dir: Path
    tex_path: Path
    bib_paths: list[Path]
    log_path: Path
    tex_lines: list[str]
    records: list[LogRecord]


@dataclass
class EvaluationItem:
    paper_source: str
    name: str
    ext_source: str
    record_index: int
    line_number: int
    line_text: str
    line_context: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Iterate over random_arxiv_source_samples, query GPT for each logged theorem-like "
            "citation item, and write a CSV with local and GPT-extracted labels."
        )
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("random_arxiv_source_samples"),
        help="Root directory containing paper_N folders and statement_reference_audit_logs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Output CSV path. Defaults to <samples-root>/gpt_citation_outputs.csv.",
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
        "--delete-uploads",
        action="store_true",
        help="Delete uploaded OpenAI files after each paper finishes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Optional cap on the total number of log items to query.",
    )
    return parser.parse_args()


def parse_log_file(log_path: Path) -> PaperContext:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    input_line = next((line for line in lines if line.startswith("Input: ")), None)
    bibliography_line = next((line for line in lines if line.startswith("Bibliography files: ")), None)
    if input_line is None:
        raise RuntimeError(f"Missing Input header in {log_path}")
    if bibliography_line is None:
        raise RuntimeError(f"Missing Bibliography files header in {log_path}")

    tex_path = Path(input_line[len("Input: ") :].strip()).resolve()
    if not tex_path.exists():
        raise RuntimeError(f"TeX file from log does not exist: {tex_path}")

    bib_value = bibliography_line[len("Bibliography files: ") :].strip()
    if bib_value == "(none found)":
        bib_paths: list[Path] = []
    else:
        bib_paths = [Path(part.strip()).resolve() for part in bib_value.split(", ") if part.strip()]

    records: list[LogRecord] = []
    index = 0
    while index < len(lines):
        match = LOG_ENTRY_RE.match(lines[index].strip())
        if not match:
            index += 1
            continue
        record_index = int(match.group(1))
        line_number = int(match.group(2))
        cursor = index + 1
        name = ""
        ext_source = ""
        if cursor < len(lines) and lines[cursor].startswith("Name: "):
            name = lines[cursor][len("Name: ") :].strip()
            cursor += 1
        if cursor < len(lines) and lines[cursor].startswith("External Source: "):
            ext_source = lines[cursor][len("External Source: ") :].strip()
            cursor += 1
        line_text = lines[cursor].rstrip() if cursor < len(lines) else ""
        records.append(
            LogRecord(
                record_index=record_index,
                line_number=line_number,
                name=name,
                ext_source=ext_source,
                line_text=line_text,
            )
        )
        index = cursor + 1

    paper_dir = tex_path.parent.resolve()
    paper_source = infer_paper_source(paper_dir)
    tex_lines = read_candidate_text(tex_path).splitlines()
    return PaperContext(
        paper_source=paper_source,
        paper_dir=paper_dir,
        tex_path=tex_path,
        bib_paths=bib_paths,
        log_path=log_path.resolve(),
        tex_lines=tex_lines,
        records=records,
    )


def infer_paper_source(paper_dir: Path) -> str:
    metadata_candidates = (
        paper_dir / "paper_metadata.json",
        paper_dir / "metadata.json",
        paper_dir / "source_metadata.json",
        paper_dir / "paper_source.txt",
        paper_dir / "original_arxiv_id.txt",
    )
    for candidate in metadata_candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".json":
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key in ("paper_source", "resolved_id", "source_id", "arxiv_id", "arxiv"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        else:
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                return value
    return paper_dir.name


def build_evaluation_items(context: PaperContext) -> list[EvaluationItem]:
    bibliography = load_bibliography(context.bib_paths)
    cite_commands = set(DEFAULT_CITE_COMMANDS)
    items: list[EvaluationItem] = []
    for record in context.records:
        statement_name = record.name
        ext_source = normalize_arxiv_id(record.ext_source) if record.ext_source else ""

        if not statement_name or not ext_source:
            citations = [
                occurrence
                for occurrence in find_macro_occurrences(record.line_text, cite_commands)
                if citation_mentions_statement_locator(record.line_text, occurrence)
            ]
            if not citations:
                citations = list(find_macro_occurrences(record.line_text, cite_commands))
            if not citations:
                print(
                    f"Warning: no citation macro found for {context.log_path.name} item {record.record_index}",
                    file=sys.stderr,
                )
                continue

            citation = citations[0]
            if not ext_source:
                ext_source = select_first_arxiv_id(citation.keys, bibliography)
            if not statement_name:
                statement_name = extract_statement_name(record.line_text, citation)
            if not ext_source:
                print(
                    f"Warning: no arXiv id found for {context.log_path.name} item {record.record_index}",
                    file=sys.stderr,
                )
                continue
            if not statement_name:
                print(
                    f"Warning: no theorem-like locator found for {context.log_path.name} item {record.record_index}",
                    file=sys.stderr,
                )
                continue

        items.append(
            EvaluationItem(
                paper_source=context.paper_source,
                name=statement_name,
                ext_source=ext_source,
                record_index=record.record_index,
                line_number=record.line_number,
                line_text=record.line_text,
                line_context=extract_line_context(context.tex_lines, record.line_number),
            )
        )
    return items


def select_first_arxiv_id(keys: list[str], bibliography: dict[str, object]) -> str:
    for key in keys:
        bib_entry = bibliography.get(key)
        arxiv_ids = getattr(bib_entry, "arxiv_ids", None)
        if not arxiv_ids:
            continue
        return normalize_arxiv_id(str(arxiv_ids[0]))
    return ""


def extract_statement_name(line_text: str, citation) -> str:
    candidate_chunks = list(citation.optional_args)
    if citation.postnote:
        candidate_chunks.append(citation.postnote)
    candidate_chunks.append(line_text)

    for chunk in candidate_chunks:
        locator = first_statement_locator(chunk)
        if locator:
            return locator
    return ""


def first_statement_locator(text: str) -> str:
    normalized = normalize_locator_text(text)
    match = STATEMENT_LOCATOR_RE.search(normalized)
    return match.group(0).strip() if match else ""


def normalize_locator_text(text: str) -> str:
    normalized = text.replace("~", " ").replace(r"\ ", " ")
    normalized = re.sub(r"\\[,:;!]", " ", normalized)
    normalized = normalized.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def extract_line_context(tex_lines: list[str], target_line_number: int, radius: int = 2) -> str:
    if not tex_lines:
        return ""
    start = max(1, target_line_number - radius)
    end = min(len(tex_lines), target_line_number + radius)
    context_lines = [
        f"{line_number}: {tex_lines[line_number - 1]}"
        for line_number in range(start, end + 1)
    ]
    return "\n".join(context_lines)


def build_bibliography_upload_file(paper_source: str, bib_paths: list[Path], temp_dir: Path) -> Path:
    output_path = temp_dir / f"{sanitize_for_filename(paper_source)}_bibliography.txt"
    parts: list[str] = []
    for bib_path in bib_paths:
        parts.append(f"===== {bib_path.name} =====\n")
        content = read_candidate_text(bib_path)
        parts.append(content)
        if not content.endswith("\n"):
            parts.append("\n")
        parts.append("\n")
    output_path.write_text("".join(parts), encoding="utf-8")
    return output_path


def sanitize_for_filename(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in {".", "-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned) or "paper"


def build_item_prompt(item: EvaluationItem, log_name: str) -> str:
    return (
        "Analyze one logged theorem-style citation occurrence in the attached LaTeX source and bibliography.\n\n"
        f"Paper identifier: {item.paper_source}\n"
        f"Log file: {log_name}\n"
        f"Log item: {item.record_index}\n"
        f"Logged source line number: {item.line_number}\n"
        "Logged line text:\n"
        f"{item.line_text}\n\n"
        "Nearby source context:\n"
        "```tex\n"
        f"{item.line_context}\n"
        "```\n\n"
        "The relevant citation has been masked as [Citation Needed] in the saved TeX source.\n"
        "Focus on the theorem-like result referenced by that masked citation on the logged line.\n\n"
        "Task:\n"
        '1. Identify the cited theorem-like result name/number, for example "Theorem 1.2" or "Proposition 3.4".\n'
        "2. Identify the external arXiv identifier of the cited source paper containing that result.\n"
        "Use theorem_search if it helps.\n\n"
        "Return JSON only, with exactly this shape:\n"
        '{"gpt_name": <string or null>, "gpt_ext_source": <string or null>}'
    )


def parse_gpt_response(output_text: str) -> tuple[str, str]:
    text = output_text.strip()
    if not text:
        return "", ""

    fenced_match = CODE_FENCE_RE.match(text)
    if fenced_match:
        text = fenced_match.group(1).strip()

    data = try_parse_json_object(text)
    if not isinstance(data, dict):
        return "", ""

    name_value = first_present_value(data, ("gpt_name", "name"))
    ext_source_value = first_present_value(data, ("gpt_ext_source", "ext_source"))
    gpt_name = normalize_optional_string(name_value)
    gpt_ext_source = normalize_optional_string(ext_source_value)
    if gpt_ext_source:
        gpt_ext_source = normalize_arxiv_id(gpt_ext_source)
    return gpt_name, gpt_ext_source


def try_parse_json_object(text: str) -> dict[str, object] | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"{", text):
        try:
            data, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def first_present_value(data: dict[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in data:
            return data[key]
    return None


def normalize_optional_string(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"null", "none", "unknown"}:
        return ""
    return cleaned


def sorted_log_paths(logs_dir: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"paper_(\d+)_wholebody\.log$", path.name)
        if match:
            return (int(match.group(1)), path.name)
        return (10**9, path.name)

    return sorted(logs_dir.glob("*.log"), key=sort_key)


def main() -> int:
    args = parse_args()
    samples_root = args.samples_root.resolve()
    if not samples_root.exists() or not samples_root.is_dir():
        print(f"Error: samples root does not exist: {samples_root}", file=sys.stderr)
        return 1
    if args.max_items is not None and args.max_items <= 0:
        print("Error: max-items must be positive when provided.", file=sys.stderr)
        return 1

    logs_dir = samples_root / "statement_reference_audit_logs"
    if not logs_dir.exists() or not logs_dir.is_dir():
        print(f"Error: logs directory does not exist: {logs_dir}", file=sys.stderr)
        return 1

    output_csv = (args.output_csv or (samples_root / "gpt_citation_outputs.csv")).resolve()
    if output_csv.exists() and not args.overwrite:
        print(
            f"Error: output CSV already exists: {output_csv}. Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 1

    try:
        dotenv_path = load_environment_from_dotenv()
        api_key = read_api_key(args.api_key_env)
        client = make_client(api_key, enable_langfuse=False)
        mcp_headers = parse_mcp_headers(args.mcp_headers or [])
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if dotenv_path:
        print(f"Loaded environment from {dotenv_path}", file=sys.stderr)

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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processed_items = 0
    queried_items = 0
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()

        for log_path in log_paths:
            if args.max_items is not None and queried_items >= args.max_items:
                break

            try:
                context = parse_log_file(log_path)
                items = build_evaluation_items(context)
            except RuntimeError as exc:
                print(f"Warning: skipping {log_path.name}: {exc}", file=sys.stderr)
                continue

            if not items:
                print(f"Skipping {log_path.name}: no evaluation items parsed.", file=sys.stderr)
                continue

            with tempfile.TemporaryDirectory(prefix="tex-mcp-batch-") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                bibliography_upload_path = build_bibliography_upload_file(
                    context.paper_source,
                    context.bib_paths,
                    temp_dir,
                )

                uploaded_file_ids: list[str] = []
                try:
                    tex_upload = upload_file(client, context.tex_path)
                    bib_upload = upload_file(client, bibliography_upload_path)
                    uploaded_file_ids.extend(
                        [
                            getattr(tex_upload, "id", ""),
                            getattr(bib_upload, "id", ""),
                        ]
                    )
                    tex_upload_id = required_file_id(tex_upload, context.tex_path)
                    bib_upload_id = required_file_id(bib_upload, bibliography_upload_path)

                    print(
                        f"Processing {context.paper_source} from {log_path.name} with {len(items)} item(s)...",
                        file=sys.stderr,
                    )

                    for item in items:
                        if args.max_items is not None and queried_items >= args.max_items:
                            break

                        prompt = build_item_prompt(item, log_path.name)
                        gpt_name = ""
                        gpt_ext_source = ""
                        try:
                            response, response_data = run_response(
                                client=client,
                                model=args.model,
                                reasoning_effort=args.reasoning_effort,
                                max_output_tokens=args.max_output_tokens,
                                prompt=prompt,
                                tex_upload_id=tex_upload_id,
                                bib_upload_id=bib_upload_id,
                                mcp_tool=mcp_tool,
                            )
                            output_text = extract_output_text(response, response_data)
                            gpt_name, gpt_ext_source = parse_gpt_response(output_text)
                        except Exception as exc:
                            print(
                                f"Warning: GPT query failed for {context.paper_source} item {item.record_index}: {exc}",
                                file=sys.stderr,
                            )

                        writer.writerow(
                            {
                                "paper_source": item.paper_source,
                                "name": item.name,
                                "ext_source": item.ext_source,
                                "gpt_name": gpt_name,
                                "gpt_ext_source": gpt_ext_source,
                            }
                        )
                        handle.flush()
                        queried_items += 1
                        processed_items += 1
                finally:
                    if args.delete_uploads:
                        cleanup_uploaded_files(client, uploaded_file_ids)

    print(f"Wrote {processed_items} row(s) to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
