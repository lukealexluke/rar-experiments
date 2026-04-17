#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from statement_reference_audit_wholebody import normalize_arxiv_id


LOG_ENTRY_RE = re.compile(r"^\[(\d+)\]\s+Line:\s+(\d+)$")
DEFAULT_LOGS_DIR = Path("statement_reference_audit_logs")
FALLBACK_LOGS_DIRS = (
    Path("random_arxiv_source_samples") / "statement_reference_audit_logs",
)


@dataclass(frozen=True)
class GoldRecord:
    record_index: int
    line_number: int
    name: str
    ext_source: str
    line_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate batch_query_audit_logs JSONL output against the ground-truth "
            "Name and External Source fields stored in the audit logs."
        )
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=None,
        help=(
            "Path to the JSONL results file. Defaults to <logs-dir>/openai_query_results.jsonl."
        ),
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help=(
            "Directory containing the audit logs. Defaults to statement_reference_audit_logs "
            "and falls back to random_arxiv_source_samples/statement_reference_audit_logs."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Optional path to write the evaluation summary JSON.",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        help="Optional path to write per-item evaluation details as CSV.",
    )
    return parser.parse_args()


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


def resolve_results_jsonl(results_jsonl: Path | None, logs_dir: Path) -> Path:
    if results_jsonl is not None:
        resolved = results_jsonl.resolve()
    else:
        resolved = (logs_dir / "openai_query_results.jsonl").resolve()
    if not resolved.exists() or not resolved.is_file():
        raise RuntimeError(f"results JSONL does not exist: {resolved}")
    return resolved


def parse_log_file(log_path: Path) -> dict[int, GoldRecord]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    records: dict[int, GoldRecord] = {}
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
        records[record_index] = GoldRecord(
            record_index=record_index,
            line_number=line_number,
            name=name,
            ext_source=normalize_ext_source(ext_source),
            line_text=line_text,
        )
        index = cursor + 1
    return records


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"null", "none", "unknown"}:
        return ""
    return cleaned


def normalize_statement_name(value: object) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = text.replace("~", " ").replace("{", " ").replace("}", " ")
    text = re.sub(r"[,:;]+$", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def normalize_ext_source(value: object) -> str:
    text = normalize_text(value)
    return normalize_arxiv_id(text) if text else ""


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"Expected JSON object on line {line_number} of {path}")
            rows.append(row)
    return rows


def resolve_log_path(raw_log_file: object, logs_dir: Path) -> Path:
    log_file = normalize_text(raw_log_file)
    if not log_file:
        raise RuntimeError("Result row is missing log_file")

    direct_path = Path(log_file)
    if direct_path.exists():
        return direct_path.resolve()

    by_name = (logs_dir / direct_path.name).resolve()
    if by_name.exists():
        return by_name

    raise RuntimeError(f"Could not resolve log file from results row: {log_file}")


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def main() -> int:
    args = parse_args()

    try:
        logs_dir = resolve_logs_dir(args.logs_dir)
        results_jsonl = resolve_results_jsonl(args.results_jsonl, logs_dir)
        rows = load_jsonl(results_jsonl)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print(f"Error: no rows found in {results_jsonl}", file=sys.stderr)
        return 1

    gold_cache: dict[Path, dict[int, GoldRecord]] = {}
    detail_rows: list[dict[str, object]] = []

    total_items = 0
    completed_items = 0
    incomplete_items = 0
    error_items = 0
    name_gold_items = 0
    ext_gold_items = 0
    both_gold_items = 0
    name_match_items = 0
    ext_match_items = 0
    both_match_items = 0

    for row in rows:
        total_items += 1
        response_status = normalize_text(row.get("response_status"))
        status = normalize_text(row.get("status"))
        if response_status == "completed":
            completed_items += 1
        elif response_status:
            incomplete_items += 1
        if normalize_text(row.get("error")):
            error_items += 1

        try:
            log_path = resolve_log_path(row.get("log_file"), logs_dir)
            if log_path not in gold_cache:
                gold_cache[log_path] = parse_log_file(log_path)
            record_index = int(row.get("record_index"))
            gold_record = gold_cache[log_path].get(record_index)
            if gold_record is None:
                raise RuntimeError(
                    f"No matching record [{record_index}] in {log_path.name}"
                )
        except Exception as exc:
            detail_rows.append(
                {
                    "log_file": normalize_text(row.get("log_file")),
                    "record_index": row.get("record_index"),
                    "line_number": row.get("line_number"),
                    "status": status,
                    "response_status": response_status,
                    "gold_name": "",
                    "gold_ext_source": "",
                    "gpt_name": normalize_text(row.get("gpt_name")),
                    "gpt_ext_source": normalize_ext_source(row.get("gpt_ext_source")),
                    "name_match": "",
                    "ext_source_match": "",
                    "both_match": "",
                    "error": str(exc),
                }
            )
            continue

        gold_name = gold_record.name
        gold_ext_source = gold_record.ext_source
        gpt_name = normalize_text(row.get("gpt_name"))
        gpt_ext_source = normalize_ext_source(row.get("gpt_ext_source"))

        gold_name_norm = normalize_statement_name(gold_name)
        gold_ext_norm = normalize_ext_source(gold_ext_source)
        gpt_name_norm = normalize_statement_name(gpt_name)
        gpt_ext_norm = normalize_ext_source(gpt_ext_source)

        name_match = bool(gold_name_norm) and gold_name_norm == gpt_name_norm
        ext_match = bool(gold_ext_norm) and gold_ext_norm == gpt_ext_norm
        both_match = name_match and ext_match

        if gold_name_norm:
            name_gold_items += 1
            if name_match:
                name_match_items += 1
        if gold_ext_norm:
            ext_gold_items += 1
            if ext_match:
                ext_match_items += 1
        if gold_name_norm and gold_ext_norm:
            both_gold_items += 1
            if both_match:
                both_match_items += 1

        detail_rows.append(
            {
                "log_file": str(log_path),
                "record_index": gold_record.record_index,
                "line_number": gold_record.line_number,
                "status": status,
                "response_status": response_status,
                "gold_name": gold_name,
                "gold_ext_source": gold_ext_source,
                "gpt_name": gpt_name,
                "gpt_ext_source": gpt_ext_source,
                "name_match": name_match,
                "ext_source_match": ext_match,
                "both_match": both_match,
                "error": normalize_text(row.get("error")),
            }
        )

    summary = {
        "results_jsonl": str(results_jsonl),
        "logs_dir": str(logs_dir),
        "total_items": total_items,
        "completed_items": completed_items,
        "incomplete_items": incomplete_items,
        "error_items": error_items,
        "name_gold_items": name_gold_items,
        "ext_source_gold_items": ext_gold_items,
        "both_gold_items": both_gold_items,
        "name_match_items": name_match_items,
        "ext_source_match_items": ext_match_items,
        "both_match_items": both_match_items,
        "completion_rate": safe_ratio(completed_items, total_items),
        "name_accuracy": safe_ratio(name_match_items, name_gold_items),
        "ext_source_accuracy": safe_ratio(ext_match_items, ext_gold_items),
        "both_accuracy": safe_ratio(both_match_items, both_gold_items),
    }

    details_csv = args.details_csv or results_jsonl.with_name("openai_query_evaluation_details.csv")
    details_csv.parent.mkdir(parents=True, exist_ok=True)
    with details_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "log_file",
                "record_index",
                "line_number",
                "status",
                "response_status",
                "gold_name",
                "gold_ext_source",
                "gpt_name",
                "gpt_ext_source",
                "name_match",
                "ext_source_match",
                "both_match",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote per-item details to {details_csv}", file=sys.stderr)
    if args.summary_json:
        print(f"Wrote summary JSON to {args.summary_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
