#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from statement_reference_audit_wholebody import normalize_arxiv_id


DEFAULT_LOGS_DIR = Path("statement_reference_audit_logs")
FALLBACK_LOGS_DIRS = (
    Path("random_arxiv_source_samples") / "statement_reference_audit_logs",
)
RESULT_CSV_FIELDS = ("baseline_id", "baseline_num", "ai_id", "ai_num")
ARXIV_VERSION_SUFFIX_RE = re.compile(r"v\d+$", re.IGNORECASE)
STATEMENT_PREFIX_RE = re.compile(
    r"^(theorems?|thms?\.?|thm\.?|lemmas?|lems?\.?|lem\.?|"
    r"corollaries|cors?\.?|cor\.?|propositions?|props?\.?|prop\.?|"
    r"claims?|conjectures?|definitions?|defs?\.?|defn\.?)\s*(.+)$",
    re.IGNORECASE,
)
CANONICAL_STATEMENT_PREFIXES = {
    "theorem": "theorem",
    "theorems": "theorem",
    "thm": "theorem",
    "thms": "theorem",
    "lemma": "lemma",
    "lemmas": "lemma",
    "lem": "lemma",
    "lems": "lemma",
    "corollary": "corollary",
    "corollaries": "corollary",
    "cor": "corollary",
    "cors": "corollary",
    "proposition": "proposition",
    "propositions": "proposition",
    "prop": "proposition",
    "props": "proposition",
    "claim": "claim",
    "claims": "claim",
    "conjecture": "conjecture",
    "conjectures": "conjecture",
    "definition": "definition",
    "definitions": "definition",
    "def": "definition",
    "defs": "definition",
    "defn": "definition",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate batch_query_audit_logs CSV output using baseline_id/baseline_num "
            "and ai_id/ai_num columns."
        )
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help=(
            "Path to the CSV results file. Defaults to <logs-dir>/openai_query_results.csv."
        ),
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=None,
        help="Deprecated alias for --results-csv.",
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


def resolve_results_csv(results_csv: Path | None, results_jsonl: Path | None, logs_dir: Path) -> Path:
    requested_results = results_csv or results_jsonl
    if requested_results is not None:
        resolved = requested_results.resolve()
    else:
        resolved = (logs_dir / "openai_query_results.csv").resolve()
    if not resolved.exists() or not resolved.is_file():
        raise RuntimeError(f"results CSV does not exist: {resolved}")
    return resolved


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
    text = re.sub(r"\\[,;:! ]", " ", text)
    text = re.sub(r"[,:;]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    match = STATEMENT_PREFIX_RE.match(text)
    if not match:
        return text

    raw_prefix = match.group(1).rstrip(".").lower()
    canonical_prefix = CANONICAL_STATEMENT_PREFIXES.get(raw_prefix, raw_prefix)
    return f"{canonical_prefix} {match.group(2).strip()}"


def normalize_ext_source(value: object) -> str:
    text = normalize_text(value)
    return ARXIV_VERSION_SUFFIX_RE.sub("", normalize_arxiv_id(text)) if text else ""


def load_result_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = [field for field in RESULT_CSV_FIELDS if field not in (reader.fieldnames or [])]
        if missing_fields:
            raise RuntimeError(
                f"results CSV is missing required column(s): {', '.join(missing_fields)}"
            )
        rows = [dict(row) for row in reader]
    return rows


def safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def main() -> int:
    args = parse_args()

    try:
        logs_dir = resolve_logs_dir(args.logs_dir)
        results_csv = resolve_results_csv(args.results_csv, args.results_jsonl, logs_dir)
        rows = load_result_csv(results_csv)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not rows:
        print(f"Error: no rows found in {results_csv}", file=sys.stderr)
        return 1

    detail_rows: list[dict[str, object]] = []

    total_items = 0
    num_gold_items = 0
    id_gold_items = 0
    both_gold_items = 0
    num_match_items = 0
    id_match_items = 0
    both_match_items = 0

    for row in rows:
        total_items += 1
        baseline_num = normalize_text(row.get("baseline_num"))
        baseline_id = normalize_ext_source(row.get("baseline_id"))
        ai_num = normalize_text(row.get("ai_num"))
        ai_id = normalize_ext_source(row.get("ai_id"))

        baseline_num_norm = normalize_statement_name(baseline_num)
        baseline_id_norm = normalize_ext_source(baseline_id)
        ai_num_norm = normalize_statement_name(ai_num)
        ai_id_norm = normalize_ext_source(ai_id)

        num_match = bool(baseline_num_norm) and baseline_num_norm == ai_num_norm
        id_match = bool(baseline_id_norm) and baseline_id_norm == ai_id_norm
        both_match = num_match and id_match

        if baseline_num_norm:
            num_gold_items += 1
            if num_match:
                num_match_items += 1
        if baseline_id_norm:
            id_gold_items += 1
            if id_match:
                id_match_items += 1
        if baseline_num_norm and baseline_id_norm:
            both_gold_items += 1
            if both_match:
                both_match_items += 1

        detail_rows.append(
            {
                "baseline_id": baseline_id,
                "baseline_num": baseline_num,
                "ai_id": ai_id,
                "ai_num": ai_num,
                "id_match": id_match,
                "num_match": num_match,
                "both_match": both_match,
            }
        )

    summary = {
        "results_csv": str(results_csv),
        "logs_dir": str(logs_dir),
        "total_items": total_items,
        "baseline_num_items": num_gold_items,
        "baseline_id_items": id_gold_items,
        "both_gold_items": both_gold_items,
        "num_match_items": num_match_items,
        "id_match_items": id_match_items,
        "both_match_items": both_match_items,
        "num_accuracy": safe_ratio(num_match_items, num_gold_items),
        "id_accuracy": safe_ratio(id_match_items, id_gold_items),
        "both_accuracy": safe_ratio(both_match_items, both_gold_items),
    }

    details_csv = args.details_csv or results_csv.with_name("query_evaluation_details.csv")
    details_csv.parent.mkdir(parents=True, exist_ok=True)
    with details_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "baseline_id",
                "baseline_num",
                "ai_id",
                "ai_num",
                "id_match",
                "num_match",
                "both_match",
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
