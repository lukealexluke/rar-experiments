#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence


DEFAULT_ENVIRONMENTS = ("theorem", "lemma", "corollary", "prop", "proposition")
DEFAULT_REF_COMMANDS = (
    "ref",
    "eqref",
    "autoref",
    "pageref",
    "nameref",
    "vref",
    "Vref",
    "cref",
    "Cref",
    "cpageref",
    "Cpageref",
    "labelcref",
    "labelcpageref",
    "namecref",
    "Namecref",
    "thmref",
    "lemref",
    "corref",
    "propref",
    "secref",
    "appref",
)
DEFAULT_CITE_COMMANDS = (
    "cite",
    "Cite",
    "citet",
    "Citet",
    "citep",
    "Citep",
    "citealp",
    "Citealp",
    "citealt",
    "Citealt",
    "citeauthor",
    "citeyear",
    "citeyearpar",
    "textcite",
    "Textcite",
    "parencite",
    "Parencite",
    "footcite",
    "Footcite",
    "autocite",
    "Autocite",
)
RANGE_REF_COMMANDS = (
    "crefrange",
    "Crefrange",
    "cpagerefrange",
    "Cpagerefrange",
    "vrefrange",
    "Vrefrange",
)
DISPLAY_ENVIRONMENTS = {
    "prop": "proposition",
}

MODERN_ARXIV_RE = re.compile(r"arxiv\s*:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", re.IGNORECASE)
LEGACY_ARXIV_RE = re.compile(
    r"arxiv\s*:\s*([A-Za-z.-]+(?:\.[A-Za-z.-]+)?/[0-9]{7}(?:v\d+)?)",
    re.IGNORECASE,
)
ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|src)/([A-Za-z.-]+(?:\.[A-Za-z.-]+)?/[0-9]{7}(?:v\d+)?|[0-9]{4}\.[0-9]{4,5}(?:v\d+)?)(?:\.pdf)?",
    re.IGNORECASE,
)
BARE_MODERN_ARXIV_ID_RE = re.compile(r"^[0-9]{4}\.[0-9]{4,5}(?:v\d+)?$")
BARE_LEGACY_ARXIV_ID_RE = re.compile(r"^[A-Za-z.-]+(?:\.[A-Za-z.-]+)?/[0-9]{7}(?:v\d+)?$")
EPRINT_RE = re.compile(r"\beprint\s*=\s*[{\"]([^}\"]+)[}\"]", re.IGNORECASE)
ARCHIVE_PREFIX_RE = re.compile(r"\barchiveprefix\s*=\s*[{\"]arxiv[}\"]", re.IGNORECASE)
EPRINT_TYPE_RE = re.compile(r"\beprinttype\s*=\s*[{\"]arxiv[}\"]", re.IGNORECASE)
LABEL_RE = re.compile(r"\\label\s*{([^{}]+)}")
BEGIN_ENV_RE = re.compile(r"\\begin\s*{([^{}]+)}")
BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*{document}")
END_DOCUMENT_RE = re.compile(r"\\end\s*{document}")
DOCUMENTCLASS_RE = re.compile(r"\\documentclass(?:\s*\[[^\]]*\])?\s*{[^{}]+}")
INPUT_OR_INCLUDE_RE = re.compile(r"\\(?:input|include)\s*{[^{}]+}")
TEXT_BIB_RE = re.compile(r"\\bibliography\s*{([^{}]+)}")
RESOURCE_BIB_RE = re.compile(r"\\addbibresource\s*{([^{}]+)}")
STATEMENT_LOCATOR_RE = re.compile(
    r"\b(?:theorem|thm|thm\.|lemma|lem|lem\.|corollary|cor|cor\.|proposition|prop|prop\.|claim)\s+"
    r"[A-Za-z0-9][A-Za-z0-9().:-]*",
    re.IGNORECASE,
)
NEWTHEOREM_RE = re.compile(
    r"\\newtheorem(\*?)\s*{([^{}]+)}(?:\s*\[[^\]]+\])?\s*{([^{}]+)}(?:\s*\[[^\]]+\])?",
    re.IGNORECASE,
)
NON_LETTER_RE = re.compile(r"[^A-Za-z]+")


@dataclass
class BibEntry:
    key: str
    entry_type: str
    raw: str
    arxiv_ids: list[str]


@dataclass
class MacroOccurrence:
    name: str
    start: int
    end: int
    full_text: str
    keys: list[str]
    optional_args: list[str]
    postnote: str | None = None


@dataclass
class StatementRecord:
    environment: str
    label: str | None
    line: int | None
    internal_references: list[dict]
    arxiv_citations: list[dict]
    masked_source: str
    original_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use plasTeX to inspect theorem-like statements and log the ones "
            "that reference only in-paper results and arXiv bibliography entries."
        )
    )
    parser.add_argument(
        "source",
        help="Path to the main .tex file, a directory containing it, or an arXiv identifier.",
    )
    parser.add_argument(
        "--bib",
        dest="bibs",
        action="append",
        type=Path,
        help="Optional bibliography file. Repeat for multiple .bib files.",
    )
    parser.add_argument(
        "--main-tex",
        help="Relative path to the main .tex file when the input is a directory or arXiv identifier.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help="Directory where arXiv sources should be extracted. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Re-download and re-extract arXiv source even if it already exists locally.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output log path. Defaults next to the TeX file.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "jsonl"),
        default="text",
        help="Log format.",
    )
    parser.add_argument(
        "--environment",
        dest="environments",
        action="append",
        help="Additional theorem-like environment name to include.",
    )
    parser.add_argument(
        "--require-internal-ref",
        action="store_true",
        help="Deprecated; retained for CLI compatibility and ignored.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        tex_path, resolution_notes = resolve_input_source(
            args.source,
            main_tex=args.main_tex,
            download_dir=args.download_dir or Path.cwd(),
            redownload=args.redownload,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for note in resolution_notes:
        print(note)

    tex_text = tex_path.read_text(encoding="utf-8")
    bibliography_paths = resolve_bibliography_paths(tex_path, tex_text, args.bibs or [])
    bibliography = load_bibliography(bibliography_paths)
    document = parse_plastex_document(tex_path)

    target_environments = list(DEFAULT_ENVIRONMENTS)
    for extra_environment in args.environments or []:
        if extra_environment not in target_environments:
            target_environments.append(extra_environment)

    environment_aliases, environment_display_names = discover_target_environments(
        tex_text,
        target_environments,
    )

    records, stats = collect_statement_records(
        document=document,
        tex_text=tex_text,
        bibliography=bibliography,
        target_environments=environment_aliases,
        environment_display_names=environment_display_names,
        require_internal_ref=args.require_internal_ref,
    )

    output_path = args.output or default_output_path(tex_path, args.format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "jsonl":
        write_jsonl_log(output_path, records)
    else:
        write_text_log(output_path, tex_path, bibliography_paths, records, stats)

    print(
        "Processed "
        f"{stats['statements_seen']} statements, kept {len(records)}, "
        f"skipped {stats['skipped_non_arxiv_cite']} for non-arXiv citations, "
        f"skipped {stats['skipped_no_relevant_refs']} with no relevant references."
    )
    print(f"Wrote log to {output_path}")
    return 0


def default_output_path(tex_path: Path, output_format: str) -> Path:
    suffix = ".jsonl" if output_format == "jsonl" else ".log"
    paper_id = infer_paper_id(tex_path)
    logs_root = tex_path.parent.parent / "statement_reference_audit_logs"
    return logs_root / f"{sanitize_filename(paper_id)}{suffix}"


def infer_paper_id(tex_path: Path) -> str:
    parent_name = tex_path.parent.name.strip()
    if parent_name:
        if parent_name.lower().startswith("arxiv-"):
            return parent_name[6:]
        return parent_name
    return tex_path.stem


def sanitize_filename(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in {".", "-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned) or "arxiv_source"


def resolve_bibliography_paths(tex_path: Path, tex_text: str, explicit_bibs: Sequence[Path]) -> list[Path]:
    base_dir = tex_path.parent
    requested_paths: list[Path] = []
    if explicit_bibs:
        for path in explicit_bibs:
            requested_paths.extend(candidate_bibliography_paths(base_dir, str(path)))
        return existing_bibliography_paths(requested_paths, tex_path.stem, base_dir)

    active_tex_text = strip_tex_comments(tex_text)
    discovered: list[Path] = []
    for match in TEXT_BIB_RE.finditer(active_tex_text):
        for item in split_csv(match.group(1)):
            for bib_path in candidate_bibliography_paths(base_dir, item):
                if bib_path not in discovered:
                    discovered.append(bib_path)

    for match in RESOURCE_BIB_RE.finditer(active_tex_text):
        for bib_path in candidate_bibliography_paths(base_dir, match.group(1)):
            if bib_path not in discovered:
                discovered.append(bib_path)

    return existing_bibliography_paths(discovered, tex_path.stem, base_dir)


def candidate_bibliography_paths(base_dir: Path, raw_name: str) -> list[Path]:
    raw_name = raw_name.strip()
    if not raw_name:
        return []
    path = Path(raw_name)
    suffix = path.suffix.lower()
    candidates: list[Path] = []
    if suffix in {".bib", ".bbl"}:
        candidates.append(resolve_path(base_dir, path))
        alternate_suffix = ".bbl" if suffix == ".bib" else ".bib"
        candidates.append(resolve_path(base_dir, path.with_suffix(alternate_suffix)))
        return dedupe_paths(candidates)

    candidates.append(resolve_path(base_dir, path.with_suffix(".bib")))
    candidates.append(resolve_path(base_dir, path.with_suffix(".bbl")))
    return dedupe_paths(candidates)


def existing_bibliography_paths(requested_paths: Sequence[Path], tex_stem: str, base_dir: Path) -> list[Path]:
    existing = [path for path in dedupe_paths(requested_paths) if path.exists()]
    if existing:
        return sort_bibliography_paths(existing, tex_stem)

    fallback = list(base_dir.glob("*.bib")) + list(base_dir.glob("*.bbl"))
    return sort_bibliography_paths(dedupe_paths(fallback), tex_stem)


def dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def sort_bibliography_paths(paths: Sequence[Path], tex_stem: str) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, int, str]:
        stem_priority = 0 if path.stem == tex_stem else 1
        suffix_priority = 0 if path.suffix.lower() == ".bib" else 1
        return (stem_priority, suffix_priority, path.name.lower())

    return sorted(paths, key=sort_key)


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def split_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def strip_tex_comments(tex_text: str) -> str:
    cleaned_lines = []
    for line in tex_text.splitlines():
        comment_start = None
        for index, char in enumerate(line):
            if char != "%":
                continue
            if index > 0 and line[index - 1] == "\\":
                continue
            comment_start = index
            break
        if comment_start is None:
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line[:comment_start])
    return "\n".join(cleaned_lines)


def discover_target_environments(
    tex_text: str,
    requested_environments: Sequence[str],
) -> tuple[set[str], dict[str, str]]:
    active_tex_text = strip_tex_comments(tex_text)
    aliases = {environment.lower(): environment.lower() for environment in requested_environments}

    for match in NEWTHEOREM_RE.finditer(active_tex_text):
        environment_name = match.group(2).strip()
        title = match.group(3).strip()
        canonical_name = canonical_theorem_title(title)
        if canonical_name:
            aliases[environment_name.lower()] = canonical_name

    return set(aliases), aliases


def canonical_theorem_title(title: str) -> str | None:
    normalized = NON_LETTER_RE.sub("", title).lower()
    if normalized == "theorem":
        return "theorem"
    if normalized == "lemma":
        return "lemma"
    if normalized == "corollary":
        return "corollary"
    if normalized == "proposition":
        return "proposition"
    return None


def load_bibliography(paths: Sequence[Path]) -> dict[str, BibEntry]:
    bibliography: dict[str, BibEntry] = {}
    for path in paths:
        if not path.exists():
            continue
        raw_text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".bib":
            entries = iter_bib_entries(raw_text)
        elif path.suffix.lower() == ".bbl":
            entries = iter_bbl_entries(raw_text)
        else:
            entries = ()
        for key, entry_type, raw_entry in entries:
            new_entry = BibEntry(
                key=key,
                entry_type=entry_type,
                raw=raw_entry,
                arxiv_ids=extract_arxiv_ids(raw_entry),
            )
            existing = bibliography.get(key)
            if existing is None:
                bibliography[key] = new_entry
                continue

            merged_arxiv_ids = list(dict.fromkeys(existing.arxiv_ids + new_entry.arxiv_ids))
            bibliography[key] = BibEntry(
                key=key,
                entry_type=existing.entry_type or new_entry.entry_type,
                raw=existing.raw if existing.raw else new_entry.raw,
                arxiv_ids=merged_arxiv_ids,
            )
    return bibliography


def iter_bib_entries(text: str) -> Iterator[tuple[str, str, str]]:
    cursor = 0
    length = len(text)
    while cursor < length:
        at_sign = text.find("@", cursor)
        if at_sign == -1:
            return

        index = at_sign + 1
        while index < length and text[index].isspace():
            index += 1

        type_start = index
        while index < length and (text[index].isalpha() or text[index] in "-_"):
            index += 1
        entry_type = text[type_start:index].strip()
        if not entry_type:
            cursor = at_sign + 1
            continue

        while index < length and text[index].isspace():
            index += 1
        if index >= length or text[index] not in "{(":
            cursor = at_sign + 1
            continue

        opening = text[index]
        closing = "}" if opening == "{" else ")"
        entry_start = at_sign
        entry_end = find_matching_delimiter(text, index, opening, closing)
        if entry_end is None:
            return

        raw_entry = text[entry_start : entry_end + 1]
        inner = text[index + 1 : entry_end]
        key = extract_bib_key(inner)
        if key and entry_type.lower() not in {"comment", "preamble", "string"}:
            yield key, entry_type, raw_entry
        cursor = entry_end + 1


def iter_bbl_entries(text: str) -> Iterator[tuple[str, str, str]]:
    yield from iter_amsrefs_bbl_entries(text)
    yield from iter_bibitem_bbl_entries(text)


def iter_amsrefs_bbl_entries(text: str) -> Iterator[tuple[str, str, str]]:
    cursor = 0
    marker = r"\bib"
    while cursor < len(text):
        start = text.find(marker, cursor)
        if start == -1:
            return
        group_start = start + len(marker)
        group_start = skip_whitespace(text, group_start)
        parsed_key = parse_balanced_group(text, group_start, "{", "}")
        if parsed_key is None:
            cursor = start + len(marker)
            continue
        key, cursor_after_key = parsed_key
        cursor_after_key = skip_whitespace(text, cursor_after_key)
        parsed_type = parse_balanced_group(text, cursor_after_key, "{", "}")
        if parsed_type is None:
            cursor = cursor_after_key
            continue
        entry_type, cursor_after_type = parsed_type
        cursor_after_type = skip_whitespace(text, cursor_after_type)
        parsed_body = parse_balanced_group(text, cursor_after_type, "{", "}")
        if parsed_body is None:
            cursor = cursor_after_type
            continue
        _, cursor = parsed_body
        raw_entry = text[start:cursor]
        yield key.strip(), entry_type.strip(), raw_entry


def iter_bibitem_bbl_entries(text: str) -> Iterator[tuple[str, str, str]]:
    marker = r"\bibitem"
    positions: list[int] = []
    cursor = 0
    while cursor < len(text):
        start = text.find(marker, cursor)
        if start == -1:
            break
        positions.append(start)
        cursor = start + len(marker)

    for index, start in enumerate(positions):
        cursor = start + len(marker)
        cursor = skip_whitespace(text, cursor)
        if cursor < len(text) and text[cursor] == "[":
            parsed_optional = parse_balanced_group(text, cursor, "[", "]")
            if parsed_optional is None:
                continue
            _, cursor = parsed_optional
            cursor = skip_whitespace(text, cursor)
        parsed_key = parse_balanced_group(text, cursor, "{", "}")
        if parsed_key is None:
            continue
        key, cursor = parsed_key
        end = positions[index + 1] if index + 1 < len(positions) else len(text)
        raw_entry = text[start:end].rstrip()
        yield key.strip(), "bibitem", raw_entry


def extract_bib_key(inner_text: str) -> str | None:
    comma_index = find_top_level_comma(inner_text)
    if comma_index is None:
        return None
    key = inner_text[:comma_index].strip()
    return key or None


def find_top_level_comma(text: str) -> int | None:
    depth = 0
    in_quote = False
    escape = False
    for index, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth = max(depth - 1, 0)
            continue
        if char == "," and depth == 0:
            return index
    return None


def find_matching_delimiter(text: str, start: int, opening: str, closing: str) -> int | None:
    depth = 0
    in_quote = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if char == opening:
            depth += 1
            continue
        if char == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def extract_arxiv_ids(raw_entry: str) -> list[str]:
    matches: list[str] = []
    for pattern in (MODERN_ARXIV_RE, LEGACY_ARXIV_RE, ARXIV_URL_RE):
        for match in pattern.finditer(raw_entry):
            normalized = normalize_arxiv_id(match.group(1))
            if normalized not in matches:
                matches.append(normalized)

    lower_entry = raw_entry.lower()
    if matches or ("arxiv" not in lower_entry and "archiveprefix" not in lower_entry and "eprinttype" not in lower_entry):
        return matches

    if ARCHIVE_PREFIX_RE.search(raw_entry) or EPRINT_TYPE_RE.search(raw_entry):
        eprint_match = EPRINT_RE.search(raw_entry)
        if eprint_match:
            normalized = normalize_arxiv_id(eprint_match.group(1))
            if normalized and normalized not in matches:
                matches.append(normalized)

    return matches


def normalize_arxiv_id(value: str) -> str:
    normalized = value.strip().strip("{}")
    if normalized.lower().startswith("arxiv:"):
        normalized = normalized[6:]
    if normalized.lower().endswith(".pdf"):
        normalized = normalized[:-4]
    return normalized.strip()


def parse_arxiv_identifier(value: str) -> str | None:
    raw_value = value.strip()
    if not raw_value:
        return None

    url_match = ARXIV_URL_RE.search(raw_value)
    if url_match:
        return normalize_arxiv_id(url_match.group(1))

    normalized = normalize_arxiv_id(raw_value)
    if BARE_MODERN_ARXIV_ID_RE.fullmatch(normalized) or BARE_LEGACY_ARXIV_ID_RE.fullmatch(normalized):
        return normalized

    for pattern in (MODERN_ARXIV_RE, LEGACY_ARXIV_RE):
        match = pattern.search(raw_value)
        if match:
            return normalize_arxiv_id(match.group(1))
    return None


def resolve_input_source(
    raw_source: str,
    main_tex: str | None,
    download_dir: Path,
    redownload: bool,
) -> tuple[Path, list[str]]:
    requested_path = Path(raw_source).expanduser()
    notes: list[str] = []

    if requested_path.exists():
        resolved_path = requested_path.resolve()
        if resolved_path.is_dir():
            tex_path, auto_selected = resolve_main_tex_from_directory(resolved_path, main_tex)
            notes.append(f"Using TeX source directory {resolved_path}")
            if auto_selected:
                notes.append(f"Selected main TeX file {tex_path}")
            return tex_path, notes
        return resolved_path, notes

    arxiv_id = parse_arxiv_identifier(raw_source)
    if arxiv_id is None:
        raise RuntimeError(f"TeX file not found: {requested_path.resolve()}")

    source_dir, resolved_id, downloaded = ensure_arxiv_source_tree(
        arxiv_id,
        download_dir=download_dir,
        redownload=redownload,
    )
    tex_path, auto_selected = resolve_main_tex_from_directory(source_dir, main_tex)
    action = "Downloaded" if downloaded else "Using cached"
    notes.append(f"{action} arXiv source for {resolved_id} in {source_dir}")
    if auto_selected:
        notes.append(f"Selected main TeX file {tex_path}")
    return tex_path, notes


def ensure_arxiv_source_tree(
    arxiv_id: str,
    download_dir: Path,
    redownload: bool,
) -> tuple[Path, str, bool]:
    result = fetch_arxiv_result(arxiv_id)
    resolved_id = normalize_arxiv_id(result.get_short_id())
    source_dir = download_dir.resolve() / f"arXiv-{sanitize_filename(resolved_id)}"

    if source_dir.exists() and not redownload:
        return source_dir, resolved_id, False

    prepare_empty_directory(source_dir)

    with tempfile.TemporaryDirectory(prefix="statement-reference-audit-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        def download_source():
            return result.download_source(
                dirpath=str(temp_dir),
                filename="source-archive",
            )

        archive_path = Path(run_with_retries(download_source, "download arXiv source"))
        extract_downloaded_source(
            archive_path,
            destination=source_dir,
            preferred_stem=sanitize_filename(resolved_id),
        )

    return source_dir, resolved_id, True


def fetch_arxiv_result(arxiv_id: str):
    try:
        import arxiv
        from urllib.error import HTTPError
    except ImportError as exc:
        raise RuntimeError(
            "The `arxiv` package is not installed. Install it first with `pip install arxiv`."
        ) from exc

    def fetch_once():
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        return next(client.results(search))

    try:
        return run_with_retries(fetch_once, "fetch arXiv metadata")
    except StopIteration as exc:
        raise RuntimeError(f"No arXiv record found for {arxiv_id}.") from exc
    except HTTPError as exc:
        raise RuntimeError(f"Could not download arXiv metadata for {arxiv_id}: HTTP {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(f"Could not download arXiv metadata for {arxiv_id}: {exc}") from exc


def run_with_retries(action, description: str, retries: int = 3, base_delay: float = 2.0):
    for attempt in range(retries + 1):
        try:
            return action()
        except Exception as exc:
            if attempt >= retries or not is_rate_limited_error(exc):
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Rate limited while trying to {description}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise RuntimeError(f"Failed to {description} after retries.")


def is_rate_limited_error(exc: Exception) -> bool:
    message = str(exc)
    if "429" in message or "Too Many Requests" in message:
        return True
    code = getattr(exc, "code", None)
    return code == 429


def prepare_empty_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def extract_downloaded_source(archive_path: Path, destination: Path, preferred_stem: str) -> None:
    if try_extract_tar_archive(archive_path, destination):
        return
    if try_extract_gzip_file(archive_path, destination, preferred_stem):
        return

    with archive_path.open("rb") as handle:
        raw_prefix = handle.read(4096)
    output_name = infer_single_file_name(
        preferred_stem,
        raw_prefix,
        default_extension=archive_path.suffix or ".src",
    )
    shutil.copy2(archive_path, destination / output_name)


def try_extract_tar_archive(archive_path: Path, destination: Path) -> bool:
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            safe_extract_tar_archive(archive, destination)
        return True
    except tarfile.TarError:
        return False


def try_extract_gzip_file(archive_path: Path, destination: Path, preferred_stem: str) -> bool:
    if not is_gzip_file(archive_path):
        return False

    try:
        with gzip.open(archive_path, "rb") as source:
            prefix = source.read(4096)
            output_name = infer_single_file_name(preferred_stem, prefix, default_extension=".src")
            with (destination / output_name).open("wb") as handle:
                handle.write(prefix)
                shutil.copyfileobj(source, handle)
        return True
    except OSError as exc:
        raise RuntimeError(f"Could not extract gzip source archive {archive_path}: {exc}") from exc


def safe_extract_tar_archive(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        if member.issym() or member.islnk():
            raise RuntimeError(f"Archive contains unsupported link entry: {member.name}")
        member_path = (destination / member.name).resolve()
        try:
            member_path.relative_to(destination)
        except ValueError as exc:
            raise RuntimeError(f"Archive entry escapes destination: {member.name}") from exc
    archive.extractall(destination)


def is_gzip_file(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def infer_single_file_name(preferred_stem: str, prefix: bytes, default_extension: str) -> str:
    if prefix.startswith(b"%PDF-"):
        extension = ".pdf"
    elif looks_like_tex_source(prefix):
        extension = ".tex"
    else:
        extension = default_extension
    if not extension.startswith("."):
        extension = f".{extension}"
    return f"{preferred_stem}{extension}"


def looks_like_tex_source(prefix: bytes) -> bool:
    if not prefix:
        return False
    snippet = prefix.decode("utf-8", errors="ignore")
    if not snippet.strip():
        return False
    markers = (
        "\\documentclass",
        "\\begin{document}",
        "\\section",
        "\\chapter",
        "\\title",
        "\\author",
    )
    if any(marker in snippet for marker in markers):
        return True
    printable = sum(1 for char in snippet if char.isprintable() or char.isspace())
    return printable >= max(20, int(len(snippet) * 0.9))


def resolve_main_tex_from_directory(source_dir: Path, requested_main_tex: str | None) -> tuple[Path, bool]:
    source_dir = source_dir.resolve()
    if requested_main_tex:
        tex_path = resolve_path(source_dir, Path(requested_main_tex))
        if not tex_path.exists():
            raise RuntimeError(f"Requested main TeX file not found: {tex_path}")
        if tex_path.is_dir():
            raise RuntimeError(f"Requested main TeX path is a directory: {tex_path}")
        return tex_path, False

    candidates = [path.resolve() for path in source_dir.rglob("*.tex") if path.is_file()]
    if not candidates:
        raise RuntimeError(f"No .tex files found in {source_dir}")

    ranked = max(candidates, key=lambda path: tex_candidate_score(path, source_dir))
    return ranked, True


def tex_candidate_score(path: Path, source_dir: Path) -> tuple[int, int, int, str]:
    active_text = strip_tex_comments(read_candidate_text(path))
    lower_name = path.name.lower()
    depth = len(path.relative_to(source_dir).parts) - 1
    score = 0

    if DOCUMENTCLASS_RE.search(active_text):
        score += 100
    if BEGIN_DOCUMENT_RE.search(active_text):
        score += 40
    if END_DOCUMENT_RE.search(active_text):
        score += 10
    if TEXT_BIB_RE.search(active_text) or RESOURCE_BIB_RE.search(active_text):
        score += 10
    if INPUT_OR_INCLUDE_RE.search(active_text):
        score += 5

    if lower_name == "main.tex":
        score += 30
    elif lower_name in {"paper.tex", "ms.tex", "manuscript.tex", "article.tex"}:
        score += 20
    elif lower_name.startswith("main"):
        score += 15

    return (score, -depth, path.stat().st_size, str(path))


def read_candidate_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_plastex_document(tex_path: Path):
    tex_path = tex_path.resolve()
    try:
        from plasTeX.TeX import TeX
    except ImportError as exc:
        raise SystemExit(
            "plasTeX is not installed. Install it first, for example with `pip install plasTeX`, "
            "then rerun this script."
        ) from exc

    apply_plastex_compat_patches()

    original_cwd = Path.cwd()
    os.chdir(tex_path.parent)
    try:
        tex = TeX()
        try:
            tex.disableLogging()
        except Exception:
            pass
        with tex_path.open("r", encoding="utf-8-sig") as handle:
            tex.input(handle)
            return tex.parse()
    finally:
        os.chdir(original_cwd)


def apply_plastex_compat_patches() -> None:
    try:
        from plasTeX import Command
        from plasTeX.TeX import TeX
        from plasTeX.Base.LaTeX.Arrays import tabular
        from plasTeX.Base.TeX.Primitives import csname, ifcsname
        from plasTeX.Packages.longtable import longtable
        from plasTeX.Tokenizer import EscapeSequence
    except Exception:
        return

    if not getattr(TeX, "_statement_audit_safe_kpsewhich", False):
        def safe_kpsewhich(self, name):
            texinputs = os.environ.get("TEXINPUTS", "")
            paths: list[str] = []
            try:
                source_filename = self.filename
            except Exception:
                source_filename = None
            if source_filename:
                source_dir = os.path.dirname(source_filename)
                if source_dir:
                    paths.append(source_dir)
            paths.extend([path for path in texinputs.split(os.path.pathsep) if path])
            paths.append(".")

            seen: set[str] = set()
            for path in paths:
                normalized = os.path.abspath(path or ".")
                if normalized in seen:
                    continue
                seen.add(normalized)
                candidate = os.path.join(normalized, name)
                if os.path.exists(candidate):
                    return candidate

            raise FileNotFoundError(f"Could not find any file named: {name}")

        TeX.kpsewhich = safe_kpsewhich
        TeX._statement_audit_safe_kpsewhich = True

    if not getattr(csname, "_statement_audit_safe_invoke", False):
        def token_text(token) -> str:
            node_name = getattr(token, "nodeName", None)
            if node_name == "#text":
                return str(token)
            if node_name:
                return str(node_name)
            return str(token)

        def safe_csname_invoke(self, tex):
            name_parts: list[str] = []
            for token in tex:
                if token.nodeType == Command.ELEMENT_NODE and token.nodeName == "endcsname":
                    break
                name_parts.append(token_text(token))
            return [EscapeSequence("".join(name_parts))]

        def safe_ifcsname_invoke(self, tex):
            name_parts: list[str] = []
            for token in tex:
                if token.nodeType == Command.ELEMENT_NODE and token.nodeName == "endcsname":
                    break
                name_parts.append(token_text(token))
            name = "".join(name_parts)
            tex.processIfContent(name in self.ownerDocument.context)
            return []

        csname.invoke = safe_csname_invoke
        csname._statement_audit_safe_invoke = True
        ifcsname.invoke = safe_ifcsname_invoke
        ifcsname._statement_audit_safe_invoke = True

    caption_class = longtable.caption
    if not getattr(caption_class, "_statement_audit_safe_digest", False):
        def safe_digest(self, tokens):
            original_digest_base = tabular.caption.digest
            original_digest_base(self, tokens)

            node = self.parentNode
            while node is not None and not isinstance(node, tabular.ArrayRow):
                node = node.parentNode
            if node is not None:
                node.isCaptionRow = True

            table_node = node
            while table_node is not None and not isinstance(table_node, longtable):
                table_node = table_node.parentNode
            if table_node is not None and getattr(table_node, "title", None) is None:
                table_node.title = self

        caption_class.digest = safe_digest
        caption_class._statement_audit_safe_digest = True


def collect_statement_records(
    document,
    tex_text: str,
    bibliography: dict[str, BibEntry],
    target_environments: set[str],
    environment_display_names: dict[str, str],
    require_internal_ref: bool,
) -> tuple[list[StatementRecord], dict[str, int]]:
    cite_commands = set(DEFAULT_CITE_COMMANDS)
    statements = list(iter_target_nodes(document, target_environments))

    stats = {
        "statements_seen": 0,
        "skipped_non_arxiv_cite": 0,
        "skipped_no_relevant_refs": 0,
    }

    records: list[StatementRecord] = []
    search_cursor = 0
    for node in statements:
        stats["statements_seen"] += 1
        source = get_node_source(node)
        if not source:
            stats["skipped_no_relevant_refs"] += 1
            continue

        cite_occurrences = [
            occurrence
            for occurrence in find_macro_occurrences(source, cite_commands)
            if citation_mentions_statement_locator(source, occurrence)
        ]

        if not cite_occurrences:
            stats["skipped_no_relevant_refs"] += 1
            continue

        arxiv_citations: list[dict] = []
        keep_statement = True
        for citation in cite_occurrences:
            cited_entries: list[dict] = []
            for key in citation.keys:
                bib_entry = bibliography.get(key)
                if bib_entry is None or not bib_entry.arxiv_ids:
                    keep_statement = False
                    break
                cited_entries.append({"key": key, "arxiv_ids": bib_entry.arxiv_ids})
            if not keep_statement:
                break
            arxiv_citations.append(
                {
                    "macro": citation.name,
                    "source": citation.full_text,
                    "entries": cited_entries,
                }
            )

        if not keep_statement:
            stats["skipped_non_arxiv_cite"] += 1
            continue

        masked_source = replace_internal_references(source, cite_occurrences)
        position = tex_text.find(source, search_cursor)
        if position == -1:
            position = tex_text.find(source)
        if position == -1:
            label = getattr(node, "id", None) or extract_first_label(source)
            if label:
                position = find_label_position(tex_text, label)
        if position != -1:
            search_cursor = position + len(source)
            line = tex_text.count("\n", 0, position) + 1
        else:
            line = None

        record = StatementRecord(
            environment=display_environment_name(
                get_statement_environment_name(node),
                environment_display_names,
            ),
            label=getattr(node, "id", None) or extract_first_label(source),
            line=line,
            internal_references=[],
            arxiv_citations=arxiv_citations,
            masked_source=masked_source,
            original_source=source,
        )
        records.append(record)

    return records, stats


def iter_target_nodes(node, target_environments: set[str]) -> Iterator[object]:
    environment_name = get_statement_environment_name(node)
    if environment_name and environment_name.lower() in target_environments:
        yield node

    for child in getattr(node, "childNodes", []) or []:
        yield from iter_target_nodes(child, target_environments)


def get_node_name(node) -> str | None:
    return getattr(node, "nodeName", None)


def get_statement_environment_name(node) -> str | None:
    node_name = get_node_name(node)
    if node_name == "thmenv":
        return getattr(node.__class__, "__name__", None)
    return node_name


def get_node_source(node) -> str:
    source = getattr(node, "source", None)
    if source is None:
        return ""
    return str(source)


def display_environment_name(
    environment: str | None,
    environment_display_names: dict[str, str] | None = None,
) -> str:
    if not environment:
        return "unknown"
    if environment_display_names:
        mapped = environment_display_names.get(environment.lower())
        if mapped:
            return DISPLAY_ENVIRONMENTS.get(mapped, mapped)
    return DISPLAY_ENVIRONMENTS.get(environment, environment)


def extract_first_label(source: str) -> str | None:
    match = LABEL_RE.search(source)
    return match.group(1) if match else None


def find_label_position(tex_text: str, label: str) -> int:
    return tex_text.find(f"\\label{{{label}}}")


def find_macro_occurrences(source: str, target_commands: set[str]) -> Iterator[MacroOccurrence]:
    index = 0
    while index < len(source):
        if source[index] != "\\":
            index += 1
            continue

        command_start = index
        index += 1
        if index >= len(source):
            break

        if source[index].isalpha() or source[index] == "@":
            name_end = index + 1
            while name_end < len(source) and (source[name_end].isalpha() or source[name_end] == "@"):
                name_end += 1
        else:
            name_end = index + 1

        command_name = source[index:name_end]
        if not command_name:
            index = name_end
            continue

        if name_end < len(source) and source[name_end] == "*":
            full_name = f"{command_name}*"
            cursor = name_end + 1
        else:
            full_name = command_name
            cursor = name_end

        if command_name not in target_commands:
            index = cursor
            continue

        optional_args: list[str] = []
        cursor = skip_whitespace(source, cursor)
        while cursor < len(source) and source[cursor] == "[":
            parsed = parse_balanced_group(source, cursor, "[", "]")
            if parsed is None:
                break
            content, cursor = parsed
            optional_args.append(content)
            cursor = skip_whitespace(source, cursor)

        required_groups = 2 if command_name in RANGE_REF_COMMANDS else 1
        mandatory_args: list[str] = []
        for _ in range(required_groups):
            cursor = skip_whitespace(source, cursor)
            if cursor >= len(source) or source[cursor] != "{":
                break
            parsed = parse_balanced_group(source, cursor, "{", "}")
            if parsed is None:
                break
            content, cursor = parsed
            mandatory_args.append(content)

        if len(mandatory_args) != required_groups:
            index = name_end
            continue

        postnote = None
        if command_name in DEFAULT_CITE_COMMANDS:
            trailing_cursor = skip_whitespace(source, cursor)
            if trailing_cursor < len(source) and source[trailing_cursor] == "*":
                trailing_cursor += 1
                trailing_cursor = skip_whitespace(source, trailing_cursor)
                parsed_postnote = parse_balanced_group(source, trailing_cursor, "{", "}")
                if parsed_postnote is not None:
                    postnote, cursor = parsed_postnote

        keys = split_macro_keys(mandatory_args)
        yield MacroOccurrence(
            name=full_name,
            start=command_start,
            end=cursor,
            full_text=source[command_start:cursor],
            keys=keys,
            optional_args=optional_args,
            postnote=postnote,
        )
        index = cursor


def skip_whitespace(source: str, index: int) -> int:
    while index < len(source) and source[index].isspace():
        index += 1
    return index


def parse_balanced_group(source: str, start: int, opening: str, closing: str) -> tuple[str, int] | None:
    if start >= len(source) or source[start] != opening:
        return None

    depth = 0
    index = start
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
            continue
        if char == opening:
            depth += 1
            index += 1
            continue
        if char == closing:
            depth -= 1
            index += 1
            if depth == 0:
                return source[start + 1 : index - 1], index
            continue
        index += 1
    return None


def split_macro_keys(groups: Iterable[str]) -> list[str]:
    keys: list[str] = []
    for group in groups:
        for item in group.split(","):
            cleaned = item.strip()
            if cleaned:
                keys.append(cleaned)
    return keys


def citation_mentions_statement_locator(source: str, citation: MacroOccurrence) -> bool:
    for text in citation.optional_args:
        if text_mentions_statement_locator(text):
            return True
    if citation.postnote and text_mentions_statement_locator(citation.postnote):
        return True
    if source:
        window_start = max(0, citation.start - 80)
        window_end = min(len(source), citation.end + 80)
        window = source[window_start:window_end]
        if text_mentions_statement_locator(window):
            return True
    return False


def text_mentions_statement_locator(text: str) -> bool:
    normalized = text.replace("~", " ").replace(r"\ ", " ")
    normalized = re.sub(r"\\[,:;!]", " ", normalized)
    normalized = normalized.replace("{", " ").replace("}", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    return bool(STATEMENT_LOCATOR_RE.search(normalized))


def replace_internal_references(source: str, internal_refs: Sequence[MacroOccurrence]) -> str:
    masked_source = source
    if internal_refs:
        chunks: list[str] = []
        cursor = 0
        for index, occurrence in enumerate(internal_refs, start=1):
            chunks.append(masked_source[cursor : occurrence.start])
            chunks.append(f"[Citation Needed #{index}]")
            cursor = occurrence.end
        chunks.append(masked_source[cursor:])
        masked_source = "".join(chunks)

    return LABEL_RE.sub("", masked_source)


def write_jsonl_log(output_path: Path, records: Sequence[StatementRecord]) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_text_log(
    output_path: Path,
    tex_path: Path,
    bibliography_paths: Sequence[Path],
    records: Sequence[StatementRecord],
    stats: dict[str, int],
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Statement reference audit\n")
        handle.write("=========================\n")
        handle.write(f"Input: {tex_path}\n")
        handle.write(
            "Bibliography files: "
            + (", ".join(str(path) for path in bibliography_paths) if bibliography_paths else "(none found)")
            + "\n"
        )
        handle.write(f"Statements seen: {stats['statements_seen']}\n")
        handle.write(f"Statements kept: {len(records)}\n")
        handle.write(f"Skipped for non-arXiv citations: {stats['skipped_non_arxiv_cite']}\n")
        handle.write(f"Skipped for no relevant references: {stats['skipped_no_relevant_refs']}\n\n")

        for index, record in enumerate(records, start=1):
            handle.write(f"[{index}] {record.environment}\n")
            if record.label:
                handle.write(f"Label: {record.label}\n")
            if record.line is not None:
                handle.write(f"Line: {record.line}\n")

            if record.internal_references:
                handle.write("Internal references:\n")
                for item in record.internal_references:
                    handle.write(
                        f"  - #{item['index']} {item['macro']} -> {', '.join(item['labels'])}\n"
                    )

            if record.arxiv_citations:
                handle.write("ArXiv citations:\n")
                for citation in record.arxiv_citations:
                    rendered_entries = []
                    for entry in citation["entries"]:
                        rendered_entries.append(
                            f"{entry['key']} [{', '.join(entry['arxiv_ids'])}]"
                        )
                    handle.write(
                        f"  - {citation['macro']} -> {', '.join(rendered_entries)}\n"
                    )

            handle.write("Masked statement:\n")
            handle.write(record.masked_source.rstrip() + "\n\n")
            handle.write("Original statement:\n")
            handle.write(record.original_source.rstrip() + "\n")
            handle.write("\n" + ("-" * 80) + "\n\n")


if __name__ == "__main__":
    raise SystemExit(main())
