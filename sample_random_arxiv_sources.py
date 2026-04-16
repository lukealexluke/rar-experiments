#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
import re
import shutil
import sys
import time
from pathlib import Path

from statement_reference_audit_wholebody import (
    ARXIV_ID_CUTOFF,
    ARXIV_VERSION_SUFFIX_RE,
    DEFAULT_CITE_COMMANDS,
    STATEMENT_LOCATOR_RE,
    bib_entry_uses_allowed_arxiv_ids,
    citation_mentions_statement_locator,
    ensure_arxiv_source_tree,
    find_macro_occurrences,
    is_rate_limited_error,
    load_bibliography,
    normalize_arxiv_id,
    read_candidate_text,
    resolve_bibliography_paths,
    resolve_main_tex_from_directory,
    sanitize_filename,
    strip_tex_comments,
)

ARXIV_SUBMISSION_DATE_QUERY = "submittedDate:[202602010000 TO 202603312359]"
ARXIV_CATEGORY_QUERY = "cat:math.*"
ARXIV_SEARCH_QUERY = f"{ARXIV_SUBMISSION_DATE_QUERY} AND {ARXIV_CATEGORY_QUERY}"
ARXIV_SEARCH_PAGE_SIZE = 1_000
ARXIV_SEARCH_PROGRESS_INTERVAL = 500


@dataclass
class MaskedAuditRecord:
    line: int
    masked_text: str
    statement_name: str
    ext_source: str
    mask_start: int
    mask_end: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly download February-March 2026 arXiv math-category TeX sources, "
            "keep only the main TeX file and bibliography, and retain only papers "
            "whose whole-body citation audit produces at least one logged line."
        )
    )
    parser.add_argument(
        "count",
        type=int,
        help="Number of papers with non-empty audit logs to keep.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("random_arxiv_source_samples"),
        help="Directory where sampled paper folders and audit logs will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="Optional cap on unique paper attempts before giving up.",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=4,
        help="Number of extra whole-download retries after hitting an arXiv rate limit.",
    )
    parser.add_argument(
        "--rate-limit-base-delay",
        type=float,
        default=30.0,
        help="Base delay in seconds before retrying after an arXiv rate limit.",
    )
    return parser.parse_args()


def canonicalize_sampled_arxiv_id(value: str) -> str:
    return ARXIV_VERSION_SUFFIX_RE.sub("", normalize_arxiv_id(value))


def fetch_math_candidate_ids() -> list[str]:
    try:
        import arxiv
    except ImportError as exc:
        raise RuntimeError(
            "The `arxiv` package is not installed. Install it first with `pip install arxiv`."
        ) from exc

    print(f"Building arXiv candidate pool from query: {ARXIV_SEARCH_QUERY}")
    client = arxiv.Client(
        page_size=ARXIV_SEARCH_PAGE_SIZE,
        delay_seconds=3.0,
        num_retries=3,
    )
    search = arxiv.Search(
        query=ARXIV_SEARCH_QUERY,
        max_results=None,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )

    candidate_ids: list[str] = []
    seen_ids: set[str] = set()
    try:
        for result in client.results(search):
            arxiv_id = canonicalize_sampled_arxiv_id(result.get_short_id())
            if arxiv_id in seen_ids:
                continue
            seen_ids.add(arxiv_id)
            candidate_ids.append(arxiv_id)
            if len(candidate_ids) % ARXIV_SEARCH_PROGRESS_INTERVAL == 0:
                print(f"  loaded {len(candidate_ids)} candidate papers so far")
    except Exception as exc:
        raise RuntimeError(f"Could not build the math-only arXiv candidate pool: {exc}") from exc

    if not candidate_ids:
        raise RuntimeError("The math-only arXiv candidate pool is empty.")

    print(f"Loaded {len(candidate_ids)} math-category candidates.")
    return candidate_ids


def load_existing_kept_ids(downloads_root: Path) -> set[str]:
    kept_ids: set[str] = set()
    for metadata_path in sorted(downloads_root.glob("paper_*/paper_metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("paper_source", "resolved_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                kept_ids.add(canonicalize_sampled_arxiv_id(value))
                break
    return kept_ids


def path_is_within_directory(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def remove_source_tree(source_dir: Path, downloads_root: Path) -> None:
    if not source_dir.exists():
        return
    if not path_is_within_directory(source_dir, downloads_root):
        raise RuntimeError(f"Refusing to delete directory outside downloads root: {source_dir}")
    shutil.rmtree(source_dir)


def remove_incomplete_downloads(downloads_root: Path, existing_dirs: set[Path]) -> None:
    for path in downloads_root.glob("arXiv-*"):
        resolved = path.resolve()
        if resolved in existing_dirs or not path.is_dir():
            continue
        remove_source_tree(resolved, downloads_root)


def iter_exception_chain(exc: BaseException):
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def exception_is_rate_limited(exc: BaseException) -> bool:
    return any(is_rate_limited_error(item) for item in iter_exception_chain(exc))


def ensure_arxiv_source_tree_with_backoff(
    arxiv_id: str,
    download_dir: Path,
    rate_limit_retries: int,
    rate_limit_base_delay: float,
) -> tuple[Path, str, bool]:
    for attempt in range(rate_limit_retries + 1):
        existing_dirs = {path.resolve() for path in download_dir.glob("arXiv-*") if path.is_dir()}
        try:
            return ensure_arxiv_source_tree(
                arxiv_id,
                download_dir=download_dir,
                redownload=False,
            )
        except Exception as exc:
            remove_incomplete_downloads(download_dir, existing_dirs)
            if attempt >= rate_limit_retries or not exception_is_rate_limited(exc):
                raise
            delay = rate_limit_base_delay * (2 ** attempt)
            print(f"  rate limited by arXiv; retrying this paper in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError(f"Failed to download arXiv source for {arxiv_id} after rate-limit retries.")


def prune_source_tree(source_dir: Path, keep_files: set[Path]) -> None:
    source_dir = source_dir.resolve()
    normalized_keep_files = {path.resolve() for path in keep_files}
    for path in sorted(source_dir.rglob("*"), key=lambda candidate: len(candidate.parts), reverse=True):
        resolved = path.resolve()
        if path.is_file():
            if resolved not in normalized_keep_files:
                path.unlink()
        elif path.is_dir() and resolved != source_dir:
            if not any(path.iterdir()):
                path.rmdir()


def normalize_locator_text(text: str) -> str:
    normalized = text.replace("~", " ").replace(r"\ ", " ")
    normalized = re.sub(r"\\[,:;!]", " ", normalized)
    normalized = normalized.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def first_statement_locator(text: str) -> str:
    normalized = normalize_locator_text(text)
    match = STATEMENT_LOCATOR_RE.search(normalized)
    return match.group(0).strip() if match else ""


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


def select_first_arxiv_id(keys: list[str], bibliography: dict[str, object]) -> str:
    for key in keys:
        bib_entry = bibliography.get(key)
        arxiv_ids = getattr(bib_entry, "arxiv_ids", None)
        if not arxiv_ids:
            continue
        return normalize_arxiv_id(str(arxiv_ids[0]))
    return ""


def find_matching_line_occurrence(line_text: str, citation) -> object | None:
    cite_commands = set(DEFAULT_CITE_COMMANDS)
    occurrences = list(find_macro_occurrences(line_text, cite_commands))
    for occurrence in occurrences:
        if occurrence.full_text == citation.full_text and occurrence.keys == citation.keys:
            return occurrence
    for occurrence in occurrences:
        if occurrence.keys == citation.keys:
            return occurrence
    return occurrences[0] if occurrences else None


def collect_masked_line_citation_records(
    tex_text: str,
    bibliography: dict[str, object],
) -> tuple[list[MaskedAuditRecord], dict[str, int], str]:
    cite_commands = set(DEFAULT_CITE_COMMANDS)
    active_text = strip_tex_comments(tex_text)
    occurrences = list(find_macro_occurrences(active_text, cite_commands))
    original_lines = tex_text.splitlines()

    stats = {
        "citations_seen": 0,
        "skipped_non_arxiv_cite": 0,
        "skipped_no_locator": 0,
        "skipped_post_cutoff_arxiv_cite": 0,
        "skipped_repeat_cite": 0,
    }

    line_records: dict[int, MaskedAuditRecord] = {}
    seen_citation_keys: set[str] = set()
    for citation in occurrences:
        stats["citations_seen"] += 1

        first_instance_keys = [key for key in citation.keys if key not in seen_citation_keys]
        seen_citation_keys.update(citation.keys)
        if not first_instance_keys:
            stats["skipped_repeat_cite"] += 1
            continue

        if not citation_mentions_statement_locator(active_text, citation):
            stats["skipped_no_locator"] += 1
            continue

        keep_citation = True
        post_cutoff_citation = False
        for key in citation.keys:
            bib_entry = bibliography.get(key)
            if bib_entry is None or not getattr(bib_entry, "arxiv_ids", None):
                keep_citation = False
                break
            if not bib_entry_uses_allowed_arxiv_ids(bib_entry):
                post_cutoff_citation = True
                break
        if not keep_citation:
            stats["skipped_non_arxiv_cite"] += 1
            continue
        if post_cutoff_citation:
            stats["skipped_post_cutoff_arxiv_cite"] += 1
            continue

        line_number = active_text.count("\n", 0, citation.start) + 1
        if line_number in line_records:
            continue
        if line_number > len(original_lines):
            continue

        original_line_text = original_lines[line_number - 1]
        line_occurrence = find_matching_line_occurrence(original_line_text, citation)
        if line_occurrence is None:
            continue

        statement_name = extract_statement_name(original_line_text, citation)
        ext_source = select_first_arxiv_id(citation.keys, bibliography)
        if not statement_name or not ext_source:
            continue

        masked_text = (
            original_line_text[: line_occurrence.start]
            + "[Citation Needed]"
            + original_line_text[line_occurrence.end :]
        )
        line_records[line_number] = MaskedAuditRecord(
            line=line_number,
            masked_text=masked_text.rstrip(),
            statement_name=statement_name,
            ext_source=ext_source,
            mask_start=line_occurrence.start,
            mask_end=line_occurrence.end,
        )

    records: list[MaskedAuditRecord] = []
    masked_lines = list(original_lines)
    for line_number in sorted(line_records):
        original_line_text = original_lines[line_number - 1]
        line_occurrences = list(find_macro_occurrences(original_line_text, cite_commands))
        if not line_occurrences:
            continue
        if any(not citation_mentions_statement_locator(original_line_text, occurrence) for occurrence in line_occurrences):
            continue

        record = line_records[line_number]
        masked_lines[line_number - 1] = (
            original_line_text[: record.mask_start]
            + "[Citation Needed]"
            + original_line_text[record.mask_end :]
        )
        records.append(record)

    masked_tex_text = "\n".join(masked_lines)
    if tex_text.endswith("\n"):
        masked_tex_text += "\n"
    return records, stats, masked_tex_text


def write_masked_text_log(
    output_path: Path,
    tex_path: Path,
    bibliography_paths: list[Path],
    records: list[MaskedAuditRecord],
    stats: dict[str, int],
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("Whole-body citation audit\n")
        handle.write("=========================\n")
        handle.write(f"Input: {tex_path}\n")
        handle.write(f"arXiv cutoff: <= {ARXIV_ID_CUTOFF}\n")
        handle.write(
            "Bibliography files: "
            + (", ".join(str(path) for path in bibliography_paths) if bibliography_paths else "(none found)")
            + "\n"
        )
        handle.write(f"Citations seen: {stats['citations_seen']}\n")
        handle.write(f"Lines kept: {len(records)}\n")
        handle.write(f"Skipped repeated citations: {stats['skipped_repeat_cite']}\n")
        handle.write(f"Skipped post-cutoff arXiv citations: {stats['skipped_post_cutoff_arxiv_cite']}\n")
        handle.write(f"Skipped for non-arXiv citations: {stats['skipped_non_arxiv_cite']}\n")
        handle.write(f"Skipped for no statement locator: {stats['skipped_no_locator']}\n\n")

        for index, record in enumerate(records, start=1):
            handle.write(f"[{index}] Line: {record.line}\n")
            handle.write(f"Name: {record.statement_name}\n")
            handle.write(f"External Source: {record.ext_source}\n")
            handle.write(record.masked_text + "\n")
            handle.write("\n" + ("-" * 80) + "\n\n")


def rename_successful_source_tree(
    source_dir: Path,
    tex_path: Path,
    bibliography_paths: list[Path],
    downloads_root: Path,
    paper_index: int,
) -> tuple[Path, Path, list[Path]]:
    source_dir = source_dir.resolve()
    downloads_root = downloads_root.resolve()
    target_dir = downloads_root / f"paper_{paper_index}"
    if target_dir.exists() and target_dir.resolve() != source_dir:
        raise RuntimeError(f"Target paper directory already exists: {target_dir}")

    original_source_dir = source_dir
    if source_dir != target_dir:
        source_dir.rename(target_dir)
    target_dir = target_dir.resolve()

    def relocate(old_path: Path, new_path: Path) -> Path:
        current_path = target_dir / old_path.resolve().relative_to(original_source_dir)
        if current_path.resolve() == new_path.resolve():
            return new_path
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(current_path), str(new_path))
        return new_path

    renamed_tex_path = relocate(tex_path, target_dir / f"main_{paper_index}.tex")
    renamed_bibliography_paths: list[Path] = []
    sorted_bibliography_paths = sorted(bibliography_paths, key=lambda path: str(path.resolve()))
    for bib_number, bibliography_path in enumerate(sorted_bibliography_paths, start=1):
        if len(sorted_bibliography_paths) == 1:
            target_name = f"bib_{paper_index}.txt"
        else:
            target_name = f"bib_{paper_index}_{bib_number}.txt"
        renamed_bibliography_paths.append(relocate(bibliography_path, target_dir / target_name))

    prune_source_tree(target_dir, {renamed_tex_path, *renamed_bibliography_paths})
    return target_dir, renamed_tex_path, renamed_bibliography_paths


def find_next_paper_index(downloads_root: Path) -> int:
    max_index = 0
    for path in downloads_root.glob("paper_*"):
        if not path.is_dir():
            continue
        try:
            index = int(path.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        max_index = max(max_index, index)
    return max_index + 1


def write_paper_metadata(
    paper_dir: Path,
    resolved_id: str,
    output_path: Path,
    logged_line_count: int,
) -> None:
    metadata = {
        "paper_source": resolved_id,
        "resolved_id": resolved_id,
        "paper_label": paper_dir.name,
        "main_tex": next((path.name for path in paper_dir.glob("main_*.tex")), None),
        "bibliography_files": sorted(path.name for path in paper_dir.glob("bib_*.txt")),
        "log_file": output_path.name,
        "logged_line_count": logged_line_count,
        "sample_query": ARXIV_SEARCH_QUERY,
        "sampled_without_replacement": True,
    }
    (paper_dir / "paper_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def audit_downloaded_source(
    source_dir: Path,
    resolved_id: str,
    downloads_root: Path,
    logs_dir: Path,
) -> tuple[bool, Path, Path, Path, list[Path], int]:
    tex_path, _ = resolve_main_tex_from_directory(source_dir, requested_main_tex=None)
    tex_text = read_candidate_text(tex_path)
    bibliography_paths = resolve_bibliography_paths(tex_path, tex_text, [])
    keep_files = {tex_path.resolve(), *(path.resolve() for path in bibliography_paths)}
    prune_source_tree(source_dir, keep_files)

    tex_text = read_candidate_text(tex_path)
    bibliography = load_bibliography(bibliography_paths)
    records, stats, masked_tex_text = collect_masked_line_citation_records(tex_text, bibliography)

    output_path = logs_dir / f"{sanitize_filename(resolved_id)}_wholebody.log"
    if not records:
        if output_path.exists():
            output_path.unlink()
        return False, output_path, source_dir, tex_path, bibliography_paths, 0

    tex_path.write_text(masked_tex_text, encoding="utf-8")
    paper_index = find_next_paper_index(downloads_root)
    output_path = logs_dir / f"paper_{paper_index}_wholebody.log"
    try:
        paper_dir, tex_path, bibliography_paths = rename_successful_source_tree(
            source_dir=source_dir,
            tex_path=tex_path,
            bibliography_paths=bibliography_paths,
            downloads_root=downloads_root,
            paper_index=paper_index,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_masked_text_log(output_path, tex_path, bibliography_paths, records, stats)
        return True, output_path, paper_dir, tex_path, bibliography_paths, len(records)
    except Exception:
        if output_path.exists():
            output_path.unlink()
        paper_dir = downloads_root / f"paper_{paper_index}"
        if paper_dir.exists():
            remove_source_tree(paper_dir, downloads_root)
        raise


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        print("Error: count must be positive.", file=sys.stderr)
        return 1
    if args.max_attempts is not None and args.max_attempts <= 0:
        print("Error: max-attempts must be positive when provided.", file=sys.stderr)
        return 1
    if args.rate_limit_retries < 0:
        print("Error: rate-limit-retries must be non-negative.", file=sys.stderr)
        return 1
    if args.rate_limit_base_delay <= 0:
        print("Error: rate-limit-base-delay must be positive.", file=sys.stderr)
        return 1

    downloads_root = args.download_dir.resolve()
    logs_dir = downloads_root / "statement_reference_audit_logs"
    downloads_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    try:
        candidate_ids = fetch_math_candidate_ids()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    existing_kept_ids = load_existing_kept_ids(downloads_root)
    if existing_kept_ids:
        original_candidate_count = len(candidate_ids)
        candidate_ids = [arxiv_id for arxiv_id in candidate_ids if arxiv_id not in existing_kept_ids]
        skipped_count = original_candidate_count - len(candidate_ids)
        print(
            f"Excluded {skipped_count} already-kept papers found in {downloads_root} "
            "from the candidate pool."
        )

    if not candidate_ids:
        print("Error: no math-category candidates remain after exclusions.", file=sys.stderr)
        return 1

    rng.shuffle(candidate_ids)
    print(
        f"Sampling without replacement from {len(candidate_ids)} February-March 2026 "
        "math-category arXiv papers."
    )

    max_unique_attempts = len(candidate_ids)
    if args.max_attempts is not None:
        max_unique_attempts = min(args.max_attempts, len(candidate_ids))
        if args.max_attempts > len(candidate_ids):
            print(
                f"Requested max-attempts={args.max_attempts}, but only {len(candidate_ids)} "
                "unique math-category candidates are available."
            )

    successes = 0
    attempts = 0

    while successes < args.count:
        if attempts >= max_unique_attempts:
            print(
                f"Stopped after {attempts} unique attempts and kept {successes} papers.",
                file=sys.stderr,
            )
            return 1

        sampled_id = candidate_ids.pop()
        attempts += 1
        source_dir: Path | None = None
        resolved_id = sampled_id

        print(f"[attempt {attempts}] sampled {sampled_id}")
        try:
            source_dir, resolved_id, downloaded = ensure_arxiv_source_tree_with_backoff(
                sampled_id,
                download_dir=downloads_root,
                rate_limit_retries=args.rate_limit_retries,
                rate_limit_base_delay=args.rate_limit_base_delay,
            )
            action = "downloaded" if downloaded else "using cached"
            print(f"  {action}: {resolved_id}")

            kept, output_path, paper_dir, tex_path, bibliography_paths, record_count = audit_downloaded_source(
                source_dir=source_dir,
                resolved_id=resolved_id,
                downloads_root=downloads_root,
                logs_dir=logs_dir,
            )
            if not kept:
                remove_source_tree(source_dir, downloads_root)
                print("  skipped: no logged citation lines found")
                continue

            successes += 1
            write_paper_metadata(
                paper_dir=paper_dir,
                resolved_id=resolved_id,
                output_path=output_path,
                logged_line_count=record_count,
            )
            bib_summary = ", ".join(path.name for path in bibliography_paths) if bibliography_paths else "(none)"
            print(f"  kept [{successes}/{args.count}]: {output_path}")
            print(f"  folder: {paper_dir.name}")
            print(f"  main: {tex_path.name}")
            print(f"  bibliography: {bib_summary}")
            print(f"  logged lines: {record_count}")
        except RuntimeError as exc:
            if "The `arxiv` package is not installed" in str(exc):
                print(f"Error: {exc}", file=sys.stderr)
                return 1
            if source_dir is not None and source_dir.exists():
                remove_source_tree(source_dir, downloads_root)
            print(f"  skipped: {exc}")
        except Exception as exc:
            if source_dir is not None and source_dir.exists():
                remove_source_tree(source_dir, downloads_root)
            print(f"  skipped: unexpected error for {resolved_id}: {exc}")

    print(
        f"Finished after {attempts} attempts. Kept {successes} papers in {downloads_root}."
    )
    print(f"Audit logs are in {logs_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
