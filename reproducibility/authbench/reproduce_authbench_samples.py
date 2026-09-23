#!/usr/bin/env python3
"""Reproduce the AuthBench query cohorts reported in the PromptRemix paper."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "reproducibility" / "authbench"
MANIFESTS = {
    "canonical": "canonical_query_ids.json",
    "short_text": "short_text_query_ids.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the exact PromptRemix paper cohorts from a local AuthBench copy."
        )
    )
    parser.add_argument(
        "--authbench-dir",
        type=Path,
        required=True,
        help="Directory containing <language>/per_genre AuthBench query files.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=DEFAULT_MANIFEST_DIR,
        help=f"Directory containing the frozen query-ID files (default: {DEFAULT_MANIFEST_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("authbench_promptremix_samples"),
        help="Destination for reproduced query and sampled-ID files.",
    )
    parser.add_argument(
        "--cohort",
        choices=["canonical", "short-text", "all"],
        default="all",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check that every frozen query ID exists without writing files.",
    )
    return parser.parse_args()


def load_manifest(path: Path, expected_cohort: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    if manifest.get("cohort") != expected_cohort:
        raise ValueError(
            f"{path}: expected cohort {expected_cohort!r}, "
            f"found {manifest.get('cohort')!r}"
        )

    seen_ids: set[str] = set()
    language_counts: Counter[str] = Counter()
    total = 0
    for cell in manifest.get("cells", []):
        language = cell.get("language")
        genre = cell.get("genre")
        query_ids = cell.get("query_ids")
        if not isinstance(language, str) or not isinstance(genre, str):
            raise ValueError(f"{path}: every cell needs string language and genre")
        if not isinstance(query_ids, list) or not all(
            isinstance(query_id, str) for query_id in query_ids
        ):
            raise ValueError(f"{path}: {language}/{genre} has invalid query_ids")
        if len(query_ids) != cell.get("query_count"):
            raise ValueError(f"{path}: {language}/{genre} query_count mismatch")
        duplicates = seen_ids.intersection(query_ids)
        if duplicates:
            example = sorted(duplicates)[0]
            raise ValueError(f"{path}: duplicate query ID {example!r}")
        seen_ids.update(query_ids)
        language_counts[language] += len(query_ids)
        total += len(query_ids)

    if total != manifest.get("total_queries"):
        raise ValueError(f"{path}: total_queries mismatch")
    if dict(sorted(language_counts.items())) != manifest.get("language_counts"):
        raise ValueError(f"{path}: language_counts mismatch")
    return manifest


def source_path(authbench_dir: Path, language: str, genre: str) -> Path:
    return (
        authbench_dir
        / language
        / "per_genre"
        / f"authbench_{language}_{genre}_queries.jsonl"
    )


def select_queries(
    path: Path,
    query_ids: list[str],
) -> list[tuple[int, dict[str, Any]]]:
    wanted = set(query_ids)
    selected: dict[str, tuple[int, dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("documentID"))
            if query_id not in wanted:
                continue
            if query_id in selected:
                raise ValueError(f"{path}: duplicate source query ID {query_id!r}")
            selected[query_id] = (index, row)

    missing = [query_id for query_id in query_ids if query_id not in selected]
    if missing:
        preview = ", ".join(repr(query_id) for query_id in missing[:5])
        raise ValueError(
            f"{path}: missing {len(missing)} frozen query IDs; first: {preview}"
        )
    return [selected[query_id] for query_id in query_ids]


def write_cell(
    output_dir: Path,
    cohort: str,
    language: str,
    genre: str,
    selected: list[tuple[int, dict[str, Any]]],
) -> None:
    query_path = (
        output_dir
        / cohort
        / "queries"
        / language
        / "per_genre"
        / f"authbench_{language}_{genre}_queries.jsonl"
    )
    id_path = (
        output_dir
        / cohort
        / "sampled_ids"
        / language
        / genre
        / f"authbench_{language}_{genre}_sampled_ids.json"
    )
    query_path.parent.mkdir(parents=True, exist_ok=True)
    id_path.parent.mkdir(parents=True, exist_ok=True)

    with query_path.open("w", encoding="utf-8") as handle:
        for _, row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    sampled_ids = [
        {"index": index, "documentID": str(row["documentID"])}
        for index, row in selected
    ]
    with id_path.open("w", encoding="utf-8") as handle:
        json.dump(sampled_ids, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def reproduce_cohort(
    manifest: dict[str, Any],
    authbench_dir: Path,
    output_dir: Path,
    verify_only: bool,
) -> int:
    cohort = str(manifest["cohort"])
    total = 0
    for cell in manifest["cells"]:
        language = str(cell["language"])
        genre = str(cell["genre"])
        path = source_path(authbench_dir, language, genre)
        if not path.exists():
            raise FileNotFoundError(path)
        selected = select_queries(path, cell["query_ids"])
        total += len(selected)
        if not verify_only:
            write_cell(output_dir, cohort, language, genre, selected)
    return total


def main() -> None:
    args = parse_args()
    cohorts = (
        ["canonical", "short_text"]
        if args.cohort == "all"
        else [args.cohort.replace("-", "_")]
    )
    for cohort in cohorts:
        manifest_path = args.manifest_dir / MANIFESTS[cohort]
        manifest = load_manifest(manifest_path, cohort)
        total = reproduce_cohort(
            manifest,
            args.authbench_dir,
            args.output_dir,
            args.verify_only,
        )
        action = "verified" if args.verify_only else "wrote"
        print(f"{cohort}: {action} {total} queries across {len(manifest['cells'])} cells")


if __name__ == "__main__":
    main()
