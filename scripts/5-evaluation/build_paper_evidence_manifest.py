#!/usr/bin/env python3
"""Inventory PromptRemix paper evidence and build canonical summary artifacts.

The script is intentionally read-only with respect to experiment outputs. It scans
the metrics directory, matches the frozen Blablablab privacy evaluations to their
exact sense-score files, and writes a traceable paper-evidence manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ATTRIBUTOR = "Blablablab/multilingual-style-representation"
ATTRIBUTOR_TAG = ATTRIBUTOR.replace("/", "_")
METHOD_NOTES = {
    "jamdec": (
        "Canonical cohort only; executed reduced-search configuration with "
        "likelihood-gpt2, constraints_only, beam width 5, deterministic "
        "decoding, and a 384-token maximum decode length."
    )
}

GENRES = {
    "en": [
        "blog",
        "ecommerce_reviews",
        "literature",
        "news",
        "poetry",
        "qna",
        "research_paper",
    ],
    "ru": ["literature", "news", "poetry", "qna", "social_media"],
    "zh": [
        "ecommerce_reviews",
        "literature",
        "media_reviews",
        "news",
        "poetry",
        "social_media",
    ],
    "ar": ["literature", "news", "poetry", "social_media"],
}

SENSE_METRICS = [
    "semantic_coverage",
    "factuality",
    "informativeness",
    "relevance",
    "specificity",
    "correctness",
    "accuracy",
    "semantically_appropriate",
    "consistency",
    "coherence",
    "fluency",
    "quality",
    "understandability",
    "cosine_similarity",
]

PRIVACY_FIELDS = [
    "eer_original",
    "eer_privatized",
    "delta_eer",
    "n_queries",
    "n_candidates",
    "n_positive_pairs",
    "n_negative_pairs",
]


def variant_specs(language: str) -> list[dict[str, Any]]:
    """Return the paper-facing method variants expected for one language."""
    specs = [
        {
            "variant": "mutantx",
            "method": "mutantx",
            "num_styles": None,
            "pipeline_suffix": "mutantx",
        },
        {
            "variant": "steer",
            "method": "steer",
            "num_styles": None,
            "pipeline_suffix": "steer",
        },
        {
            "variant": "paraphrasing",
            "method": "paraphrasing",
            "num_styles": None,
            "pipeline_suffix": "paraphrasing",
        },
        {
            "variant": "round_trip_mt",
            "method": "round_trip_mt",
            "num_styles": None,
            "pipeline_suffix": "round_trip_mt",
        },
        {
            "variant": "stylometry",
            "method": "stylometry",
            "num_styles": None,
            "pipeline_suffix": "stylometry",
        },
        {
            "variant": "jamdec",
            "method": "jamdec",
            "num_styles": None,
            "pipeline_suffix": (
                f"obfuscate_eval_jamdec_authbench_{language}-baseline"
            ),
        },
    ]

    for k in (1, 2, 3):
        specs.append(
            {
                "variant": f"styleremix_k{k}",
                "method": "styleremix",
                "num_styles": k,
                "pipeline_suffix": (
                    f"obfuscate_eval_styleremix_authbench_"
                    f"{language}-baseline-{k}"
                ),
            }
        )
    for k in (1, 2, 3):
        specs.append(
            {
                "variant": f"promptremix_k{k}",
                "method": "promptremix",
                "num_styles": k,
                "pipeline_suffix": f"promptremix_authbench-{language}0-{k}",
            }
        )
    return specs


def index_by_name(paths: Iterable[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[path.name].append(path)
    return index


def choose_path(
    candidates: list[Path],
    preferred_parent: Path | None = None,
) -> tuple[Path | None, str]:
    if not candidates:
        return None, "missing"
    if preferred_parent is not None:
        preferred = [p for p in candidates if p.parent == preferred_parent]
        if len(preferred) == 1:
            return preferred[0], "complete"
    if len(candidates) == 1:
        return candidates[0], "complete"
    return None, "ambiguous"


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize_sense(path: Path) -> dict[str, Any]:
    sums = {metric: 0.0 for metric in SENSE_METRICS}
    valid_counts = {metric: 0 for metric in SENSE_METRICS}
    n_rows = 0
    n_complete_rows = 0
    n_parse_errors = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                n_parse_errors += 1
                continue

            n_rows += 1
            complete = True
            for metric in SENSE_METRICS:
                value = row.get(metric)
                if isinstance(value, (int, float)):
                    sums[metric] += float(value)
                    valid_counts[metric] += 1
                else:
                    complete = False
            if complete:
                n_complete_rows += 1

    summary: dict[str, Any] = {
        "status": "complete",
        "path": str(path),
        "n_rows": n_rows,
        "n_complete_rows": n_complete_rows,
        "n_parse_errors": n_parse_errors,
        "complete_fraction": n_complete_rows / n_rows if n_rows else None,
    }
    for metric in SENSE_METRICS:
        count = valid_counts[metric]
        summary[f"{metric}_mean"] = sums[metric] / count if count else None
        summary[f"{metric}_n"] = count
        summary[f"{metric}_missing"] = n_rows - count
    return summary


def summarize_identity(source_path: Path, privatized_path: Path) -> dict[str, Any]:
    if not source_path.is_file() or not privatized_path.is_file():
        return {
            "status": "missing",
            "source_path": str(source_path),
            "privatized_path": str(privatized_path),
        }

    source_texts: dict[str, str] = {}
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                source_texts[str(row["documentID"])] = str(row["fullText"])

    n_compared = 0
    n_exact = 0
    n_missing_source = 0
    with privatized_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_text = source_texts.get(str(row["documentID"]))
            if source_text is None:
                n_missing_source += 1
                continue
            n_compared += 1
            n_exact += int(str(row["fullText"]) == source_text)

    return {
        "status": "complete" if n_missing_source == 0 else "incomplete",
        "source_path": str(source_path),
        "privatized_path": str(privatized_path),
        "n_compared": n_compared,
        "n_exact": n_exact,
        "n_missing_source": n_missing_source,
        "exact_fraction": n_exact / n_compared if n_compared else None,
    }


def privacy_filename(language: str, genre: str, suffix: str) -> str:
    return f"authbench_{language}_{genre}_{ATTRIBUTOR_TAG}_{suffix}.json"


def sense_filename(genre: str, suffix: str) -> str:
    return f"{genre}_{suffix}_llm.jsonl"


def load_privacy(
    path: Path,
    language: str,
    genre: str,
    suffix: str,
) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    validation_errors = []
    expected = {
        "model": ATTRIBUTOR,
        "language": language,
        "dataset": "authbench",
        "genre": genre,
        "pipeline_suffix": suffix,
    }
    for field, value in expected.items():
        if data.get(field) != value:
            validation_errors.append(
                f"{field}: expected {value!r}, got {data.get(field)!r}"
            )

    status = "complete" if not validation_errors else "invalid"
    result = {
        "status": status,
        "path": str(path),
        "validation_errors": validation_errors,
    }
    result.update({field: data.get(field) for field in PRIVACY_FIELDS})
    return result


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    privacy = record["privacy"]
    sense = record["sense"]
    row = {
        "language": record["language"],
        "genre": record["genre"],
        "variant": record["variant"],
        "method": record["method"],
        "num_styles": record["num_styles"],
        "pipeline_suffix": record["pipeline_suffix"],
        "privacy_status": privacy["status"],
        "privacy_path": privacy.get("path"),
        "sense_status": sense["status"],
        "sense_path": sense.get("path"),
        "sense_candidate_count": len(record["sense_candidates"]),
        "sense_query_count_matches_privacy": record[
            "sense_query_count_matches_privacy"
        ],
    }
    for field in PRIVACY_FIELDS:
        row[field] = privacy.get(field)
    row["sense_n_rows"] = sense.get("n_rows")
    row["sense_n_complete_rows"] = sense.get("n_complete_rows")
    row["sense_complete_fraction"] = sense.get("complete_fraction")
    for metric in SENSE_METRICS:
        row[f"sense_{metric}_mean"] = sense.get(f"{metric}_mean")
    identity = record.get("identity") or {}
    row["identity_status"] = identity.get("status")
    row["identity_source_path"] = identity.get("source_path")
    row["identity_privatized_path"] = identity.get("privatized_path")
    row["identity_compared_count"] = identity.get("n_compared")
    row["identity_exact_count"] = identity.get("n_exact")
    row["identity_exact_fraction"] = identity.get("exact_fraction")
    return row


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str] | None = None,
) -> None:
    if fieldnames is None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = [flatten_record(record) for record in records]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in flat:
        groups[(row["language"], row["variant"])].append(row)
        groups[("macro", row["variant"])].append(row)

    rows = []
    for (language, variant), group in sorted(groups.items()):
        privacy_rows = [row for row in group if row["privacy_status"] == "complete"]
        sense_rows = [row for row in group if row["sense_status"] == "complete"]
        result: dict[str, Any] = {
            "language": language,
            "variant": variant,
            "method": group[0]["method"],
            "num_styles": group[0]["num_styles"],
            "expected_cells": len(group),
            "privacy_cells": len(privacy_rows),
            "sense_cells": len(sense_rows),
        }
        for field in ("eer_original", "eer_privatized", "delta_eer"):
            result[f"{field}_macro_mean"] = mean(
                float(row[field])
                for row in privacy_rows
                if isinstance(row.get(field), (int, float))
            )
        for metric in SENSE_METRICS:
            field = f"sense_{metric}_mean"
            result[f"{metric}_macro_mean"] = mean(
                float(row[field])
                for row in sense_rows
                if isinstance(row.get(field), (int, float))
            )
        identity_rows = [
            row
            for row in group
            if isinstance(row.get("identity_compared_count"), int)
            and isinstance(row.get("identity_exact_count"), int)
        ]
        identity_compared = sum(
            row["identity_compared_count"] for row in identity_rows
        )
        identity_exact = sum(row["identity_exact_count"] for row in identity_rows)
        result["identity_compared_count"] = identity_compared or None
        result["identity_exact_count"] = identity_exact if identity_rows else None
        result["identity_exact_fraction"] = (
            identity_exact / identity_compared if identity_compared else None
        )
        rows.append(result)
    return rows


def build_manifest(metrics_dir: Path, authbench_query_dir: Path) -> dict[str, Any]:
    privacy_index = index_by_name(metrics_dir.glob("*/authbench_blabla/authbench/*.json"))
    sense_index = index_by_name(metrics_dir.rglob("*_llm.jsonl"))
    records = []

    for language, genres in GENRES.items():
        for spec in variant_specs(language):
            for genre in genres:
                suffix = spec["pipeline_suffix"]
                privacy_name = privacy_filename(language, genre, suffix)
                privacy_candidates = privacy_index.get(privacy_name, [])
                preferred_privacy_parent = (
                    metrics_dir / spec["method"] / "authbench_blabla" / "authbench"
                )
                privacy_path, privacy_status = choose_path(
                    privacy_candidates, preferred_privacy_parent
                )
                if privacy_path:
                    privacy = load_privacy(
                        privacy_path, language, genre, suffix
                    )
                else:
                    privacy = {
                        "status": privacy_status,
                        "candidates": [str(path) for path in privacy_candidates],
                    }

                sense_name = sense_filename(genre, suffix)
                sense_candidates = sense_index.get(sense_name, [])
                preferred_sense_parent = metrics_dir / language
                sense_path, sense_status = choose_path(
                    sense_candidates, preferred_sense_parent
                )
                if sense_path:
                    sense = summarize_sense(sense_path)
                else:
                    sense = {
                        "status": sense_status,
                        "candidates": [str(path) for path in sense_candidates],
                    }

                n_queries = privacy.get("n_queries")
                n_sense = sense.get("n_rows")
                count_matches = (
                    n_queries == n_sense
                    if isinstance(n_queries, int) and isinstance(n_sense, int)
                    else None
                )
                identity = None
                if spec["method"] == "jamdec":
                    source_path = (
                        authbench_query_dir
                        / language
                        / "per_genre"
                        / f"authbench_{language}_{genre}_queries.jsonl"
                    )
                    privatized_path = (
                        metrics_dir.parent
                        / "privatized"
                        / "authbench"
                        / language
                        / (
                            f"authbench_{language}_{genre}_privatized_queries_"
                            f"{suffix}.jsonl"
                        )
                    )
                    identity = summarize_identity(source_path, privatized_path)
                records.append(
                    {
                        "language": language,
                        "genre": genre,
                        **spec,
                        "privacy_candidates": [
                            str(path) for path in privacy_candidates
                        ],
                        "sense_candidates": [
                            str(path) for path in sense_candidates
                        ],
                        "privacy": privacy,
                        "sense": sense,
                        "sense_query_count_matches_privacy": count_matches,
                        "identity": identity,
                    }
                )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_dir": str(metrics_dir),
        "primary_attributor": ATTRIBUTOR,
        "sense_metrics": SENSE_METRICS,
        "languages_and_genres": GENRES,
        "method_notes": METHOD_NOTES,
        "records": records,
    }


def write_readme(
    path: Path,
    manifest: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
) -> None:
    privacy_complete = sum(
        row["privacy_status"] == "complete" for row in coverage_rows
    )
    sense_complete = sum(
        row["sense_status"] == "complete" for row in coverage_rows
    )
    count_mismatches = sum(
        row["sense_query_count_matches_privacy"] is False
        for row in coverage_rows
    )
    missing_sense_variants = sorted(
        {
            row["variant"]
            for row in coverage_rows
            if row["privacy_status"] == "complete"
            and row["sense_status"] != "complete"
        }
    )
    missing_privacy_variants = sorted(
        {
            row["variant"]
            for row in coverage_rows
            if row["privacy_status"] != "complete"
        }
    )

    text = f"""# PromptRemix paper evidence inventory

Generated: `{manifest["generated_at_utc"]}`

Primary privacy attributor: `{ATTRIBUTOR}`

## Coverage

- Expected language/genre/method cells: {len(coverage_rows)}
- Complete privacy cells: {privacy_complete}
- Complete sense-score cells: {sense_complete}
- Privacy/sense query-count mismatches: {count_mismatches}
- Variants with missing privacy cells: {", ".join(missing_privacy_variants) or "none"}
- Variants with privacy but missing canonical sense scores: {", ".join(missing_sense_variants) or "none"}

## Artifacts

- `evidence_manifest.json`: full provenance and per-cell summaries
- `evidence_coverage.csv`: flat completion matrix and source paths
- `preliminary_results_by_language.csv`: equal-weight macro means over available genre cells
- `missing_evidence.csv`: cells requiring investigation or evaluation

## Method-specific scope

- JAMDEC: {METHOD_NOTES["jamdec"]}

The preliminary CSV contains point estimates only. It must not be used for
significance claims until source-author bootstrap confidence intervals are added.
Sense scores are paired only by exact pipeline suffix; legacy or similarly named
outputs are never silently substituted.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("/gscratch/stf/hardiksr/hiatus/metrics"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("promptremix-paper/evidence"),
    )
    parser.add_argument(
        "--authbench-query-dir",
        type=Path,
        default=Path("/gscratch/ifml1/hiatus/data/raw/authbench"),
    )
    args = parser.parse_args()

    if not args.metrics_dir.is_dir():
        raise FileNotFoundError(f"Metrics directory not found: {args.metrics_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        args.metrics_dir.resolve(),
        args.authbench_query_dir.resolve(),
    )
    coverage_rows = [flatten_record(record) for record in manifest["records"]]
    preliminary_rows = aggregate_records(manifest["records"])
    missing_rows = [
        row
        for row in coverage_rows
        if row["privacy_status"] != "complete"
        or row["sense_status"] != "complete"
        or row["sense_query_count_matches_privacy"] is False
    ]

    with (args.output_dir / "evidence_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    write_csv(args.output_dir / "evidence_coverage.csv", coverage_rows)
    write_csv(
        args.output_dir / "preliminary_results_by_language.csv",
        preliminary_rows,
    )
    write_csv(
        args.output_dir / "missing_evidence.csv",
        missing_rows,
        fieldnames=list(coverage_rows[0].keys()),
    )
    write_readme(
        args.output_dir / "README.md",
        manifest,
        coverage_rows,
    )

    print(f"Wrote paper evidence inventory to {args.output_dir.resolve()}")
    print(f"Expected cells: {len(coverage_rows)}")
    print(
        "Complete privacy cells: "
        f"{sum(row['privacy_status'] == 'complete' for row in coverage_rows)}"
    )
    print(
        "Complete sense cells: "
        f"{sum(row['sense_status'] == 'complete' for row in coverage_rows)}"
    )


if __name__ == "__main__":
    main()
