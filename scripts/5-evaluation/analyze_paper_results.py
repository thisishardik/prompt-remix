#!/usr/bin/env python3
"""Freeze and summarize complete PromptRemix paper results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


SENSE_DIMENSIONS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("promptremix-paper/evidence/evidence_coverage.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("promptremix-paper/analysis"),
    )
    parser.add_argument(
        "--exclude-method",
        action="append",
        default=[],
        help="Method to mark pending and exclude from the frozen analysis.",
    )
    return parser.parse_args()


def optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in (
            "eer_original",
            "eer_privatized",
            "delta_eer",
            "sense_n_rows",
            "sense_n_complete_rows",
            "sense_complete_fraction",
            "sense_cosine_similarity_mean",
        ):
            row[field] = optional_float(row.get(field))
        row["num_styles"] = optional_int(row.get("num_styles"))
        row["identity_compared_count"] = optional_int(
            row.get("identity_compared_count")
        )
        row["identity_exact_count"] = optional_int(row.get("identity_exact_count"))
        for dimension in SENSE_DIMENSIONS:
            field = f"sense_{dimension}_mean"
            row[field] = optional_float(row.get(field))
    return rows


def inspect_sense_file(path: Path) -> dict[str, Any]:
    n_rows = 0
    n_all_zero = 0
    n_missing = 0
    for line in path.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        n_rows += 1
        row = json.loads(line)
        values = [row.get(dimension) for dimension in SENSE_DIMENSIONS]
        if any(value is None for value in values):
            n_missing += 1
        elif all(value == 0 for value in values):
            n_all_zero += 1
    return {
        "n_rows": n_rows,
        "n_all_zero": n_all_zero,
        "n_missing": n_missing,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    rows: list[dict[str, Any]],
    language: str,
    variant: str,
) -> dict[str, Any]:
    sense_means = [
        mean(row[f"sense_{dimension}_mean"] for dimension in SENSE_DIMENSIONS)
        for row in rows
    ]
    n_zero = sum(row["sense_zero_rows"] for row in rows)
    n_sense = sum(row["sense_audited_rows"] for row in rows)
    identity_compared = sum(
        row["identity_compared_count"] or 0 for row in rows
    )
    identity_exact = sum(row["identity_exact_count"] or 0 for row in rows)
    return {
        "language": language,
        "variant": variant,
        "method": rows[0]["method"],
        "num_styles": rows[0]["num_styles"],
        "genre_cells": len(rows),
        "query_count": sum(int(row["n_queries"]) for row in rows),
        "delta_eer_macro_mean": mean(row["delta_eer"] for row in rows),
        "eer_original_macro_mean": mean(row["eer_original"] for row in rows),
        "eer_privatized_macro_mean": mean(
            row["eer_privatized"] for row in rows
        ),
        "sense_13_macro_mean_0_20": mean(sense_means),
        "sense_13_normalized_0_1": mean(sense_means) / 20.0,
        "cosine_similarity_macro_mean": mean(
            row["sense_cosine_similarity_mean"] for row in rows
        ),
        "zero_rubric_rows": n_zero,
        "sense_rows": n_sense,
        "zero_rubric_fraction": n_zero / n_sense if n_sense else None,
        "identity_compared_count": identity_compared or None,
        "identity_exact_count": identity_exact if identity_compared else None,
        "identity_exact_fraction": (
            identity_exact / identity_compared if identity_compared else None
        ),
    }


def pareto_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)
    for language, group in by_language.items():
        for candidate in group:
            dominated_by = [
                other["variant"]
                for other in group
                if other["variant"] != candidate["variant"]
                and other["delta_eer_macro_mean"]
                >= candidate["delta_eer_macro_mean"]
                and other["sense_13_macro_mean_0_20"]
                >= candidate["sense_13_macro_mean_0_20"]
                and (
                    other["delta_eer_macro_mean"]
                    > candidate["delta_eer_macro_mean"]
                    or other["sense_13_macro_mean_0_20"]
                    > candidate["sense_13_macro_mean_0_20"]
                )
            ]
            result.append(
                {
                    "language": language,
                    "variant": candidate["variant"],
                    "delta_eer_macro_mean": candidate[
                        "delta_eer_macro_mean"
                    ],
                    "sense_13_macro_mean_0_20": candidate[
                        "sense_13_macro_mean_0_20"
                    ],
                    "on_pareto_front": not dominated_by,
                    "dominated_by": ";".join(sorted(dominated_by)),
                }
            )
    return result


def audit_severity(fraction: float) -> str:
    if fraction >= 0.8:
        return "critical"
    if fraction >= 0.25:
        return "high"
    if fraction >= 0.05:
        return "review"
    return "low"


def main() -> None:
    args = parse_args()
    all_rows = load_rows(args.coverage)
    excluded = set(args.exclude_method)
    pending = [row for row in all_rows if row["method"] in excluded]
    included = [row for row in all_rows if row["method"] not in excluded]

    incomplete = [
        row
        for row in included
        if row["privacy_status"] != "complete"
        or row["sense_status"] != "complete"
        or row["sense_query_count_matches_privacy"] != "True"
    ]
    if incomplete:
        cells = [
            f"{row['language']}/{row['genre']}/{row['variant']}"
            for row in incomplete
        ]
        raise RuntimeError(f"Core analysis has incomplete cells: {cells}")

    sense_cache: dict[str, dict[str, Any]] = {}
    frozen_rows = []
    audit_groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"n_rows": 0, "n_all_zero": 0, "n_missing": 0}
    )
    for row in included:
        sense_path = str(row["sense_path"])
        if sense_path not in sense_cache:
            sense_cache[sense_path] = inspect_sense_file(Path(sense_path))
        audit = sense_cache[sense_path]
        row["sense_audited_rows"] = audit["n_rows"]
        row["sense_zero_rows"] = audit["n_all_zero"]
        row["sense_missing_rows"] = audit["n_missing"]
        frozen_rows.append(row)
        group = audit_groups[(row["language"], row["variant"])]
        for key in ("n_rows", "n_all_zero", "n_missing"):
            group[key] += audit[key]

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in frozen_rows:
        groups[(row["language"], row["variant"])].append(row)
        groups[("macro", row["variant"])].append(row)
    aggregate_rows = [
        aggregate(rows, language, variant)
        for (language, variant), rows in sorted(groups.items())
    ]

    audit_rows = []
    for (language, variant), values in sorted(audit_groups.items()):
        fraction = (
            values["n_all_zero"] / values["n_rows"]
            if values["n_rows"]
            else 0.0
        )
        audit_rows.append(
            {
                "language": language,
                "variant": variant,
                "sense_rows": values["n_rows"],
                "all_zero_rubric_rows": values["n_all_zero"],
                "missing_rubric_rows": values["n_missing"],
                "all_zero_rubric_fraction": fraction,
                "severity": audit_severity(fraction),
                "excluded_from_results": False,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "frozen_core_cells.csv", frozen_rows)
    write_csv(
        args.output_dir / "core_results_by_language.csv",
        aggregate_rows,
    )
    write_csv(args.output_dir / "sense_quality_audit.csv", audit_rows)
    write_csv(
        args.output_dir / "privacy_utility_pareto.csv",
        pareto_rows(aggregate_rows),
    )

    macro_rows = {
        row["variant"]: row
        for row in aggregate_rows
        if row["language"] == "macro"
    }
    promptremix = {
        variant: macro_rows[variant]
        for variant in ("promptremix_k1", "promptremix_k2", "promptremix_k3")
    }
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_source": str(args.coverage.resolve()),
        "excluded_methods_pending": sorted(excluded),
        "pending_cells": len(pending),
        "included_complete_cells": len(frozen_rows),
        "privacy_sense_count_mismatches": 0,
        "aggregation": "equal-weight macro mean over language/genre cells",
        "sense_composite": "equal-weight mean of 13 rubric dimensions, each 0-20",
        "zero_rubric_policy": (
            "Retained as observed utility outcomes; never treated as missing."
        ),
        "promptremix_macro": promptremix,
    }
    with (args.output_dir / "analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
