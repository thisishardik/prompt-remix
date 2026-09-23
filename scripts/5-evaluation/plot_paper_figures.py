#!/usr/bin/env python3
"""Generate paper figures with one ACL-aligned theme."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2] / "promptremix-paper"
ANALYSIS = ROOT / "analysis"
OUT = ROOT / "generated" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FIG_WIDE = 6.30
FIG_COL = 3.03

LANGS = ["en", "ru", "zh", "ar"]
LANG_LABEL = {"en": "English", "ru": "Russian", "zh": "Chinese", "ar": "Arabic", "macro": "Macro"}
DESTRUCTIVE = {"paraphrasing", "stylometry", "steer"}

METHOD_ORDER = [
    "promptremix_k1",
    "promptremix_k2",
    "promptremix_k3",
    "styleremix_k1",
    "styleremix_k2",
    "styleremix_k3",
    "mutantx",
    "jamdec",
    "round_trip_mt",
    "paraphrasing",
    "stylometry",
    "steer",
]
METHOD_LABEL = {
    "promptremix_k1": r"PR $k{=}1$",
    "promptremix_k2": r"PR $k{=}2$",
    "promptremix_k3": r"PR $k{=}3$",
    "styleremix_k1": r"SR $k{=}1$",
    "styleremix_k2": r"SR $k{=}2$",
    "styleremix_k3": r"SR $k{=}3$",
    "mutantx": "Mutant-X",
    "jamdec": "JAMDEC",
    "round_trip_mt": "Round-trip MT",
    "paraphrasing": "Paraphrase",
    "stylometry": "Stylometry",
    "steer": "STEER",
}
METHOD_MARKER = {
    "promptremix_k1": "o",
    "promptremix_k2": "o",
    "promptremix_k3": "o",
    "styleremix_k1": "s",
    "styleremix_k2": "s",
    "styleremix_k3": "s",
    "mutantx": "^",
    "jamdec": "h",
    "round_trip_mt": "D",
    "paraphrasing": "X",
    "stylometry": "P",
    "steer": "v",
}

PR_COLOR = "#1b5e3b"
SR_COLOR = "#6b7280"
BAD_COLOR = "#b45309"
OTHER_COLOR = "#1f4e79"
NEUT_COLOR = "#4b5563"
LABEL_DARK = "#111827"
FILL_GRAY = "#eef1f4"
GRID_KW = {"linestyle": ":", "linewidth": 0.5, "alpha": 0.55}

LANG_STYLE = {
    "en": {"color": PR_COLOR, "marker": "o", "ls": "-"},
    "ru": {"color": "#9a3412", "marker": "s", "ls": "-"},
    "zh": {"color": NEUT_COLOR, "marker": "D", "ls": "-"},
    "ar": {"color": LABEL_DARK, "marker": "^", "ls": "-"},
    "macro": {"color": PR_COLOR, "marker": "o", "ls": "--"},
}

NAVY_CMAP = LinearSegmentedColormap.from_list(
    "pr_navy",
    ["#f8fafc", "#93c5fd", "#1f4e79", "#0f172a"],
)


def method_color(variant: str) -> str:
    if variant.startswith("promptremix"):
        return PR_COLOR
    if variant in DESTRUCTIVE:
        return BAD_COLOR
    return OTHER_COLOR


def family_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=PR_COLOR, edgecolor="none", label="PromptRemix"),
        Patch(facecolor=OTHER_COLOR, edgecolor="none", label="Utility-preserving"),
        Patch(facecolor=BAD_COLOR, edgecolor="none", label="Destructive"),
    ]


def method_marker_legend_handles() -> list[Line2D]:
    methods = [
        ("promptremix_k1", "PromptRemix"),
        ("styleremix_k1", "StyleRemix"),
        ("mutantx", "Mutant-X"),
        ("jamdec", "JAMDEC"),
        ("round_trip_mt", "Round-trip MT"),
        ("paraphrasing", "Paraphrase"),
        ("stylometry", "Stylometry"),
        ("steer", "STEER"),
    ]
    return [
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker=METHOD_MARKER[variant],
            markerfacecolor=method_color(variant),
            markeredgecolor="white",
            markeredgewidth=0.5,
            markersize=5.5,
            label=label,
        )
        for variant, label in methods
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fget(row: dict[str, str], key: str) -> float:
    return float(row[key])


def style_mpl() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "hatch.linewidth": 0.55,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def load_family(family: str) -> list[dict[str, str]]:
    rows = read_csv(ANALYSIS / "frozen_family_results_by_language.csv")
    return [r for r in rows if r["family"] == family]


def load_bootstrap() -> dict[tuple[str, str, str], dict[str, float]]:
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in read_csv(ANALYSIS / "statistics" / "bootstrap_aggregates.csv"):
        out[(row["family"], row["language"], row["variant"])] = {
            "delta": fget(row, "delta_eer"),
            "lo": fget(row, "delta_eer_ci_low"),
            "hi": fget(row, "delta_eer_ci_high"),
        }
    return out


def load_pareto_flags() -> set[tuple[str, str]]:
    flags: set[tuple[str, str]] = set()
    for row in read_csv(ANALYSIS / "privacy_utility_pareto.csv"):
        if row["on_pareto_front"].lower() == "true":
            flags.add((row["language"], row["variant"]))
    return flags


def _undominated_max(points: list[tuple[str, float, float]]) -> list[tuple[str, float, float]]:
    kept: list[tuple[str, float, float]] = []
    for name, x, y in points:
        dominated = any(
            (ox >= x and oy >= y) and (ox > x or oy > y)
            for other, ox, oy in points
            if other != name
        )
        if not dominated:
            kept.append((name, x, y))
    kept.sort(key=lambda item: item[1])
    return kept


def fig_macro_frontier_upgraded() -> Path:
    rows = {r["variant"]: r for r in load_family("canonical") if r["language"] == "macro"}
    boot = load_bootstrap()
    zeros = {
        r["variant"]: (
            float(r["sense_all_zero_rows"] or 0) / max(float(r["query_count"]), 1.0)
        )
        for r in load_family("canonical")
        if r["language"] == "macro"
    }
    coords: dict[str, tuple[float, float]] = {}
    for variant in METHOD_ORDER:
        if variant not in rows:
            continue
        r = rows[variant]
        coords[variant] = (
            fget(r, "delta_eer_macro_mean"),
            fget(r, "sense_13_macro_mean_0_20"),
        )

    baseline_pts = [
        (variant, *coords[variant])
        for variant in METHOD_ORDER
        if variant in coords and not variant.startswith("promptremix")
    ]
    baseline_front = _undominated_max(baseline_pts)
    pr_order = [v for v in ("promptremix_k1", "promptremix_k2", "promptremix_k3") if v in coords]

    label_spec = {
        "promptremix_k1": {"xytext": (0, 14), "ha": "center", "va": "bottom"},
        "promptremix_k2": {"xytext": (20, 12), "ha": "left", "va": "bottom"},
        "promptremix_k3": {"xytext": (16, -13), "ha": "left", "va": "top"},
        "styleremix_k1": {"xytext": (0, 12), "ha": "center", "va": "bottom"},
        "styleremix_k2": {"xytext": (-10, 10), "ha": "right", "va": "bottom"},
        "styleremix_k3": {"xytext": (0, -12), "ha": "center", "va": "top"},
        "mutantx": {"xytext": (-8, -11), "ha": "right", "va": "top"},
        "jamdec": {"xytext": (8, 6), "ha": "left", "va": "bottom"},
        "round_trip_mt": {"xytext": (8, 7), "ha": "left", "va": "bottom"},
        "paraphrasing": {"xytext": (8, 7), "ha": "left", "va": "bottom"},
        "stylometry": {"xytext": (0, 11), "ha": "center", "va": "bottom"},
        "steer": {"xytext": (0, 10), "ha": "center", "va": "bottom"},
    }

    fig, ax = plt.subplots(figsize=(FIG_WIDE, 2.85))
    ax.set_xlim(-0.03, 0.36)
    ax.set_ylim(-1.2, 22.4)

    bx = [p[1] for p in baseline_front]
    by = [p[2] for p in baseline_front]
    ax.fill(bx + [bx[-1], bx[0]], by + [-1.2, -1.2], color=FILL_GRAY, zorder=0, lw=0)
    ax.plot(
        bx,
        by,
        color=SR_COLOR,
        linestyle=(0, (5, 2.5)),
        linewidth=1.6,
        zorder=1,
    )
    ax.plot(
        [coords[v][0] for v in pr_order],
        [coords[v][1] for v in pr_order],
        color=PR_COLOR,
        linewidth=2.0,
        zorder=2,
    )

    for variant, (x, y) in coords.items():
        is_pr = variant.startswith("promptremix")
        color = method_color(variant)
        size = 95 if variant == "promptremix_k3" else (78 if is_pr else 52 + 160 * zeros.get(variant, 0.0))
        ax.scatter(
            x,
            y,
            s=size,
            marker=METHOD_MARKER[variant],
            color=color,
            edgecolors="white",
            linewidths=0.6,
            alpha=0.95,
            zorder=4 if is_pr else 3,
        )
        if is_pr and ("num_styles", "macro", variant) in boot:
            b = boot[("num_styles", "macro", variant)]
            ax.errorbar(
                x,
                y,
                xerr=[[x - b["lo"]], [b["hi"] - x]],
                fmt="none",
                ecolor=PR_COLOR,
                elinewidth=1.2,
                capsize=3,
                zorder=3,
            )
        spec = label_spec[variant]
        label_color = PR_COLOR if is_pr else ("#9a3412" if variant in DESTRUCTIVE else LABEL_DARK)
        ax.annotate(
            METHOD_LABEL[variant],
            (x, y),
            textcoords="offset points",
            xytext=spec["xytext"],
            ha=spec["ha"],
            va=spec["va"],
            fontsize=8,
            color=label_color,
            fontweight="bold" if is_pr else "normal",
            zorder=5,
        )

    ax.set_xlabel(r"$\Delta$EER $\uparrow$")
    ax.set_ylabel(r"SENSE $\uparrow$")
    ax.grid(True, **GRID_KW, zorder=0)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=SR_COLOR,
                linestyle=(0, (5, 2.5)),
                linewidth=1.6,
                label="Baseline frontier",
            ),
            *family_legend_handles(),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#d1d5db",
        handlelength=2.4,
    )
    fig.tight_layout()
    return savefig(fig, OUT / "privacy_utility_frontier_wide.pdf")


def fig_macro_frontier_column() -> Path:
    """One-column layout of the macro privacy--utility frontier."""
    rows = {r["variant"]: r for r in load_family("canonical") if r["language"] == "macro"}
    coords: dict[str, tuple[float, float]] = {}
    for variant in METHOD_ORDER:
        if variant not in rows:
            continue
        r = rows[variant]
        coords[variant] = (
            fget(r, "delta_eer_macro_mean"),
            fget(r, "sense_13_macro_mean_0_20"),
        )
    baseline_pts = [
        (variant, *coords[variant])
        for variant in METHOD_ORDER
        if variant in coords and not variant.startswith("promptremix")
    ]
    baseline_front = _undominated_max(baseline_pts)
    pr_order = [v for v in ("promptremix_k1", "promptremix_k2", "promptremix_k3") if v in coords]
    label_spec = {
        "promptremix_k1": {"xytext": (2, 7), "ha": "left", "va": "bottom"},
        "promptremix_k2": {"xytext": (8, 2), "ha": "left", "va": "bottom"},
        "promptremix_k3": {"xytext": (6, -6), "ha": "left", "va": "top"},
        "styleremix_k2": {"xytext": (6, 3), "ha": "left", "va": "bottom"},
        "styleremix_k1": {"xytext": (0, 12), "ha": "center", "va": "bottom"},
        "styleremix_k3": {"xytext": (0, -7), "ha": "center", "va": "top"},
        "mutantx": {"xytext": (-3, -6), "ha": "right", "va": "top"},
        "jamdec": {"xytext": (4, 5), "ha": "left", "va": "bottom"},
        "round_trip_mt": {"xytext": (4, 3), "ha": "left", "va": "bottom"},
        "paraphrasing": {"xytext": (4, 3), "ha": "left", "va": "bottom"},
        "stylometry": {"xytext": (-3, 5), "ha": "right", "va": "bottom"},
    }

    x_right = coords["steer"][0] + 0.042
    fig, ax = plt.subplots(figsize=(FIG_COL, 2.82))
    ax.set_xlim(-0.025, x_right)
    ax.set_ylim(-1.2, 23.0)
    bx = [p[1] for p in baseline_front]
    by = [p[2] for p in baseline_front]
    ax.fill(bx + [bx[-1], bx[0]], by + [-1.2, -1.2], color=FILL_GRAY, zorder=0, lw=0)
    ax.plot(
        bx,
        by,
        color=SR_COLOR,
        linestyle=(0, (5, 2.5)),
        linewidth=1.4,
        zorder=1,
    )
    ax.plot(
        [coords[v][0] for v in pr_order],
        [coords[v][1] for v in pr_order],
        color=PR_COLOR,
        linewidth=1.8,
        zorder=2,
    )
    for variant, (x, y) in coords.items():
        is_pr = variant.startswith("promptremix")
        ax.scatter(
            x,
            y,
            s=46,
            marker=METHOD_MARKER[variant],
            color=method_color(variant),
            edgecolors="white",
            linewidths=0.5,
            alpha=0.95,
            zorder=4 if is_pr else 3,
        )
        if variant == "steer":
            ax.annotate(
                METHOD_LABEL[variant],
                (x, y),
                xytext=(x + 0.018, y + 0.72),
                textcoords="data",
                ha="center",
                va="bottom",
                fontsize=6,
                color="#9a3412",
                zorder=5,
            )
            continue
        spec = label_spec.get(variant)
        if spec is None:
            continue
        ax.annotate(
            METHOD_LABEL[variant],
            (x, y),
            textcoords="offset points",
            xytext=spec["xytext"],
            ha=spec["ha"],
            va=spec["va"],
            fontsize=6,
            color=PR_COLOR if is_pr else ("#9a3412" if variant in DESTRUCTIVE else LABEL_DARK),
            fontweight="bold" if is_pr else "normal",
            zorder=5,
        )
    ax.set_xlabel(r"$\Delta$EER $\uparrow$")
    ax.set_ylabel(r"SENSE $\uparrow$")
    ax.grid(True, **GRID_KW, zorder=0)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=SR_COLOR,
                linestyle=(0, (5, 2.5)),
                linewidth=1.4,
                label="Baseline frontier",
            ),
            *family_legend_handles(),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#d1d5db",
        handlelength=2.0,
        fontsize=6,
    )
    fig.tight_layout()
    return savefig(fig, OUT / "privacy_utility_frontier.pdf")


def fig_frontier_languages() -> Path:
    rows = load_family("canonical")
    pareto = load_pareto_flags()
    boot = load_bootstrap()
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDE, 3.95), sharex=False, sharey=True)
    for ax, lang in zip(axes.ravel(), LANGS):
        lang_rows = {r["variant"]: r for r in rows if r["language"] == lang}
        for variant in METHOD_ORDER:
            if variant not in lang_rows:
                continue
            r = lang_rows[variant]
            x = fget(r, "delta_eer_macro_mean")
            y = fget(r, "sense_13_macro_mean_0_20")
            is_pr = variant.startswith("promptremix")
            color = method_color(variant)
            size = 88 if variant == "promptremix_k3" else (64 if is_pr else 42)
            edge = LABEL_DARK if (lang, variant) in pareto else "white"
            ax.scatter(
                x,
                y,
                s=size,
                marker=METHOD_MARKER[variant],
                color=color,
                edgecolors=edge,
                linewidths=0.9 if (lang, variant) in pareto else 0.5,
                zorder=4 if is_pr else 3,
                alpha=0.95,
            )
            if is_pr and ("num_styles", lang, variant) in boot:
                b = boot[("num_styles", lang, variant)]
                ax.errorbar(
                    x,
                    y,
                    xerr=[[x - b["lo"]], [b["hi"] - x]],
                    fmt="none",
                    ecolor=PR_COLOR,
                    elinewidth=1.0,
                    capsize=2,
                    zorder=2,
                )
        ax.set_title(LANG_LABEL[lang])
        ax.set_xlabel(r"$\Delta$EER $\uparrow$")
        ax.set_ylim(-0.8, 21.5)
        ax.grid(True, **GRID_KW)
    axes[0, 0].set_ylabel(r"SENSE $\uparrow$")
    axes[1, 0].set_ylabel(r"SENSE $\uparrow$")
    fig.legend(
        handles=method_marker_legend_handles(),
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=1.1,
        handletextpad=0.4,
    )
    fig.legend(
        handles=family_legend_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.80])
    return savefig(fig, OUT / "privacy_utility_frontier_languages.pdf")


def fig_operating_points() -> Path:
    boot = load_bootstrap()
    num = load_family("num_styles")
    rand_mean = read_csv(ANALYSIS / "random-axis" / "random_axis_mean_std.csv")

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDE, 2.90))
    ks = [1, 2, 3]
    ax = axes[0]
    for lang in LANGS + ["macro"]:
        st = LANG_STYLE[lang]
        xs, ys, ylo, yhi = [], [], [], []
        for k in ks:
            variant = f"promptremix_k{k}"
            row = next(r for r in num if r["language"] == lang and r["variant"] == variant)
            xs.append(k)
            ys.append(fget(row, "delta_eer_macro_mean"))
            b = boot.get(("num_styles", lang, variant))
            ylo.append(b["lo"] if b else ys[-1])
            yhi.append(b["hi"] if b else ys[-1])
        ax.plot(
            xs,
            ys,
            color=st["color"],
            linestyle=st["ls"],
            marker=st["marker"],
            label=LANG_LABEL[lang],
            linewidth=1.8,
            markersize=6.5,
            markerfacecolor=st["color"],
            markeredgecolor="white",
            markeredgewidth=0.4,
        )
        ax.fill_between(xs, ylo, yhi, color=st["color"], alpha=0.10)
    ax.set_xticks(ks)
    ax.set_xlabel(r"Style directions $k$")
    ax.set_ylabel(r"$\Delta$EER")
    ax.set_title("Style count")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), handlelength=2.6)
    ax.grid(True, **GRID_KW)

    ax = axes[1]
    y_pos = np.arange(len(LANGS) + 1)
    labels = [LANG_LABEL[lang] for lang in LANGS] + ["Macro"]
    langs_plot = LANGS + ["macro"]
    rr_vals, rand_vals, rand_std = [], [], []
    for lang in langs_plot:
        row = next(r for r in rand_mean if r["language"] == lang)
        rr_vals.append(fget(row, "round_robin_delta_eer"))
        rand_vals.append(fget(row, "delta_eer_seed_mean"))
        rand_std.append(fget(row, "delta_eer_seed_std"))
    ax.hlines(y_pos, rr_vals, rand_vals, color=SR_COLOR, linewidth=1.2, zorder=1)
    ax.scatter(rr_vals, y_pos, color=PR_COLOR, marker="o", s=48, label="Round-robin", zorder=3, edgecolors="white", linewidths=0.4)
    ax.errorbar(
        rand_vals,
        y_pos,
        xerr=rand_std,
        fmt="D",
        color=NEUT_COLOR,
        markersize=5.5,
        label="Random (5 seeds)",
        zorder=3,
        capsize=2.5,
        markeredgecolor="white",
        markeredgewidth=0.4,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$\Delta$EER")
    ax.set_title("Axis selection")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(True, axis="x", **GRID_KW)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return savefig(fig, OUT / "operating_points.pdf")


def fig_canonical_vs_short() -> Path:
    canon = {r["variant"]: r for r in load_family("canonical") if r["language"] == "macro"}
    short_pr = {
        r["variant"]: r for r in load_family("short_text_promptremix") if r["language"] == "macro"
    }
    short_base = {
        r["variant"]: r for r in load_family("short_text_baselines") if r["language"] == "macro"
    }
    short = {**short_base, **short_pr}
    variants = [
        "promptremix_k1",
        "promptremix_k3",
        "styleremix_k3",
        "mutantx",
        "round_trip_mt",
        "paraphrasing",
        "stylometry",
        "steer",
    ]
    fig, ax = plt.subplots(figsize=(FIG_COL, 2.45))
    y = np.arange(len(variants))
    for i, variant in enumerate(variants):
        c = fget(canon[variant], "delta_eer_macro_mean")
        s = fget(short[variant], "delta_eer_macro_mean")
        color = method_color(variant)
        ax.plot([c, s], [i, i], color=color, linewidth=1.6, alpha=0.9)
        ax.scatter([c], [i], color=color, marker="o", s=38, zorder=3, edgecolors="white", linewidths=0.4)
        ax.scatter([s], [i], color=color, marker=">", s=42, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([METHOD_LABEL[v] for v in variants])
    ax.set_xlabel(r"$\Delta$EER")
    ax.grid(True, axis="x", **GRID_KW)
    ax.invert_yaxis()
    fig.tight_layout()
    return savefig(fig, OUT / "canonical_vs_short.pdf")


def fig_failure_rates() -> Path:
    family = {
        r["variant"]: r
        for r in load_family("canonical")
        if r["language"] == "macro"
    }
    order = [
        "promptremix_k3",
        "styleremix_k3",
        "mutantx",
        "jamdec",
        "round_trip_mt",
        "paraphrasing",
        "stylometry",
        "steer",
    ]
    labels, fracs, colors, hatches = [], [], [], []
    for variant in order:
        row = family[variant]
        frac = 100.0 * float(row["sense_all_zero_rows"]) / float(row["query_count"])
        labels.append(METHOD_LABEL[variant])
        fracs.append(frac)
        colors.append(method_color(variant))
        if variant in DESTRUCTIVE:
            hatches.append("///")
        elif variant.startswith("promptremix"):
            hatches.append("")
        else:
            hatches.append("..")

    fig, ax = plt.subplots(figsize=(FIG_COL, 2.40))
    y = np.arange(len(labels))
    ax.barh(y, fracs, color=colors, edgecolor=LABEL_DARK, linewidth=0.4, height=0.72, hatch=hatches)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("All-zero SENSE rows (%)")
    ax.grid(True, axis="x", **GRID_KW)
    ax.invert_yaxis()
    xmax = max(fracs)
    ax.set_xlim(0, xmax + 12)
    for i, frac in enumerate(fracs):
        ax.text(frac + 1.2, i, f"{frac:.1f}%", va="center", fontsize=8, color=LABEL_DARK)
    fig.tight_layout()
    return savefig(fig, OUT / "sense_failure_rates.pdf")


def fig_language_heatmap() -> Path:
    rows = load_family("canonical")
    variants = [
        "promptremix_k3",
        "styleremix_k3",
        "mutantx",
        "jamdec",
        "round_trip_mt",
        "paraphrasing",
        "stylometry",
        "steer",
    ]
    mat = np.zeros((len(variants), len(LANGS)))
    for i, variant in enumerate(variants):
        for j, lang in enumerate(LANGS):
            row = next(r for r in rows if r["language"] == lang and r["variant"] == variant)
            mat[i, j] = fget(row, "delta_eer_macro_mean")

    fig, ax = plt.subplots(figsize=(FIG_WIDE, 2.45))
    im = ax.imshow(mat, aspect="auto", cmap=NAVY_CMAP, vmin=0.0, vmax=float(np.max(mat)))
    ax.set_xticks(range(len(LANGS)))
    ax.set_xticklabels([LANG_LABEL[lang] for lang in LANGS])
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels([METHOD_LABEL[v] for v in variants])
    threshold = 0.45 * float(np.max(mat))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if val >= threshold else LABEL_DARK,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta$EER")
    cbar.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    return savefig(fig, OUT / "language_delta_heatmap.pdf")


def fig_paired_forest() -> Path:
    paired = read_csv(ANALYSIS / "statistics" / "paired_privacy_comparisons.csv")
    wanted = [
        ("num_styles", "macro", "promptremix_k3_minus_promptremix_k1", r"Macro $k{=}3{-}1$"),
        ("num_styles", "en", "promptremix_k3_minus_promptremix_k1", r"English $k{=}3{-}1$"),
        ("num_styles", "ru", "promptremix_k3_minus_promptremix_k1", r"Russian $k{=}3{-}1$"),
        ("num_styles", "zh", "promptremix_k3_minus_promptremix_k1", r"Chinese $k{=}3{-}1$"),
        ("num_styles", "ar", "promptremix_k3_minus_promptremix_k1", r"Arabic $k{=}3{-}1$"),
        ("random_axis", "macro", "random_seed_mean_minus_round_robin", r"Macro random$-$RR"),
        ("short_text_promptremix", "macro", "promptremix_k3_minus_promptremix_k1", r"Short $k{=}3{-}1$"),
    ]
    labels, diffs, los, his = [], [], [], []
    for family, lang, comparison, label in wanted:
        row = next(
            r
            for r in paired
            if r["family"] == family and r["language"] == lang and r["comparison"] == comparison
        )
        labels.append(label)
        diffs.append(fget(row, "delta_eer_difference"))
        los.append(fget(row, "delta_eer_difference_ci_low"))
        his.append(fget(row, "delta_eer_difference_ci_high"))

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(FIG_COL, 2.45))
    ax.axvline(0.0, color=SR_COLOR, linewidth=1.0, linestyle="--")
    for i, (d, lo, hi) in enumerate(zip(diffs, los, his)):
        if lo > 0:
            color, marker, ls = PR_COLOR, "o", "-"
            facecolor = color
        elif hi < 0:
            color, marker, ls = BAD_COLOR, "s", "-"
            facecolor = color
        else:
            color, marker, ls = NEUT_COLOR, "o", "--"
            facecolor = "white"
        ax.plot([lo, hi], [i, i], color=color, linewidth=2.0, linestyle=ls, clip_on=False)
        ax.scatter(
            [d],
            [i],
            s=42,
            marker=marker,
            color=facecolor,
            edgecolors=color,
            linewidths=1.0,
            zorder=3,
            clip_on=False,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Paired $\Delta$EER difference")
    ax.grid(True, axis="x", **GRID_KW)
    ax.invert_yaxis()
    pad = 0.02
    ax.set_xlim(min(los) - pad, max(his) + pad)
    fig.tight_layout()
    return savefig(fig, OUT / "paired_difference_forest.pdf")


def main() -> None:
    style_mpl()
    paths = [
        fig_macro_frontier_column(),
        fig_frontier_languages(),
        fig_operating_points(),
        fig_canonical_vs_short(),
        fig_failure_rates(),
        fig_language_heatmap(),
        fig_paired_forest(),
    ]
    for path in paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
