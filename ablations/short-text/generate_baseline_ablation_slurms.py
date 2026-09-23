#!/usr/bin/env python3
"""Generate one short-text AuthBench Slurm array in every baseline folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRATCH = "/gscratch/stf/hardiksr/hiatus"
ATTRIBUTOR = "Blablablab/multilingual-style-representation"
SENSE_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"
SAMPLE_ROOT = f"{SCRATCH}/data/sampled/authbench_short_text/bottom25"


@dataclass(frozen=True)
class Baseline:
    name: str
    script: str
    base_suffix: str
    extra_args: tuple[str, ...] = ()
    style_counts: bool = False


BASELINES = [
    Baseline(
        "mutantx",
        "scripts/7-baselines/mutantx/obfuscate_mutantx.py",
        "mutantx",
        (
            "--model-name",
            "Qwen/Qwen2.5-3B-Instruct",
            "--pop-size",
            "10",
            "--gens",
            "15",
        ),
    ),
    Baseline(
        "steer",
        "scripts/7-baselines/steer/obfuscate_steer.py",
        "steer",
        (
            "--model-ckpt",
            "/gscratch/stf/kogolobo/steer_ckpt/ckp_2900.pth",
            "--target-style",
            "english_tweet",
        ),
    ),
    Baseline(
        "paraphrasing",
        "scripts/7-baselines/paraphrasing/obfuscate_paraphrasing.py",
        "paraphrasing",
    ),
    Baseline(
        "round_trip_mt",
        "scripts/7-baselines/round_trip_mt/obfuscate_round_trip_mt.py",
        "round_trip_mt",
    ),
    Baseline(
        "stylometry",
        "scripts/7-baselines/stylometry/obfuscate_stylometric.py",
        "stylometry",
    ),
    Baseline(
        "styleremix",
        "scripts/7-baselines/styleremix/obfuscate_styleremix.py",
        "obfuscate_eval_styleremix_authbench",
        style_counts=True,
    ),
    Baseline(
        "jamdec",
        "scripts/7-baselines/jamdec/obfuscate_jamdec.py",
        "obfuscate_eval_jamdec_authbench",
        ("--cache-dir", f"{SCRATCH}/cache"),
    ),
]


def render(baseline: Baseline) -> str:
    extra_args = "\n".join(f'    "{arg}"' for arg in baseline.extra_args)
    if baseline.style_counts:
        settings = """num_styles_values=(1 2 3)"""
        suffix_logic = """for num_styles in "${num_styles_values[@]}"; do
    suffix="obfuscate_eval_styleremix_authbench_${LANGUAGE}-baseline-${num_styles}-short-bottom25"
    style_args=(--top-n-styles-to-change "${num_styles}" --config-file "${CONFIG_FILE}")"""
        close_settings = "done"
        style_args = '            "${style_args[@]}" \\\n'
    else:
        settings = """num_styles_values=(0)"""
        if baseline.name == "jamdec":
            suffix = (
                'obfuscate_eval_jamdec_authbench_${LANGUAGE}'
                '-baseline-short-bottom25'
            )
        else:
            suffix = f"{baseline.base_suffix}-short-bottom25"
        suffix_logic = f"""for _ in "${{num_styles_values[@]}}"; do
    suffix="{suffix}"
    style_args=()"""
        close_settings = "done"
        style_args = '            "${style_args[@]}" \\\n'

    return f"""#!/bin/bash
#SBATCH --job-name=short_{baseline.name}_%a
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=ifml1
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=90GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=80:00:00
#SBATCH --array=0-3
#SBATCH --chdir={REPO}
#SBATCH --export=all
#SBATCH --output={SCRATCH}/logs/short_{baseline.name}_%A_%a.out
#SBATCH --error={SCRATCH}/logs/short_{baseline.name}_%A_%a.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0 coenv/jdk/17.0.8

languages=(en ru zh ar)
LANGUAGE="${{languages[$SLURM_ARRAY_TASK_ID]}}"
case "${{LANGUAGE}}" in
    en)
        genres=(blog ecommerce_reviews literature news poetry qna research_paper)
        CONFIG_FILE="scripts/4-obfuscation/config/remix_config_en.json"
        ;;
    ru)
        genres=(literature news poetry qna social_media)
        CONFIG_FILE="scripts/4-obfuscation/config/remix_config_ru.json"
        ;;
    zh)
        genres=(ecommerce_reviews literature media_reviews news poetry social_media)
        CONFIG_FILE="scripts/4-obfuscation/config/remix_config_zh.json"
        ;;
    ar)
        genres=(literature news poetry social_media)
        CONFIG_FILE="scripts/4-obfuscation/config/remix_config_ar3.json"
        ;;
esac

SAMPLE_ROOT="{SAMPLE_ROOT}"
PRIVACY_ROOT="{SCRATCH}/metrics/paper_experiments/short_text/baselines/{baseline.name}/privacy/${{LANGUAGE}}"
SENSE_ROOT="{SCRATCH}/metrics/paper_experiments/short_text/baselines/{baseline.name}/sense/${{LANGUAGE}}"
mkdir -p "${{PRIVACY_ROOT}}" "${{SENSE_ROOT}}"
extra_args=(
{extra_args}
)

{settings}
{suffix_logic}
    for genre in "${{genres[@]}}"; do
        sample="${{SAMPLE_ROOT}}/${{LANGUAGE}}/${{genre}}/authbench_${{LANGUAGE}}_${{genre}}_sampled_ids.json"
        if [[ ! -s "${{sample}}" ]]; then
            echo "Missing short-text sample: ${{sample}}" >&2
            continue
        fi

        read -r original privatized _ _ < <(
            python scripts/7-baselines/authbench/get_authbench_paths.py \\
                --language "${{LANGUAGE}}" \\
                --genre "${{genre}}" \\
                --pipeline-suffix "${{suffix}}"
        )
        if [[ ! -s "${{privatized}}" ]]; then
            python "{baseline.script}" \\
                --dataset authbench \\
                --language "${{LANGUAGE}}" \\
                --genre "${{genre}}" \\
                --output-suffix "${{suffix}}" \\
                --sampled-data-filepath "${{sample}}" \\
{style_args}                "${{extra_args[@]}}"
        fi

        python scripts/7-baselines/authbench/authbench_eval.py \\
            --model "{ATTRIBUTOR}" \\
            --language "${{LANGUAGE}}" \\
            --dataset authbench \\
            --genre "${{genre}}" \\
            --pipeline-suffix "${{suffix}}" \\
            --output-dir "${{PRIVACY_ROOT}}" \\
            --batch-size 32 \\
            --sampled-data-filepath "${{sample}}"

        sense="${{SENSE_ROOT}}/${{genre}}_${{suffix}}_llm.jsonl"
        if [[ ! -s "${{sense}}" ]]; then
            python scripts/5-evaluation/score_sense.py \\
                --text-original-jsonl "${{original}}" \\
                --text-privatized-jsonl "${{privatized}}" \\
                --output-path "${{sense}}" \\
                --model-path "{SENSE_MODEL}" \\
                --batch-size 16 \\
                --sampled-data-filepath "${{sample}}"
        fi
    done
{close_settings}
"""


def main() -> None:
    for baseline in BASELINES:
        output_dir = (
            REPO / "scripts" / "7-baselines" / baseline.name / "ablation"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / "short_text_authbench.slurm"
        output.write_text(render(baseline), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
