#!/bin/bash

#SBATCH --job-name=obfuscate_eval_styleremix_en
#SBATCH --mail-user=hardiksr@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=cse
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=150GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=24:00:00 

#SBATCH --chdir=/mmfs1/home/hardiksr/repository/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_styleremix_en.out
#SBATCH --error=/gscratch/stf/hardiksr/hiatus/logs/obfuscate_eval_styleremix_en.err
set -e

source /mmfs1/home/hardiksr/.bashrc
conda activate vllm
module load cuda/12.8.1 gcc/12.3.0

LANGUAGE="en"
MODEL="meta-llama/Meta-Llama-3-8B"

num_styles_to_change=(1 2 3)
total_settings=${#num_styles_to_change[@]}
en_genres=("perGenre-HRS2.1" "perGenre-HRS2.2" "perGenre-HRS2.3" "perGenre-HRS2.4" "perGenre-HRS2.5" "en_crossGenre" "perGenre-HRS3.1" "perGenre-HRS3.2" "perGenre-HRS3.3" "perGenre-HRS3.4" "perGenre-HRS3.5" "en_short_crossGenre")
METRICS_DIR="/gscratch/stf/hardiksr/hiatus/metrics/styleremix"

for ((i=0; i<total_settings; i++)); do
    echo "Processing setting $i of ${total_settings}"

    num_styles=${num_styles_to_change[$i]}
    echo "Processing num_styles: $num_styles"

    # Pipeline suffix.
    pipeline_suffix="obfuscate_eval_styleremix_en-baseline-${num_styles}"

    for genre in "${en_genres[@]}"; do
        # Setup paths
        read -r input_queries_path privatized_queries_path candidate_queries_path ground_truth_path< <(python scripts/4-obfuscation/get_hrs_paths.py --genre $genre --language "en" --pipeline-suffix ${pipeline_suffix})

        tmp_path="/gscratch/scrubbed/hardiksr/hiatus/privatized/en/${genre}_${num_styles}_${pipeline_suffix}.json"
        
        echo "Input queries path: ${input_queries_path}"
        echo "Privatized queries path: ${privatized_queries_path}"

        # Run StyleRemix Baseline
        python scripts/7-baselines/styleremix/obfuscate_styleremix.py \
            --dataset hrs \
            --language "en" \
            --genre "${genre}" \
            --top-n-styles-to-change "${num_styles}" \
            --output-suffix "${pipeline_suffix}" \
            --model-ckpt "${MODEL}"
    done
done


# for num_styles in "${num_styles_to_change[@]}"; do
#     PIPELINE_SUFFIX="obfuscate_eval_styleremix_en-baseline-${num_styles}"
#     for GENRE in "${en_genres[@]}"; do
#         echo "Evaluating genre: ${GENRE}"

#         # AuthBench Eval
#         echo "Running AuthBench Delta EER Eval for (${LANGUAGE}, ${GENRE})"
#         conda activate vllm
#         python scripts/7-baselines/authbench/authbench_eval.py \
#             --model "${MODEL}" \
#             --language "${LANGUAGE}" \
#             --dataset hrs \
#             --genre "${GENRE}" \
#             --pipeline-suffix "${PIPELINE_SUFFIX}" \
#             --output-dir "${METRICS_DIR}/authbench" \
#             --batch-size 32

#         # Hiatus Eval
#         echo "Running Hiatus Delta EER Eval for (${LANGUAGE}, ${GENRE})"
#         conda activate hiatus-eval
#         python scripts/5-evaluation/score_privacy_consistency.py \
#             --model "Hydrant0585/ALERTTextAuthorshipAttribution" \
#             --genre "${GENRE}" \
#             --language "${LANGUAGE}" \
#             --pipeline-suffix "${PIPELINE_SUFFIX}" \
#             --batch-size 32
#     done
# done


# echo "Accumulating Delta EER Metrics"
# conda activate vllm

# METHODS=()
# for num_styles in "${num_styles_to_change[@]}"; do
#     METHODS+=("obfuscate_eval_styleremix_en-baseline-${num_styles}")
# done

# python scripts/5-evaluation/accumulate_delta_eer.py \
#     --method "${METHODS[@]}" \
#     --group-name "obfuscate_eval_styleremix_en-baseline" \
#     --language "${LANGUAGE}" \
#     --model-tag "${MODEL//\//_}" \
#     --genres "${en_genres[@]}" \
#     --baseline-dir "${METRICS_DIR}" \
#     --authbench-dir "${METRICS_DIR}/authbench" \
#     --hiatus-dir "/gscratch/stf/hardiksr/hiatus/metrics/${LANGUAGE}" \
#     --out-dir "/gscratch/stf/hardiksr/hiatus/metrics"
