#!/bin/bash

#SBATCH --job-name=obfuscation_genre_prep_all_langs
#SBATCH --mail-user=kogolobo@uw.edu
#SBATCH --mail-type=FAIL,END

#SBATCH --account=xlab
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=80GB
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=6
#SBATCH --time=50:00:00 

#SBATCH --chdir=/mmfs1/home/kogolobo/repos/hiatus
#SBATCH --export=all
#SBATCH --output=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_genre_prep_all_langs.out
#SBATCH --error=/gscratch/stf/kogolobo/hiatus/logs/obfuscation_genre_prep_all_langs.err
set -e

source /mmfs1/home/kogolobo/.bashrc
conda activate dl
module load cuda/12.8.1 gcc/12.3.0

ar_genres=("ar_twitter_1" "saudinewsnet" "AMINA" "bbn_aljazeera_blog_1")
en_genres=("perGenre-HRS2.1" "perGenre-HRS2.2" "perGenre-HRS2.3" "perGenre-HRS2.4" "perGenre-HRS2.5" "en_crossGenre" "perGenre-HRS3.1" "perGenre-HRS3.2" "perGenre-HRS3.3" "perGenre-HRS3.4" "perGenre-HRS3.5" "en_short_crossGenre")
ru_genres=("perGenre-HRS2.101" "perGenre-HRS2.102" "perGenre-HRS2.103" "perGenre-HRS2.104" "perGenre-HRS2.105" "ru_crossGenre")
zh_genres=("perGenre-HRS3.301" "perGenre-HRS3.302" "perGenre-HRS3.303" "perGenre-HRS3.304" "perGenre-HRS3.305" "zh_short_crossGenre" "zh_medium_crossGenre")

process_lang() {
    local lang=$1
    shift
    local genres=("$@")

    for genre in "${genres[@]}"; do
        read -r input_queries_path privatized_queries_path < <(python scripts/4-obfuscation/get_hrs_paths.py --genre ${genre} --language ${lang} --pipeline-suffix test)
        
        mkdir -p "/gscratch/ifml1/kogolobo/genre_targets/${lang}"
        output_path="/gscratch/ifml1/kogolobo/genre_targets/${lang}/${genre}_genre_targets.json"
        
        echo "--------------------------------------------------------"
        echo "Creating genre target data for Language: ${lang} | Genre: ${genre}"
        echo "Input path: ${input_queries_path}"
        echo "Output path: ${output_path}"

        python scripts/4-obfuscation/make_genre_examples.py \
            --input-path "${input_queries_path}" \
            --output-path "${output_path}" \
            --input-key "fullText" \
            --output-key "PIIfullText" \
            --seed 42
    done
}

# Execute for all languages
process_lang "ar" "${ar_genres[@]}"
process_lang "en" "${en_genres[@]}"
process_lang "ru" "${ru_genres[@]}"
process_lang "zh" "${zh_genres[@]}"

echo "All genre extractions completed successfully."