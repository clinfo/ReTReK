#!/bin/bash

python_prefix="$HOME/.pyenv/versions/miniconda3-4.7.12/envs/realretro/bin/python"

$python_prefix dataset_1_global_evaluation.py \
    --input_file_path ${INPUT_FILE_PATH} \
    --output_folder_path ${OUTPUT_FOLDER_PATH} \
    --min_atoms ${MIN_ATOMS} \
    --max_atoms ${MAX_ATOMS} \
    --fp_similarity_cutoff ${FP_SIMILARITY_CUTOFF} \
    --small_dataset_size ${SMALL_DATASET_SIZE} \
    --large_dataset_size ${LARGE_DATASET_SIZE} \
    --outlier_percentage ${OUTLIER_PERCENTAGE} \
    --random_seed ${RANDOM_SEED} \
    --num_cores ${NUM_CORES}
