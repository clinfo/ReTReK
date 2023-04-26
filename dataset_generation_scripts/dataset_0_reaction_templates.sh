#!/bin/bash

python_prefix="$HOME/.pyenv/versions/miniconda3-4.7.12/envs/realretro/bin/python"

$python_prefix dataset_0_reaction_templates.py \
    --input_file_path ${INPUT_FILE_PATH} \
    --output_folder_path ${OUTPUT_FOLDER_PATH} \
    --random_seed ${RANDOM_SEED} \
    --num_cores ${NUM_CORES}
