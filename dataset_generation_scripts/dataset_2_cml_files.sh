#!/bin/bash

python_prefix="$HOME/.pyenv/versions/miniconda3-4.7.12/envs/realretro/bin/python"

$python_prefix dataset_2_cml_files.py \
    --input_folder_path ${INPUT_FOLDER_PATH} \
    --output_folder_path ${OUTPUT_FOLDER_PATH} \
    --random_seed ${RANDOM_SEED} \
    --num_cores ${NUM_CORES}
