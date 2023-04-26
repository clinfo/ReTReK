#!/bin/bash

python_prefix="$HOME/.pyenv/versions/miniconda3-4.7.12/envs/realretro/bin/python"

$python_prefix dataset_3_negative_examples.py \
    --input_ai_file_path ${INPUT_AI_FILE_PATH} \
    --input_rt_file_path ${INPUT_RT_FILE_PATH} \
    --output_folder_path ${OUTPUT_FOLDER_PATH} \
    --yield_confidence ${YIELD_CONFIDENCE} \
    --yield_cutoff ${YIELD_CUTOFF} \
    --min_template_frequency ${MIN_TEMPLATE_FREQUENCY} \
    --rt_virtual_negatives ${RT_VIRTUAL_NEGATIVES} \
    --rp_virtual_negatives ${RP_VIRTUAL_NEGATIVES} \
    --random_seed ${RANDOM_SEED} \
    --num_cores ${NUM_CORES}
