#!/bin/bash

export OMP_NUM_THREADS=8
python_prefix="$HOME/.pyenv/versions/miniconda3-4.7.12/envs/realretro/bin/python"

$python_prefix run.py \
    --target ${target} \
    --config ${config} \
    --knowledge cdscore \
    -c 500 \
    --sel_const 10 \
    --expansion_num 500 \
    --knowledge_weights 5.0 0.5 2.0 2. \
    --time_limit 14400 \
    --random_seed 1 \
    --save_result_dir ${SAVE_DIR}
