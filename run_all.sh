#!/bin/bash

DATA=$1
CONFIG=$2
SAVE_DIR=$3
LOG_ERROR=$SAVE_DIR/error.log
LOG_OUT=$SAVE_DIR/out.log
RETREK_DIR=$4
mkdir -p $SAVE_DIR
for file in $DATA/*
do
    if [[ $file == *.mol ]]
    then
            qsub -v target=$file,SAVE_DIR=$SAVE_DIR,config=$CONFIG -e ${LOG_ERROR} -o ${LOG_OUT} -wd ${RETREK_DIR} run_one.sh
    fi
done
