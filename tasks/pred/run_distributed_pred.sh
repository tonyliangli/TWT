#!/bin/bash

N_GPUS=1

# Directories of local path
LOCAL_TEMP_DIR=./temp
LOCAL_INPUT_DIR=${LOCAL_TEMP_DIR}/input
LOCAL_DEVIDED_INPUT_DIR=${LOCAL_TEMP_DIR}/devided_input
LOCAL_DEVIDED_OUTPUT_DIR=${LOCAL_TEMP_DIR}/devided_output

# Directories of remote path
REMOTE_DIR=/bdmstorage/teamdrive/tasks/twt_release/base
REMOTE_DATA_DIR=${REMOTE_DIR}/data
REMOTE_EXPS_DIR=${REMOTE_DIR}/exps
REMOTE_SUMMARY_DIR=${REMOTE_DIR}/summary

# Create necessary dirs
mkdir -p ${LOCAL_TEMP_DIR}
mkdir -p ${LOCAL_INPUT_DIR}
mkdir -p ${LOCAL_DEVIDED_INPUT_DIR}
mkdir -p ${LOCAL_DEVIDED_OUTPUT_DIR}

# if [ -z "$(ls -A $LOCAL_INPUT_DIR)" ]; then
#   echo "Start copying input files from remote storage"
#   cp -rf ${REMOTE_DATA_DIR}/* ${LOCAL_INPUT_DIR}
#   echo "Unzip table files from ${LOCAL_INPUT_DIR}/tables.zip"
#   unzip -qo ${LOCAL_INPUT_DIR}/tables.zip -d ${LOCAL_INPUT_DIR}
# fi

# Run evaluation daemon
echo "Distributed prediction on test sets started"
python -u run_distributed_pred.py \
  --input_dir=${REMOTE_EXPS_DIR} \
  --temp_dir=${LOCAL_TEMP_DIR} \
  --summary_dir=${REMOTE_SUMMARY_DIR} \
  --dataset_split=test \
  --split_num=${N_GPUS} \
  --num_return_sequences=1