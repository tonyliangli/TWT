#!/bin/bash

N_GPUS=1
USE_REMOTE_STORAGE=true

BASE_DATASET="totto"
MASK_PATTERN="causal"
BASE_MODEL="t5"
MODEL_SIZE="base"
CUSTOM_MODEL=""
CUSTOM_SETTINGS=""
BEST_CHECKPOINT="checkpoint-20000"

TOKENIZER_NAME="t5-${MODEL_SIZE}"

# EXPERIMENT_NAME="${BASE_DATASET}_${MASK_PATTERN}_${BASE_MODEL}_${MODEL_SIZE}${CUSTOM_MODEL}${CUSTOM_SETTINGS}"
EXPERIMENT_NAME="${BASE_MODEL}_${MODEL_SIZE}${CUSTOM_MODEL}${CUSTOM_SETTINGS}"
TEST_SET_NAME="${BASE_DATASET}_test.jsonl"
TEST_MODEL_INPUTS_NAME="${BASE_DATASET}_prefix_${BASE_MODEL}_${MODEL_SIZE}_test_model_inputs.pkl.gz"
TEST_LINEARIZED_MODEL_INPUTS_NAME="${BASE_DATASET}_prefix_${BASE_MODEL}_${MODEL_SIZE}_test_model_inputs.pkl.gz"

# Directories of local path
LOCAL_TEMP_DIR=./temp
LOCAL_INPUT_DIR=${LOCAL_TEMP_DIR}/input
LOCAL_DEVIDED_INPUT_DIR=${LOCAL_TEMP_DIR}/devided_input
LOCAL_DEVIDED_OUTPUT_DIR=${LOCAL_TEMP_DIR}/devided_output

# Directories of remote path
REMOTE_DIR=/bdmstorage/teamdrive/tasks/final
REMOTE_DATA_DIR=${REMOTE_DIR}/data
REMOTE_EXPS_DIR=${REMOTE_DIR}/exps

if [ "${USE_REMOTE_STORAGE}" = true ]; then
  OUTPUT_DIR="${REMOTE_EXPS_DIR}/${EXPERIMENT_NAME}"
else
  OUTPUT_DIR="${LOCAL_TEMP_DIR}/output/${EXPERIMENT_NAME}"
fi
CHECKPOINTS_DIR="${OUTPUT_DIR}/checkpoints/${BEST_CHECKPOINT}"
PREDICTIONS_DIR="${OUTPUT_DIR}/test_predictions"

# Export envs
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_PCI_RELAXED_ORDERING=1 
export NCCL_NET_GDR_LEVEL=5

# Create necessary dirs
mkdir -p ${LOCAL_TEMP_DIR}
mkdir -p ${LOCAL_INPUT_DIR}
mkdir -p ${LOCAL_DEVIDED_INPUT_DIR}
mkdir -p ${LOCAL_DEVIDED_OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CHECKPOINTS_DIR}
mkdir -p ${PREDICTIONS_DIR}

if [ -z "$(ls -A $LOCAL_INPUT_DIR)" ]; then
  echo "Start copying input files from remote storage"
  cp -rf ${REMOTE_DATA_DIR}/* ${LOCAL_INPUT_DIR}
  echo "Unzip table files from ${LOCAL_INPUT_DIR}/tables.zip"
  unzip -qo ${LOCAL_INPUT_DIR}/tables.zip -d ${LOCAL_INPUT_DIR}
fi

# Split model inputs
echo "Splitting model inputs"
python twt_split_inputs.py \
  --model_inputs_path="${LOCAL_INPUT_DIR}/${TEST_MODEL_INPUTS_NAME}" \
  --devided_model_inputs_dir="${LOCAL_DEVIDED_INPUT_DIR}" \
  --split_num=${N_GPUS}

# Run evaluation daemon
echo "Evaluation during training daemon started"
python -u run_distributed_eval.py \
  --dataset_path="${LOCAL_INPUT_DIR}/${TEST_SET_NAME}" \
  --tables_dir="${LOCAL_INPUT_DIR}/tables" \
  --model_inputs_path="${LOCAL_INPUT_DIR}/${TEST_MODEL_INPUTS_NAME}" \
  --linearized_model_inputs_path="${LOCAL_INPUT_DIR}/${TEST_LINEARIZED_MODEL_INPUTS_NAME}" \
  --devided_model_inputs_dir="${LOCAL_DEVIDED_INPUT_DIR}" \
  --devided_predictions_dir="${LOCAL_DEVIDED_OUTPUT_DIR}" \
  --reduced_prediction_path="${PREDICTIONS_DIR}/${EXPERIMENT_NAME}_test_predictions.jsonl" \
  --split_num=${N_GPUS} \
  --model_name_or_path=${CHECKPOINTS_DIR} \
  --model_type="${BASE_MODEL}${CUSTOM_MODEL}" \
  --tokenizer_name_or_path=${TOKENIZER_NAME} \
  --num_return_sequences=1 \
  --greedy_search \
  > ${OUTPUT_DIR}/eval_daemon.log 2>&1 &