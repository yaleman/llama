#!/bin/bash

set -e

if [ ! -d ".venv" ]; then
    echo "############################################"
    echo "Creating virtualenv and installing packages"
    echo "############################################"
    ./install_deps.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate


# --nproc_per_node NPROC_PER_NODE
#   Number of workers per node; supported values: [auto, cpu, gpu, int].
NPROC_PER_NODE="1"
# NPROC_PER_NODE="cpu"

if [ -z "${MODEL_DIR}" ]; then
    MODEL_DIR="./llama-2-7b-chat"
fi

if [ ! -d "${MODEL_DIR}" ]; then
    MODEL_DIR="$(find .  -maxdepth 1 -type d -name 'llama-*' | head -n1| sed -E 's/^\.\///g')"
    if [ -z "${MODEL_DIR}" ]; then
        echo "Can't find model directory, quitting"
        exit 1
    fi
fi

if [ -z "${MAX_BATCH_SIZE}" ]; then
    MAX_BATCH_SIZE=6
fi

if [ -z "${MAX_SEQ_LEN}" ]; then
    MAX_SEQ_LEN=256
fi

echo "Using model ${MODEL_DIR} (set with MODEL_DIR env var)"
echo "Using max_batch_size=${MAX_BATCH_SIZE} (set with MAX_BATCH_SIZE env var)"
echo "Using max_seq_len=${MAX_SEQ_LEN} (set with MAX_SEQ_LEN env var)"

if [ -n "$1" ]; then
    DIALOG_FILE=" --dialog-filename $1"
else
    DIALOG_FILE=""
fi

# shellcheck disable=SC2086
torchrun --nproc_per_node "${NPROC_PER_NODE}" \
    example_chat_completion.py $DIALOG_FILE \
    --ckpt_dir "${MODEL_DIR}" \
    --tokenizer_path tokenizer.model \
    --max_seq_len "${MAX_SEQ_LEN}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --standalone


# if you get this: "AssertionError: Loading a checkpoint for MP=1 but world size is 12"
# set your NPROC_PER_NODE to "1" instead of ... something else