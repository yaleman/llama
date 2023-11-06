#!/bin/bash

# --nproc_per_node NPROC_PER_NODE
#   Number of workers per node; supported values: [auto, cpu, gpu, int].
NPROC_PER_NODE="1"
# NPROC_PER_NODE="cpu"

MODEL_DIR="./llama-2-7b-chat"

if [ ! -d "${MODEL_DIR}" ]; then
    MODEL_DIR="$(find .  -maxdepth 1 -type d -name 'llama-*' | head -n1| sed -E 's/^\.\///g')"
    if [ -z "${MODEL_DIR}" ]; then
        echo "Can't find model directory, quitting"
        exit 1
    fi
fi

echo "Using model ${MODEL_DIR}"

torchrun --nproc_per_node "${NPROC_PER_NODE}" \
    example_chat_completion.py \
    --ckpt_dir "${MODEL_DIR}" \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6


# if you get this: "AssertionError: Loading a checkpoint for MP=1 but world size is 12"
# set your NPROC_PER_NODE to "1" instead of ... something else