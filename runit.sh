#!/bin/bash

# --nproc_per_node NPROC_PER_NODE
#   Number of workers per node; supported values: [auto, cpu, gpu, int].
NPROC_PER_NODE="1"
# NPROC_PER_NODE="cpu"

MODEL_DIR="./llama-2-7b-chat"

if [ ! -d "${MODEL_DIR}" ]; then
    echo "Can't find model directory ${MODEL_DIR}, quitting"
    exit 1
fi

torchrun --nproc_per_node "${NPROC_PER_NODE}" \
    example_chat_completion.py \
    --ckpt_dir ./llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 \
    --max_batch_size 6


# if you get this: "AssertionError: Loading a checkpoint for MP=1 but world size is 12"
# set your NPROC_PER_NODE to "1" instead of ... something else