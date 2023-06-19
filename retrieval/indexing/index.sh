#!/bin/bash

# Set the arguments for the Python script:
PASSAGES_PATH="../../data/fandoms_qa/harry_potter/all/passages.tsv"
CHECKPOINT_PATH="../../data/colbertv2.0"
INDEX_PATH="../../data/fandoms_qa/harry_potter/all/passages.index.pt"
DTYPE="FP16"  # FP16, FP32, FP64
BATCH_SIZE="8"


# Execute the Python script with the provided arguments
python index.py \
    --passages-path "$PASSAGES_PATH" \
    --checkpoint-path  "$CHECKPOINT_PATH"\
    --index-path "$INDEX_PATH" \
    --dtype "$DTYPE" \
    --batch-size "$BATCH_SIZE" \
    --use-gpu
