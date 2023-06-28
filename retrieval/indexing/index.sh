#!/bin/bash

# Set the arguments for the Python script:
PASSAGES_PATH="../../data/fandoms_qa/fandoms_all/val/passages.tsv"
INDEX_PATH="../../data/fandoms_qa/fandoms_all/val/passages.idx"
CHECKPOINT_PATH="../../data/colbertv2.0"
DTYPE="FP16"  # FP16, FP32, FP64
BATCH_SIZE="8"


# Execute the Python script with the provided arguments
python index.py \
    --passages-path "$PASSAGES_PATH" \
    --checkpoint-path  "$CHECKPOINT_PATH"\
    --index-path "$INDEX_PATH" \
    --dtype "$DTYPE" \
    --batch-size "$BATCH_SIZE" \
    --use-gpu \
    # --embedding-only
