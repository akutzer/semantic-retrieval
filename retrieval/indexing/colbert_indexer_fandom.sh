#!/bin/bash

# Set the arguments for the Python script:
# dataset arguments
DATASET_NAME="harry_potter"
PASSAGES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/passages.tsv"

# model arguments
INDEXER="../../data/fandoms_qa/harry_potter/val/passages.check.indices.pt"
# CHECKPOINT="../../data/colbertv2.0"
CHECKPOINT="../../data/epoch5_2_loss0.1540_mrr0.9719_acc94.431"

# Execute the Python script with the provided arguments
python colbert_indexer.py \
    --dataset_name "$DATASET_NAME" \
    --indexer  "$INDEXER"\
    --checkpoint "$CHECKPOINT" \
    --passages_path_val "$PASSAGES_PATH_VAL" \
