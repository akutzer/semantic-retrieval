#!/bin/bash

# Set the arguments for the Python script:
# dataset arguments
DATASET_NAME="harry_potter"
PASSAGES_PATH_VAL="../../data/fandoms_qa/harry_potter/all/passages.tsv"

# model arguments
INDEXER="../../data/fandoms_qa/harry_potter/all/passages.colbert.indices.pt"
CHECKPOINT="../../data/colbertv2.0"
# CHECKPOINT="../../data/checkpoint/ms_marco/epoch4_2_loss1.8155_mrr0.5834_acc41.501"

# Execute the Python script with the provided arguments
python colbert_indexer.py \
    --dataset_name "$DATASET_NAME" \
    --indexer  "$INDEXER"\
    --checkpoint "$CHECKPOINT" \
    --passages_path_val "$PASSAGES_PATH_VAL" \
