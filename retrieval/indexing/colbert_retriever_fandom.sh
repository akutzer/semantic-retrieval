#!/bin/bash

# Set the arguments for the Python script:
# dataset arguments
DATASET_NAME="harry_potter"
DATASET_MODE="QQP"
PASSAGES_PATH_VAL="../../data/fandoms_qa/harry_potter/all/passages.tsv"
QUERIES_PATH_VAL="../../data/fandoms_qa/harry_potter/all/queries.tsv"
TRIPLES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/triples.tsv"

# model arguments
INDEXER="../../data/fandoms_qa/harry_potter/all/passages.colbert.indices.pt"
CHECKPOINT="../../data/colbertv2.0"
# CHECKPOINT="../../data/checkpoint/ms_marco/epoch4_2_loss1.8155_mrr0.5834_acc41.501"

# Execute the Python script with the provided arguments
python colbert_retriever.py \
    --dataset_name "$DATASET_NAME" \
    --dataset_mode "$DATASET_MODE" \
    --indexer  "$INDEXER"\
    --checkpoint "$CHECKPOINT" \
    --passages_path_val "$PASSAGES_PATH_VAL" \
    --queries_path_val "$QUERIES_PATH_VAL" \
    --triples_path_val "$TRIPLES_PATH_VAL" \
