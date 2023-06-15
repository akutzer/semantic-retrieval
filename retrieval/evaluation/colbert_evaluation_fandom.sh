#!/bin/bash

# Set the arguments for the Python script:
# dataset arguments
DATASET_NAME="harry_potter"
DATASET_MODE="QQP"
PASSAGES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/passages.tsv"
QUERIES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/queries.tsv"
TRIPLES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/triples.tsv"

# model arguments
INDEXER="../../data/fandoms_qa/harry_potter/val/passages.indices.pt"
CHECKPOINT="../../data/colbertv2.0"


# Execute the Python script with the provided arguments
python colbert_evaluation.py \
    --dataset_name "$DATASET_NAME" \
    --dataset_mode "$DATASET_MODE" \
    --indexer  "$INDEXER"\
    --checkpoint "$CHECKPOINT" \
    --passages_path_val "$PASSAGES_PATH_VAL" \
    --queries_path_val "$QUERIES_PATH_VAL" \
    --triples_path_val "$TRIPLES_PATH_VAL" \
