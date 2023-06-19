#!/bin/bash

# Set the arguments for the Python script:

# Example for FANDOM QA:
# DATASET_MODE="QQP" # QQP or QPP
# PASSAGES_PATH="../../data/fandoms_qa/harry_potter/all/passages.tsv"
# QUERIES_PATH="../../data/fandoms_qa/harry_potter/all/queries.tsv"
# TRIPLES_PATH="../../data/fandoms_qa/harry_potter/val/triples.tsv"
# INDEX_PATH="../../data/fandoms_qa/harry_potter/all/passages.index.pt"

# Example for MS MARCO:
DATASET_MODE="QPP" # QQP or QPP
PASSAGES_PATH="../../data/ms_marco/ms_marco_v1_1/val/passages.tsv"
QUERIES_PATH="../../data/ms_marco/ms_marco_v1_1/val/queries.tsv"
TRIPLES_PATH="../../data/ms_marco/ms_marco_v1_1/val/triples.tsv"
# INDEX_PATH="../../data/ms_marco/ms_marco_v1_1/val/passages.index.pt"


CHECKPOINT_PATH="../../data/colbertv2.0"
BATCH_SIZE="8"
DTYPE="FP16"  # FP16, FP32, FP64
K="1000"


# Execute the Python script with the provided arguments
python evaluate_retrieval.py \
    --dataset-mode "$DATASET_MODE" \
    --passages-path "$PASSAGES_PATH" \
    --queries-path "$QUERIES_PATH" \
    --triples-path "$TRIPLES_PATH" \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --dtype "$DTYPE" \
    --batch-size "$BATCH_SIZE" \
    --k "$K" \
    --use-gpu \
    # --index-path "$INDEX_PATH" \
