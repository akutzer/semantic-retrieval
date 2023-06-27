#!/bin/bash

# Set the arguments for the Python script:

# Example for FANDOM QA:
# DATASET_MODE="QQP" # QQP or QPP
# PASSAGES_PATH="../../data/fandoms_qa/harry_potter/val/passages.tsv"
# QUERIES_PATH="../../data/fandoms_qa/harry_potter/val/queries.tsv"
# TRIPLES_PATH="../../data/fandoms_qa/harry_potter/val/triples.tsv"

# INDEX_PATH="../../data/fandoms_qa/harry_potter/all/passages.index.pt"
CHECKPOINT_PATH="../../data/colbertv2.0"

# Example for MS MARCO:
DATASET_MODE="QPP" # QQP or QPP
PASSAGES_PATH="../../data/ms_marco/ms_marco_v1_1/val/passages.tsv"
QUERIES_PATH="../../data/ms_marco/ms_marco_v1_1/val/queries.tsv"
TRIPLES_PATH="../../data/ms_marco/ms_marco_v1_1/val/triples.tsv"

# INDEX_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_8/passages.ms_marco_v2_8.indices.pt"
# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_8/epoch4_2_loss1.3650_mrr0.6765_acc50.232"

# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_16/epoch4_2_loss1.3054_mrr0.6912_acc51.958"
# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_24/epoch4_2_loss1.2986_mrr0.6948_acc52.451"
# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_32/epoch3_2_loss1.2895_mrr0.6943_acc52.262"
# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_64/epoch3_2_loss1.2838_mrr0.6973_acc52.680"
# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_128/epoch3_2_loss1.2828_mrr0.6986_acc52.852"
# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_L2/epoch3_2_loss1.2783_mrr0.7018_acc52.948"
# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_L2_normalized/epoch3_2_loss1.2830_mrr0.6993_acc52.660"
# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_mse/epoch4_2_loss0.7460_mrr0.6574_acc47.342"
# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_query_64/epoch3_2_loss1.2786_mrr0.7000_acc53.038"
# CHECKPOINT_PATH="../../data/checkpoint/ms_marco/ms_marco_v2_roberta/epoch4_2_loss1.2602_mrr0.7049_acc53.571"

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
