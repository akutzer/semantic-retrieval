#!/bin/bash

# # Set the arguments for the Python script:
# # dataset arguments
# DATASET_NAME="harry_potter"
# DATASET_MODE="QQP"
# PASSAGES_PATH_TRAIN="../../data/fandoms_qa/harry_potter/train/passages.tsv"
# QUERIES_PATH_TRAIN="../../data/fandoms_qa/harry_potter/train/queries.tsv"
# TRIPLES_PATH_TRAIN="../../data/fandoms_qa/harry_potter/train/triples.tsv"
# PASSAGES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/passages.tsv"
# QUERIES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/queries.tsv"
# TRIPLES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/triples.tsv"

# # dataloader arguments
# DOC_MAXLEN="220"
# QUERY_MAXLEN="32"
# TRAIN_WORKERS="4"
# VAL_WORKERS="1"

# # model arguments
# BACKBONE="bert-base-uncased"
# DIM="128"
# DROPOUT="0.1"
# SIMILARITY="cosine"
# NORMALIZE="True"

# # training arguments
# EPOCHS="10"
# BATCH_SIZE="24"
# ACCUM_STEPS="1"
# SEED="125"
# NUM_EVAL_PER_EPOCH="6"
# CHECKPOINTS_PER_EPOCH="2"
# USE_AMP="True"
# NUM_GPUS="1"


# Set the arguments for the Python script:
# dataset arguments
DATASET_NAME="harry_potter"
DATASET_MODE="QQP"
PASSAGES_PATH_TRAIN="../../data/fandoms_qa/harry_potter/train/passages.tsv"
QUERIES_PATH_TRAIN="../../data/fandoms_qa/harry_potter/train/queries.tsv"
TRIPLES_PATH_TRAIN="../../data/fandoms_qa/harry_potter/train/triples.tsv"
PASSAGES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/passages.tsv"
QUERIES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/queries.tsv"
TRIPLES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/triples.tsv"

# dataloader arguments
DOC_MAXLEN="220"
QUERY_MAXLEN="32"
TRAIN_WORKERS="0"
VAL_WORKERS="0"

# model arguments
BACKBONE="bert-base-uncased"
DIM="128"
DROPOUT="0.1"
SIMILARITY="cosine"
NORMALIZE="True"

# training arguments
EPOCHS="10"
BATCH_SIZE="24"
ACCUM_STEPS="1"
SEED="125"
NUM_EVAL_PER_EPOCH="6"
CHECKPOINTS_PER_EPOCH="2"
USE_AMP="True"
NUM_GPUS="1"
CHECKPOINTS_PATH="../../checkpoints"
TENSORBOARD_PATH="../../runs"

# Execute the Python script with the provided arguments
python3 train_fandom_copy.py \
  --dataset-name "$DATASET_NAME" \
  --dataset-mode "$DATASET_MODE" \
  --passages-path-train "$PASSAGES_PATH_TRAIN" \
  --queries-path-train "$QUERIES_PATH_TRAIN" \
  --triples-path-train "$TRIPLES_PATH_TRAIN" \
  --passages-path-val "$PASSAGES_PATH_VAL" \
  --queries-path-val "$QUERIES_PATH_VAL" \
  --triples-path-val "$TRIPLES_PATH_VAL" \
  --doc-maxlen "$DOC_MAXLEN" \
  --query-maxlen "$QUERY_MAXLEN" \
  --train-workers "$TRAIN_WORKERS" \
  --val-workers "$VAL_WORKERS" \
  --backbone "$BACKBONE" \
  --dim "$DIM" \
  --dropout "$DROPOUT" \
  --similarity "$SIMILARITY" \
  --normalize \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --accum-steps "$ACCUM_STEPS" \
  --seed "$SEED" \
  --num-eval-per-epoch "$NUM_EVAL_PER_EPOCH" \
  --checkpoints-per-epoch "$CHECKPOINTS_PER_EPOCH" \
  --use-amp \
  --num-gpus "$NUM_GPUS" \
  --checkpoints-path "$CHECKPOINTS_PATH" \
  --tensorboard-path "$TENSORBOARD_PATH"

