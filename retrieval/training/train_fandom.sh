#!/bin/bash

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
TRAIN_WORKERS="4"
VAL_WORKERS="1"

# model arguments
BACKBONE="bert-base-uncased" # "bert-base-uncased" or "../../data/colbertv2.0/" or "roberta-base"
DIM="128"
DROPOUT="0.1"
SIMILARITY="cosine" # "cosine" or "L2"
FREEZE_UNTIL_LAYER="0"

# training arguments
EPOCHS="10"
BATCH_SIZE="22"
ACCUM_STEPS="1"
LEARNING_RATE="5e-6"
WARMUP_EPOCHS="1"
WARMUP_START_FACTOR="0.1"
SEED="125"
NUM_EVAL_PER_EPOCH="6"
CHECKPOINTS_PER_EPOCH="2"
NUM_GPUS="1"
CHECKPOINTS_PATH="../../checkpoints"
TENSORBOARD_PATH="../../runs"

# if you want to resuming training from a checkpoint comment out the CHECKPOINT variable 
# and add the path to the checkpoint
# this is also the recommended way of loading the colbertv2 weights
# CHECKPOINT="../../checkpoints/harry_potter_bert_2023-05-31T15:10:52/epoch1_2_loss0.1793_mrr0.9658_acc93.171/"
# CHECKPOINT="../../data/colbertv2.0/"
# CHECKPOINT="../../checkpoints/harry_potter_bert_2023-06-03T08:13:49/epoch3_1_loss0.1103_mrr0.9803_acc96.061"



# Execute the Python script with the provided arguments
python3 train.py \
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
  --shuffle \
  --drop-last \
  --pin-memory \
  --backbone "$BACKBONE" \
  --dim "$DIM" \
  --dropout "$DROPOUT" \
  --similarity "$SIMILARITY" \
  --normalize \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --accum-steps "$ACCUM_STEPS" \
  --learning-rate "$LEARNING_RATE"\
  --seed "$SEED" \
  --num-eval-per-epoch "$NUM_EVAL_PER_EPOCH" \
  --checkpoints-per-epoch "$CHECKPOINTS_PER_EPOCH" \
  --use-amp \
  --num-gpus "$NUM_GPUS" \
  --checkpoints-path "$CHECKPOINTS_PATH" \
  --tensorboard-path "$TENSORBOARD_PATH" \
  --warmup-epochs "$WARMUP_EPOCHS"\
  --warmup-start-factor "$WARMUP_START_FACTOR"\
  --checkpoint "$CHECKPOINT"\
  --freeze-until-layer "$FREEZE_UNTIL_LAYER"\
