#!/bin/bash
#SBATCH --time=27:59:00         # walltime
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1	            # limit to one node
#SBATCH --cpus-per-task=4
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1            # number of GPUs
#SBATCH --mem=32G
#SBATCH -A p_sp_bigdata         # name of the associated project
#SBATCH -J "training_ms_marco_v2_final_128_L2_job"  # name of the job
#SBATCH --output="logs/training_ms_marco_v2_final_128_L2_job-%j.out"    # output file name (std out)
#SBATCH --error="logs/training_ms_marco_v2_final_128_L2_job-%j.err"     # error file name (std err)
#SBATCH --mail-user="aaron.kutzer@mailbox.tu-dresden.de" # will be used to used to update you about the state of your$
#SBATCH --mail-type ALL

# clean current modules
module purge

# HPC-Cluster doesn't have newer Python Version 
module load Python/3.10.4 

# switch to virtualenv with already prepared environment 
source /scratch/ws/0/tong623c-tommy-workspace/env/bin/activate 

# Set the arguments for the Python script:
# dataset arguments
DATASET_NAME="final_ms_marco_v2"
DATASET_MODE="QPP"
PASSAGES_PATH_TRAIN="../data/ms_marco/ms_marco_v2_1/train/passages.tsv"
QUERIES_PATH_TRAIN="../data/ms_marco/ms_marco_v2_1/train/queries.tsv"
TRIPLES_PATH_TRAIN="../data/ms_marco/ms_marco_v2_1/train/triples.tsv"
PASSAGES_PATH_VAL="../data/ms_marco/ms_marco_v2_1/val/passages.tsv"
QUERIES_PATH_VAL="../data/ms_marco/ms_marco_v2_1/val/queries.tsv"
TRIPLES_PATH_VAL="../data/ms_marco/ms_marco_v2_1/val/triples.tsv"


# dataloader arguments
DOC_MAXLEN="320"
QUERY_MAXLEN="48"
PASSAGES_PER_QUERY="10"
TRAIN_WORKERS="4"
VAL_WORKERS="4"

# model arguments
BACKBONE="roberta-base" # "bert-base-uncased" or "../data/colbertv2.0/" or "roberta-base"
DIM="128"
DROPOUT="0.1"
SIMILARITY="L2" # "cosine" or "L2"

# training arguments
EPOCHS="9"
BATCH_SIZE="39"
ACCUM_STEPS="3"
LEARNING_RATE="5e-6"
WARMUP_EPOCHS="1"
WARMUP_START_FACTOR="0.05"
SEED="125"
NUM_EVAL_PER_EPOCH="6"
CHECKPOINTS_PER_EPOCH="6"
NUM_GPUS="1"
CHECKPOINTS_PATH="../checkpoints"
TENSORBOARD_PATH="../runs"

# if you want to resuming training from a checkpoint comment out the CHECKPOINT variable 
# and add the path to the checkpoint
# this is also the recommended way of loading the colbertv2 weights
# CHECKPOINT="../../checkpoints/harry_potter_bert_2023-05-31T15:10:52/epoch1_2_loss0.1793_mrr0.9658_acc93.171/"
# CHECKPOINT="../../data/colbertv2.0/"



# Execute the Python script with the provided arguments
python3 ../retrieval/training/train.py \
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
  --passages-per-query "$PASSAGES_PER_QUERY" \
  --train-workers "$TRAIN_WORKERS" \
  --val-workers "$VAL_WORKERS" \
  --shuffle \
  --drop-last \
  --pin-memory \
  --backbone "$BACKBONE" \
  --dim "$DIM" \
  --dropout "$DROPOUT" \
  --similarity "$SIMILARITY" \
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


deactivate 

