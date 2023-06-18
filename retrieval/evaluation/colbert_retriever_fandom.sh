#!/bin/bash

#SBATCH --time=00:15:00         # walltime
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1                  # limit to one node
#SBATCH --cpus-per-task=4
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1            # number of GPUs
#SBATCH --mem=32G
#SBATCH -A p_sp_bigdata         # name of the associated project
#SBATCH -J "fandom_harry_potter"  # name of the job
#SBATCH --output="retriever_fandom_job-%j.out"    # output file name (std out)
#SBATCH --error="retriever_fandom_job-%j.err"     # error file name (std err)
#SBATCH --mail-user="zhiwei.zhan@mailbox.tu-dresden.de" # will be used to used to update you about the state of your$
#SBATCH --mail-type ALL

# clean current modules
module purge

# HPC-Cluster doesn't have newer Python Version
module load Python/3.10.4

# switch to virtualenv to setup our environment, modules we need
# if needed change to your own workspace
virtualenv --system-site-packages /scratch/ws/0/zhzh622c-test-workspace/env
source /scratch/ws/0/zhzh622c-test-workspace/env/bin/activate

# Set the arguments for the Python script:
# dataset arguments
DATASET_NAME="harry_potter"
DATASET_MODE="QQP"
PASSAGES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/passages.tsv"
QUERIES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/queries.tsv"
TRIPLES_PATH_VAL="../../data/fandoms_qa/harry_potter/val/triples.tsv"

# model arguments
INDEXER="../../data/fandoms_qa/harry_potter/val/passages.colbert.indices.pt"
CHECKPOINT="../../data/colbertv2.0"
# CHECKPOINT="../../data/checkpoint/harry_potter/epoch8_1_loss0.1437_mrr0.9791_acc95.819"

# Execute the Python script with the provided arguments
python colbert_retriever.py \
    --dataset_name "$DATASET_NAME" \
    --dataset_mode "$DATASET_MODE" \
    --indexer  "$INDEXER"\
    --checkpoint "$CHECKPOINT" \
    --passages_path_val "$PASSAGES_PATH_VAL" \
    --queries_path_val "$QUERIES_PATH_VAL" \
    --triples_path_val "$TRIPLES_PATH_VAL" \

deactivate
