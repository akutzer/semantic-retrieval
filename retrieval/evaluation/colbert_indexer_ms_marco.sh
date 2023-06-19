#!/bin/bash

#SBATCH --time=00:15:00         # walltime
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1                  # limit to one node
#SBATCH --cpus-per-task=4
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1            # number of GPUs
#SBATCH --mem=96G
#SBATCH -A p_sp_bigdata         # name of the associated project
#SBATCH -J "indexer_ms_marco"  # name of the job
#SBATCH --output="indexer_ms_marco_v2-%j.out"    # output file name (std out)
#SBATCH --error="indexer_ms_marco_v2-%j.err"     # error file name (std err)
#SBATCH --mail-user="zhiwei.zhan@mailbox.tu-dresden.de" # will be used to used to update you about the state of your$
#SBATCH --mail-type ALL

# clean current modules
# module purge

# HPC-Cluster doesn't have newer Python Version
# module load Python/3.10.4

# switch to virtualenv to setup our environment, modules we need
# if needed change to your own workspace
# virtualenv --system-site-packages /scratch/ws/0/zhzh622c-test-workspace/env
# source /scratch/ws/0/zhzh622c-test-workspace/env/bin/activate

# dataset arguments
DATASET_NAME="ms_marco"
PASSAGES_PATH_VAL="../../data/ms_marco/ms_marco_v1_1/val/passages.tsv"

# model arguments
INDEXER="../../data/ms_marco/ms_marco_v1_1/val/passages.colbert.indices.pt"
CHECKPOINT="../../data/colbertv2.0"
# CHECKPOINT="../../data/checkpoint/ms_marco/epoch4_2_loss1.8155_mrr0.5834_acc41.501"

# Execute the Python script with the provided arguments
python colbert_indexer.py \
    --dataset_name "$DATASET_NAME" \
    --indexer  "$INDEXER"\
    --checkpoint "$CHECKPOINT" \
    --passages_path_val "$PASSAGES_PATH_VAL" \

# deactivate
