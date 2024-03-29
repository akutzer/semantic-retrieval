#!/bin/bash

#SBATCH --time=30:00:00         # walltime
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1	            # limit to one node
#SBATCH --cpus-per-task=4
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1            # number of GPUs
#SBATCH --mem=200G
#SBATCH -A p_sp_bigdata         # name of the associated project
#SBATCH -J "m_16"  # name of the job
#SBATCH --output="m_16_%j.out"    # output file name (std out)
#SBATCH --error="m_16_%j.err"     # error file name (std err)
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

# Example for MS MARCO:
DATASET_MODE="QPP" # QQP or QPP
PASSAGES_PATH="../../data/ms_marco/ms_marco_v2_1/val/passages.tsv"
QUERIES_PATH="../../data/ms_marco/ms_marco_v2_1/val/queries.tsv"
TRIPLES_PATH="../../data/ms_marco/ms_marco_v2_1/val/triples.tsv"
# INDEX_PATH="../../data/passages.colbert.indices.pt"

CHECKPOINT_PATH="../../data/checkpoint/ms_marco_v2_16/epoch4_2_loss1.3054_mrr0.6912_acc51.958"

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
    # --embedding-only \  # enabling this option needs ~6x more memory compared to 128 dim embeddings
    # --index-path "$INDEX_PATH" \

deactivate
