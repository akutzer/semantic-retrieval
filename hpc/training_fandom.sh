#!/bin/bash
#SBATCH --time=00:10:00         # walltime
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1	            # limit to one node
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1            # number of GPUs
#SBATCH --mem=40G
#SBATCH -A p_sp_bigdata         # name of the associated project
#SBATCH -J "training_fandom_job"  # name of the job
#SBATCH --output="training_fandom_job-%j.out"    # output file name (std out)
#SBATCH --error="training_fandom_job-%j.err"     # error file name (std err)
#SBATCH --mail-user="tommy.nguyen@mailbox.tu-dresden.de" # will be used to used to update you about the state of your$
#SBATCH --mail-type ALL

# clean current modules
module purge

# HPC-Cluster doesn't have newer Python Version 
module load Python/3.10.4 

# switch to virtualenv with already prepared environment 
source /scratch/ws/0/tong623c-tommy-workspace/env/bin/activate 

python ../training/train_fandom.py

deactivate 