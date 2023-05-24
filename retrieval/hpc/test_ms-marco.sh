#!/bin/bash
#SBATCH --time=01:00:00         # walltime
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1	            # limit to one node
#SBATCH --cpus-per-task=1       # number of processor cores (i.e. threads)
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1            # number of GPUs
#SBATCH --mem-per-gpu=10000M
#SBATCH -A p_sp_bigdata         # name of the associated project
#SBATCH -J "test_ms_marco_job"  # name of the job
#SBATCH --output="test_ms_marco_job-%j.out"    # output file name (std out)
#SBATCH --error="test_ms_marco_job-%j.err"     # error file name (std err)
#SBATCH --mail-user="tommy.nguyen@mailbox.tu-dresden.de" # will be used to used to update you about the state of your$
#SBATCH --mail-type ALL

# clean current modules
module purge

# HPC-Cluster doesn't have newer Python Version 
module load Python/3.10.4 

# switch to virtualenv with already prepared environment 
source /scratch/ws/0/tong623c-tommy-workspace/env/bin/activate 
pip install datasets

python ../preprocessing/generate_ms_marco.py

deactivate 