#!/bin/bash
#SBATCH --time=0:10:00          # walltime
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1	            # limit to one node
#SBATCH --cpus-per-task=1       # number of processor cores (i.e. threads)
#SBATCH --partition=haswell
#SBATCH --mem=8000M     
#SBATCH -A p_sp_bigdata         # name of the associated project
#SBATCH -J "torch_version_job"      # name of the job
#SBATCH --output="torch_version_job-%j.out"    # output file name (std out)
#SBATCH --error="torch_version_job-%j.err"     # error file name (std err)
#SBATCH --mail-user="tommy.nguyen@mailbox.tu-dresden.de" # will be used to used to update you about the state of your$
#SBATCH --mail-type ALL

# clean current modules
module purge

# HPC-Cluster doesn't have newer Python Version 
module load Python/3.10.4

# switch to virtualenv to setup our environment
source /scratch/ws/0/tong623c-tommy-workspace/env/bin/activate

# maybe switch Python Version with pyenv

# see torch version
python -c "import torch; print(torch.__version__)"

deactivate
