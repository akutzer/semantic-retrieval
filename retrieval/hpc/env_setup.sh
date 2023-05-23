#!/bin/bash
#SBATCH --time=0:10:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1	# limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --partition=haswell
#SBATCH --mem-per-cpu=2000M   # memory per CPU core
#SBATCH -A p_sp_bigdata # name of the associated project
#SBATCH -J "first_script" # name of the job
#SBATCH -o "first_script-%j.out" # output file name (std out)
#SBATCH --error="first_script-%j.err" # error file name (std err)
#SBATCH --mail-user="tommy.nguyen@mailbox.tu-dresden.de" # will be used to used to update you about the state of your$
#SBATCH --mail-type ALL

# HPC-Cluster doesn't have new Python Version 
module load Python/3.10.4

# switch to virtualenv to setup our environment, modules we need 
# if needed change to your own workspace
virtualenv --system-site-packages /scratch/ws/0/tong623c-tommy-workspace/env
source /scratch/ws/0/tong623c-tommy-workpsace/env/bin/activate

# maybe switch Python Version with pyenv

# load modules, no modules and their version are in the HPC modules
# have to install everything via pip
pip install -r ../requirements.txt

deactivate

