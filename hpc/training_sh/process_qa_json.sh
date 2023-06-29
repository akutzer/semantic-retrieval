#!/bin/bash
#SBATCH --time=0:10:00          # walltime
#SBATCH --nodes=1               # number of nodes
#SBATCH --ntasks=1	            # limit to one node
#SBATCH --cpus-per-task=4       # number of processor cores (i.e. threads)
#SBATCH --partition=haswell
#SBATCH --mem=20000M     
#SBATCH -A p_sp_bigdata         # name of the associated project
#SBATCH -J "process_qa_json_job"      # name of the job
#SBATCH --output="process_qa_json_job-%j.out"    # output file name (std out)
#SBATCH --error="process_qa_json_job-%j.err"     # error file name (std err)
#SBATCH --mail-user="tommy.nguyen@mailbox.tu-dresden.de" # will be used to used to update you about the state of your$
#SBATCH --mail-type ALL

# clean current modules
module purge

# HPC-Cluster doesn't have newer Python Version 
module load Python/3.10.4

# switch to virtualenv with already prepared environment 
source /scratch/ws/0/tong623c-tommy-workspace/env/bin/activate 

python ../preprocessing/process_qa_json.py

deactivate 

