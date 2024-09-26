#!/bin/bash
#SBATCH -p high-mem
#SBATCH --time=00:30:00
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --array=0-5
#SBATCH --mem=120GB
# #SBATCH -n 16

echo "Current time : " $(date +"%T")
cd ${HOME}
source ${HOME}/miniconda3/bin/activate deepice

python ${HOME}/a_basic_analysis/f_scripts/1_py_scripts/srun${SLURM_ARRAY_TASK_ID}.py

echo "Current time : " $(date +"%T")

# partition: https://help.jasmin.ac.uk/article/4881-lotus-queues
