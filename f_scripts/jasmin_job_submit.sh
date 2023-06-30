#!/bin/bash
#SBATCH -p par-single
#SBATCH -n 16
#SBATCH --mem-per-cpu=64GB
#SBATCH -t 12:00:00
#SBATCH -o %j.out
#SBATCH -e %j.err

echo Start time: $(date)

cd ${HOME}
source ${HOME}/miniconda3/bin/activate deepice
which python

python ${HOME}/a_basic_analysis/c_codes/deprecated/0.3_srun/0.3.0_srun0.py

echo End time: $(date)

# partition: https://help.jasmin.ac.uk/article/4881-lotus-queues



# SBATCH --array=1982-2016
# SBATCH -o %A_%a.out
# SBATCH -e %A_%a.err

# ${SLURM_ARRAY_TASK_ID}

