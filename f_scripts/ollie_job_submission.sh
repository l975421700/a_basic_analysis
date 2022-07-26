#!/bin/bash
#SBATCH -p mpp
#SBATCH --time=00:30:00
#SBATCH -o job.out
#SBATCH -e job.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=2008-2026
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err


echo "Current time : " $(date +"%T")

/work/ollie/qigao001/a_basic_analysis/f_scripts/0_awiesm_post_processing/get_monthly_data_from_submonthly_data.sh ${SLURM_ARRAY_TASK_ID}

echo "Current time : " $(date +"%T")

#Xsrun  I know what I am doing