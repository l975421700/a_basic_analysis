#!/bin/bash
#SBATCH -p mpp
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --array=2010-2089
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err


echo "Current time : " $(date +"%T")

# /work/ollie/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.0_get_monthly_uvq_plev.sh ${SLURM_ARRAY_TASK_ID} output/echam-6.3.05p2-wiso/pi pi_m_416_4.9

# /work/ollie/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.1_get_psl_zh.sh ${SLURM_ARRAY_TASK_ID} output/echam-6.3.05p2-wiso/pi pi_m_416_4.9

/work/ollie/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.2_get_monthly_wisoq_plev.sh ${SLURM_ARRAY_TASK_ID} output/echam-6.3.05p2-wiso/pi pi_m_416_4.9

echo "Current time : " $(date +"%T")

#Xsrun  I know what I am doing