#!/bin/bash
#SBATCH -p mpp
#SBATCH --qos=12h
#SBATCH --time=1:00:00
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --array=0-5
#SBATCH --ntasks=6

echo "Current time : " $(date +"%T")
cd $WORK
source ${HOME}/miniconda3/bin/activate deepice

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.0_get_monthly_uv_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_600_5.0

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.1_get_psl_zh.sh ${SLURM_ARRAY_TASK_ID} /albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_603_5.3

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.2_get_monthly_wisoq_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_603_5.3

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.3_get_monthly_tpot_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_600_5.0

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.4_get_monthly_st_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_603_5.3

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.5_get_monthly_q_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_600_5.0

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.6_get_monthly_geop_t_ml.sh ${SLURM_ARRAY_TASK_ID} /albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_603_5.3

python /albedo/work/user/qigao001/a_basic_analysis/f_scripts/1_py_scripts/srun${SLURM_ARRAY_TASK_ID}.py

echo "Current time : " $(date +"%T")

#Xsrun  I know what I am doing

# #SBATCH --cpus-per-task=36
# #SBATCH --ntasks=1
# #SBATCH --array=2000-2029
# #SBATCH --account=paleodyn.paleodyn
