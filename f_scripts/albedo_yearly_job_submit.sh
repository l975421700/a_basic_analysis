#!/bin/bash
#SBATCH -p mpp
#SBATCH --qos=12h
#SBATCH --time=12:00:00
#SBATCH -o %A_%a.out
#SBATCH -e %A_%a.err
#SBATCH --array=1979-2022
# #SBATCH --mem=512GB

echo "Current time : " $(date +"%T")
cd $WORK
source ${HOME}/miniconda3/bin/activate deepice

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.0_get_monthly_uv_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_600_5.0

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.1_get_psl_zh.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_701_5.0

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.2_get_monthly_wisoq_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_705_6.0

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.3_get_monthly_tpot_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_600_5.0

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.4_get_monthly_st_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_603_5.3

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.5_get_monthly_q_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_600_5.0

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.6_get_monthly_geop_t_ml.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi pi_603_5.3

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.7_get_6h_wisoq_sfc.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_703_6.0_k52

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.8_get_6h2monthly_wisoq.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_703_6.0_k52

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.9_get_daily_uv.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_703_6.0_k52

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.10_get_era5_uv.sh ${SLURM_ARRAY_TASK_ID}

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.11_get_daily_geop_t_ml.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_703_6.0_k52

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.12_get_6h2daily_wisoq.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_705_6.0

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.13_get_monthly_aps.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_703_6.0_k52

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.14_get_daily2monthly_wisoq.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_703_6.0_k52

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.15_get_daily_wisoq_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_703_6.0_k52

# /albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.16_get_daily_psl_zh.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_703_6.0_k52

/albedo/work/user/qigao001/a_basic_analysis/f_scripts/0_echam_pp/0.17_get_monthly_relhum_plev.sh ${SLURM_ARRAY_TASK_ID} /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi nudged_703_6.0_k52

# python /albedo/work/user/qigao001/a_basic_analysis/f_scripts/1_py_scripts/srun${SLURM_ARRAY_TASK_ID}.py

echo "Current time : " $(date +"%T")

#Xsrun  I know what I am doing

# #SBATCH --cpus-per-task=36
# #SBATCH --ntasks=6
