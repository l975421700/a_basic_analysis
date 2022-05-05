#!/bin/bash
#SBATCH --partition=par-multi
#SBATCH --time=10:00:00
#SBATCH -o job.out
#SBATCH -e job.err

# SBATCH --array=1982-2016
# SBATCH -o %A_%a.out
# SBATCH -e %A_%a.err

# /home/users/qino/a_basic_analysis/c_codes/1_cmip6/1.0_hist/1.0.4_sst/1.0.4.0_combine_daily_esacci_sst.sh ${SLURM_ARRAY_TASK_ID}

# /home/users/qino/a_basic_analysis/c_codes/1_cmip6/1.0_hist/1.0.4_sst/1.0.4.1_combine_monthly_esacci_sst.sh ${SLURM_ARRAY_TASK_ID}

source /home/users/qino/miniconda3/bin/activate deepice
which cdo

echo $(date)
echo 'job started'

cdo -mergetime /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0_*_yearly_monthly.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/sst_mon_ESACCI-2.1_198201_201612.nc

echo 'job finished'
echo $(date)