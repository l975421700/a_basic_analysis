#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=mpp120
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH -o %A_job.out
#SBATCH -e %A_job.err

source /home/ollie/qigao001/miniconda3/bin/activate deepice

cd $WORK

python "/work/ollie/qigao001/a_basic_analysis/c_codes/2_tagging/2.5_extreme_precipitation/2.5.0.2.2_get_transport_distance_epe_bin.py"

#Xsrun  I know what I am doing


# data_sources, output, scratch, startdump
# rsync -avzLP --exclude '**/run_/*' /work/ollie/qigao001/startdump levante:/work/ab0246/a270200/

# output_dir='/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final'
# expid='pi_final_qg_1y_1_qgtest2.1'
# yrstart=2000
# yrend=2009
# pp_dir='/work/ollie/qigao001/a_basic_analysis/c_codes/3_shell_scripts/3.0_awiesm_post_processing'
# ${pp_dir}/job.get.echam.sh ${output_dir} ${expid} ${yrstart} ${yrend}
# ${pp_dir}/job.get.wiso.sh ${output_dir} ${expid} ${yrstart} ${yrend}
# ${pp_dir}/job.get.wiso_d.sh ${output_dir} ${expid} ${yrstart} ${yrend}

# /work/ollie/qigao001/a_basic_analysis/f_scripts/short_run_get_echam_nc.sh

# python /work/ollie/qigao001/a_basic_analysis/c_codes/2_awiesm/2.0._tagging/2.0.1.0_check_tagging_results/2.0.1.0.0_collect_correction_factors.py


# partition: https://spaces.awi.de/display/HELP/SLURM


