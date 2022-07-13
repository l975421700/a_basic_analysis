#!/bin/bash
#SBATCH --time=03:30:00        # Set a limit on the total run time
#SBATCH --partition=mpp       # Specify partition name
#SBATCH --ntasks=1             # Specify max. number of tasks to be invoked
#SBATCH --cpus-per-task=1

# output_dir='/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final'
# expid='pi_final_qg_1y_1_qgtest2.1'
# yrstart=2000
# yrend=2009
# pp_dir='/work/ollie/qigao001/a_basic_analysis/c_codes/3_shell_scripts/3.0_awiesm_post_processing'

# ${pp_dir}/job.get.echam.sh ${output_dir} ${expid} ${yrstart} ${yrend}
# ${pp_dir}/job.get.wiso.sh ${output_dir} ${expid} ${yrstart} ${yrend}
# ${pp_dir}/job.get.wiso_d.sh ${output_dir} ${expid} ${yrstart} ${yrend}


/work/ollie/qigao001/a_basic_analysis/f_shell_scripts/short_run_get_echam_nc.sh

# source /home/ollie/qigao001/miniconda3/bin/activate training
# which python

# python /work/ollie/qigao001/a_basic_analysis/c_codes/2_awiesm/2.0._tagging/2.0.1.0_check_tagging_results/2.0.1.0.0_collect_correction_factors.py

#Xsrun  I know what I am doing