#!/bin/bash
#SBATCH --time=01:00:00        # Set a limit on the total run time
#SBATCH --partition=mpp       # Specify partition name
#SBATCH --ntasks=1             # Specify max. number of tasks to be invoked
#SBATCH --cpus-per-task=1

output_dir='/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final'
expid='pi_final_qg_1y_1_qgtest2.1'
yrstart=2000
yrend=2009
pp_dir='/work/ollie/qigao001/a_basic_analysis/c_codes/3_shell_scripts/3.0_awiesm_post_processing'

${pp_dir}/job.get.echam.sh ${output_dir} ${expid} ${yrstart} ${yrend}
${pp_dir}/job.get.wiso.sh ${output_dir} ${expid} ${yrstart} ${yrend}
${pp_dir}/job.get.wiso_d.sh ${output_dir} ${expid} ${yrstart} ${yrend}

#Xsrun  I know what I am doing