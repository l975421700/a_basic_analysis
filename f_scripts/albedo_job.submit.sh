#!/bin/bash
#SBATCH -p mpp
#SBATCH --qos=12h
#SBATCH --time=12:00:00
#SBATCH -o %A.out
#SBATCH -e %A.err
#SBATCH --mem=240GB

echo "Current time : " $(date +"%T")
cd $WORK
source ${HOME}/miniconda3/bin/activate deepice

python /albedo/work/user/qigao001/a_basic_analysis/f_scripts/1_py_scripts/srun3.py

echo "Current time : " $(date +"%T")

#Xsrun  I know what I am doing




# output_dir='/home/ollie/qigao001/output/awiesm-2.1-wiso/pi_final'
# expid='pi_final_qg_1y_1_qgtest2.1'
# yrstart=2000
# yrend=2009
# pp_dir='/work/ollie/qigao001/a_basic_analysis/c_codes/3_shell_scripts/3.0_awiesm_post_processing'
# ${pp_dir}/job.get.echam.sh ${output_dir} ${expid} ${yrstart} ${yrend}
# ${pp_dir}/job.get.wiso.sh ${output_dir} ${expid} ${yrstart} ${yrend}
# ${pp_dir}/job.get.wiso_d.sh ${output_dir} ${expid} ${yrstart} ${yrend}

# partition: https://spaces.awi.de/display/HELP/Slurm-Albedo

# #SBATCH --ntasks=1
# #SBATCH --account=paleodyn.paleodyn


