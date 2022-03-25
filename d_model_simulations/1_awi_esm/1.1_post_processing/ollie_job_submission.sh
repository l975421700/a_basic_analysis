#!/bin/bash
#SBATCH --output=job.out
#SBATCH --time=02:00:00        # Set a limit on the total run time
#SBATCH -p mpp                 # Specify partition name
#SBATCH --ntasks=1             # Specify max. number of tasks to be invoked
#SBATCH --cpus-per-task=1

echo "Current time : " $(date +"%T")

# cd /home/ollie/qigao001/model_codes/awiesm-2.1-wiso
# srun ./comp-echam-6.3.05p2-wiso_script.sh

echo "Current time : " $(date +"%T")

#Xsrun  I know what I am doing