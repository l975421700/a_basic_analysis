

echo '#-------- basic settings'

YEAR=$1
output_dir=$2
expid=$3
cd ${output_dir}/${expid}

echo 'settings: ' ${YEAR} ${output_dir} ${expid}


echo '#-------- activate conda env'

source ${HOME}/miniconda3/bin/activate deepice

cdo="/albedo/soft/sw/spack-sw/cdo/2.2.0-7hyzlde/bin/cdo"


echo '#-------- processing daily output'

task(){
    MONTH=$1
    echo ${YEAR} ${MONTH}

    ${cdo} --timestat_date first -monmean ./unknown/${expid}_${YEAR}${MONTH}.01_wiso_q_6h_daily.nc ./unknown/${expid}_${YEAR}${MONTH}.01_wiso_q_6h_mon.nc
}

for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    task ${MONTH} &
done

wait

# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop

echo 'job done'
