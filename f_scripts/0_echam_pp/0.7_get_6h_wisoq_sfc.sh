

echo '#-------- basic settings'

YEAR=$1
output_dir=$2
expid=$3
cd ${output_dir}/${expid}

echo 'settings: ' ${YEAR} ${output_dir} ${expid}

echo '#-------- activate conda env'

source ${HOME}/miniconda3/bin/activate deepice

cdo="/albedo/soft/sw/spack-sw/cdo/2.2.0-7hyzlde/bin/cdo"

echo '#-------- processing 6 hourly output'

task(){
    MONTH=$1
    echo ${YEAR} ${MONTH}

    ${cdo} -sellevel,47 unknown/${expid}_${YEAR}${MONTH}.01_wiso_q_6h.nc unknown/${expid}_${YEAR}${MONTH}.01_wiso_q_6h_sfc.nc

}

for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    task ${MONTH} &
done

wait

echo 'job done'

