

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

for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ${YEAR} ${MONTH}

    cdo -dv2uv -selname,sd,svo unknown/${expid}_${YEAR}${MONTH}.01_sp_1d.nc outdata/echam/${expid}_${YEAR}${MONTH}.daily_uv.nc

done

echo 'job done'
