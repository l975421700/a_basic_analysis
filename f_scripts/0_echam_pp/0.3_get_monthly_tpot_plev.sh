

echo '#-------- basic settings'

YEAR=$1
output_dir=$2
expid=$3
cd ${WORK}/${output_dir}/${expid}

echo 'settings: ' ${YEAR} ${output_dir} ${expid}


echo '#-------- get cdo'

cdo="/global/AWIsoft/cdo/1.9.8/bin/cdo"
echo ${cdo}

echo '#-------- processing monthly output'

for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ${YEAR} ${MONTH}

    ${cdo} -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 -selname,aps,tpot unknown/${expid}_${YEAR}${MONTH}.01_g3b_1m.nc outdata/echam/${expid}_${YEAR}${MONTH}.monthly_tpot_plev.nc

done

echo 'job done'


