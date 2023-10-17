

echo '#-------- basic settings'

YEAR=$1
output_dir=$2
expid=$3
cd ${output_dir}/${expid}

echo 'settings: ' ${YEAR} ${output_dir} ${expid}


echo '#-------- activate conda env'

source ${HOME}/miniconda3/bin/activate deepice

cdo="/albedo/soft/sw/spack-sw/cdo/2.2.0-7hyzlde/bin/cdo"

echo '#-------- processing monthly output'

task(){
    MONTH=$1
    echo ${YEAR} ${MONTH}

    mkdir tmp_${YEAR}${MONTH}
    cd tmp_${YEAR}${MONTH}

    ${cdo} -selname,aps ../unknown/${expid}_${YEAR}${MONTH}.01_g3b_1m.nc aps

    # ${cdo} -merge aps ../unknown/${expid}_${YEAR}${MONTH}.01_wiso_q_1m.nc aps_q
    ${cdo} -merge aps ../unknown/${expid}_${YEAR}${MONTH}.01_wiso_q_6h_mon.nc aps_q

    ${cdo} -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 aps_q ../outdata/echam/${expid}_${YEAR}${MONTH}.monthly_wiso_q_plev.nc

    cd ..
    rm -rf tmp_${YEAR}${MONTH}
}

for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    task ${MONTH} &
done

wait

# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop


echo 'job done'


# #-------- check
# source /home/ollie/qigao001/miniconda3/bin/activate deepice

# cd /work/ollie/qigao001/scratch/test
# ${cdo} -selname,aps /work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_208912.01_g3b_1m.nc aps

# ${cdo} -merge aps -selname,q_25 /work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_208912.01_wiso_q_1m.nc aps_q

# ${cdo} -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 aps_q wiso_q_plev

# ipython
# import xarray as xr
# import numpy as np
# ncfile1 = xr.open_dataset('/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/outdata/echam/pi_m_416_4.9_208912.monthly_wiso_q_plev.nc')

# ncfile2 = xr.open_dataset('/work/ollie/qigao001/scratch/test/wiso_q_plev')

# # (ncfile2.q_25.values[np.isfinite(ncfile2.q_25.values)] == \
# #     ncfile1.q_25.values[np.isfinite(ncfile1.q_25.values)]).all()

# np.max(abs(ncfile2.q_25.values[np.isfinite(ncfile2.q_25.values)] - \
#     ncfile1.q_25.values[np.isfinite(ncfile1.q_25.values)]))

# test = ncfile2.q_25.values[np.isfinite(ncfile2.q_25.values)] - \
#     ncfile1.q_25.values[np.isfinite(ncfile1.q_25.values)]

# wheremax = np.where(test == np.max(abs(test)))
# print(np.max(abs(test)))
# print(test[wheremax])
# print(ncfile2.q_25.values[np.isfinite(ncfile2.q_25.values)][wheremax])
# print(ncfile1.q_25.values[np.isfinite(ncfile1.q_25.values)][wheremax])


# #-------- sequential version
# for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
#     echo ${YEAR} ${MONTH}

#     mkdir tmp_${YEAR}${MONTH}
#     cd tmp_${YEAR}${MONTH}

#     ${cdo} -selname,aps ../unknown/${expid}_${YEAR}${MONTH}.01_g3b_1m.nc aps

#     ${cdo} -merge aps ../unknown/${expid}_${YEAR}${MONTH}.01_wiso_q_1m.nc aps_q

#     ${cdo} -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 aps_q ../outdata/echam/${expid}_${YEAR}${MONTH}.monthly_wiso_q_plev.nc

#     cd ..
#     rm -rf tmp_${YEAR}${MONTH}
# done
