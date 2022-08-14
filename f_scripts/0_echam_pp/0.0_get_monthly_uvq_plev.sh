

echo '#-------- basic settings'

YEAR=$1
output_dir=$2
expid=$3
cd ${WORK}/${output_dir}/${expid}

echo 'settings: ' ${YEAR} ${output_dir} ${expid}


echo '#-------- activate conda env'

source /home/ollie/qigao001/miniconda3/bin/activate training
which cdo
which python


echo '#-------- processing monthly output'

for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ${YEAR} ${MONTH}

    cdo -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 -merge -dv2uv -selname,sd,svo unknown/${expid}_${YEAR}${MONTH}.01_sp_1m.nc -selname,aps unknown/${expid}_${YEAR}${MONTH}.01_g3b_1m.nc -selname,q,xl,xi unknown/${expid}_${YEAR}${MONTH}.01_gl_1m.nc outdata/echam/${expid}_${YEAR}${MONTH}.monthly_uvq_plev.nc

done

echo 'job done'


# #-------- check

# source /home/ollie/qigao001/miniconda3/bin/activate training

# cdo -selname,sd,svo output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200001.01_sp_1m.nc scratch/test/test0.nc

# cdo -selname,aps output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200001.01_g3b_1m.nc scratch/test/test1.nc

# cdo -selname,q,xl,xi output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200001.01_gl_1m.nc scratch/test/test2.nc

# cdo -dv2uv scratch/test/test0.nc scratch/test/test3.nc

# cdo -merge scratch/test/test3.nc scratch/test/test1.nc scratch/test/test2.nc scratch/test/test4.nc

# cdo -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 scratch/test/test4.nc scratch/test/test5.nc

# #---- check in python
# ipython

# import xarray as xr
# import numpy as np

# nc1 = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200001.01_sp_1m.nc')
# nc2 = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200001.01_g3b_1m.nc')
# nc3 = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200001.01_gl_1m.nc')
# nc4 = xr.open_dataset('scratch/test/test3.nc')

# nc01 = xr.open_dataset('scratch/test/test4.nc')
# nc02 = xr.open_dataset('scratch/test/test5.nc')
# nc11 = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/outdata/echam/pi_m_416_4.9_200001.monthly_uvq_plev.nc')

# # merge works fine: yes
# np.max(abs(nc4.u.values - nc01.u.values))
# np.max(abs(nc4.v.values - nc01.v.values))

# # two ways give the same results: yes

# np.max(abs(nc02.u.values[np.isfinite(nc02.u.values)] - nc11.u.values[np.isfinite(nc11.u.values)]))
# np.max(abs(nc02.v.values[np.isfinite(nc02.v.values)] - nc11.v.values[np.isfinite(nc11.v.values)]))
