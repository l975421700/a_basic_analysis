

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

    mkdir tmp_${YEAR}${MONTH}
    cd tmp_${YEAR}${MONTH}

    cdo -selname,aps ../unknown/${expid}_${YEAR}${MONTH}.01_g3b_1m.nc aps
    cdo -selname,geosp ../unknown/${expid}_${YEAR}${MONTH}.01_echam.nc geosp
    cdo -selname,q ../unknown/${expid}_${YEAR}${MONTH}.01_gl_1m.nc q
    cdo -sp2gp -selname,st ../unknown/${expid}_${YEAR}${MONTH}.01_sp_1m.nc t
    cdo -gheight -merge aps geosp q t geop
    cdo -selname,zh -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 -merge t aps geop zh
    cdo -sealevelpressure -merge aps geosp t psl
    cdo -merge psl zh ../outdata/echam/${expid}_${YEAR}${MONTH}.monthly_psl_zh.nc

    cd ..
    rm -rf tmp_${YEAR}${MONTH}
done

echo 'job done'




#-------------------------------- check
# region

# cdo -sealevelpressure -merge -selname,aps output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_g3b_1m.nc -selname,geosp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc -sp2gp -selname,st output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_sp_1m.nc test.nc

# cdo -selname,zh -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 -merge -sp2gp -selname,st output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_sp_1m.nc -selname,aps output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_g3b_1m.nc -gheight -merge -selname,aps output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_g3b_1m.nc -selname,geosp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc -selname,q output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_gl_1m.nc -sp2gp -selname,st output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_sp_1m.nc test1.nc


# #---- compare two estimates in python

# ipython

# import xarray as xr
# import numpy as np

# nc1 = xr.open_dataset('output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/outdata/echam/pi_m_416_4.9_200002.monthly_psl_zh.nc')
# nc2 = xr.open_dataset('test.nc')
# nc3 = xr.open_dataset('test1.nc')
# (nc1.psl == nc2.psl).all().values
# (nc1.zh.values[np.isfinite(nc1.zh.values)] == nc3.zh.values[np.isfinite(nc3.zh.values)]).all()

# test = (nc1.zh.values[np.isfinite(nc1.zh.values)] - nc3.zh.values[np.isfinite(nc3.zh.values)])
# wheremax = np.where(abs(test) == np.max(abs(test)))
# np.max(abs(test))
# test[wheremax]
# nc1.zh.values[np.isfinite(nc1.zh.values)][wheremax]
# nc3.zh.values[np.isfinite(nc3.zh.values)][wheremax]


# endregion


# region
# ! cdo -sealevelpressure -sp2gp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc scratch/test/test1.nc
# ! cdo -gheight -sp2gp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc scratch/test/test2.nc

# ! cdo after output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc scratch/test/test3.nc < /work/ollie/qigao001/a_basic_analysis/f_scripts/others/cdo_afterburner/namelistfile

# import xarray as xr
# import numpy as np

# nc1 = xr.open_dataset('scratch/test/test1.nc')
# nc2 = xr.open_dataset('scratch/test/test2.nc')
# nc3 = xr.open_dataset('scratch/test/test3.nc')


# (nc1.psl.values == nc3.var151.values).all()


# (nc2.zh.values[np.isfinite(nc2.zh.values)] == nc3.var156.values[np.isfinite(nc3.var156.values)]).all()

# test = nc2.zh.values[np.isfinite(nc2.zh.values)] - nc3.var156.values[np.isfinite(nc3.var156.values)]
# wheremax = np.where(abs(test) == np.max(abs(test)))
# np.max(abs(test))
# test[wheremax]
# nc2.zh.values[np.isfinite(nc2.zh.values)][wheremax]
# nc3.var156.values[np.isfinite(nc3.var156.values)][wheremax]

# endregion


#-------------------------------- derivation
# region
# #-------- calculate mslp
# # Sea level pressure
# # Required fields: surface_air_pressure, surface_geopotential, air_temperature

# #---- calculation based on timmean values
# ! cdo -sealevelpressure -merge -selname,aps output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_g3b_1m.nc -selname,geosp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc -sp2gp -selname,st output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_sp_1m.nc test.nc

# #---- calculation based on instantaneous values
# # deprecated. ERA5 also uses timmean values
# # ! cdo -sealevelpressure -sp2gp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc test1.nc


# #-------- calculate geop
# # Geopotential height
# # Required fields: surface_air_pressure, surface_geopotential, specific_humidity, air_temperature
# # Note, this procedure is an approximation, which doesn't take into account the effects of e.g. cloud ice and water, rain and snow.

# #---- calculation based on timmean values
# ! cdo -gheight -merge -selname,aps output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_g3b_1m.nc -selname,geosp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc -selname,q output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_gl_1m.nc -sp2gp -selname,st output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_sp_1m.nc test.nc


# #---- calculation based on instantaneous values
# # deprecated.
# # ! cdo -gheight -sp2gp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc test1.nc


# ! cdo -merge -sealevelpressure -merge -selname,aps output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_g3b_1m.nc -selname,geosp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc -sp2gp -selname,st output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_sp_1m.nc -selname,zh -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 -merge -sp2gp -selname,st output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_sp_1m.nc -selname,aps output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_g3b_1m.nc -gheight -merge -selname,aps output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_g3b_1m.nc -selname,geosp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc -selname,q output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_gl_1m.nc -sp2gp -selname,st output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_sp_1m.nc test.nc

# # instantaneous: deprecated
# # ! cdo -merge -sealevelpressure -sp2gp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc -selname,zh -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 -merge -sp2gp -selname,st output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_sp_1m.nc -selname,aps output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_g3b_1m.nc -gheight -sp2gp output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9/unknown/pi_m_416_4.9_200002.01_echam.nc test.nc

# endregion

