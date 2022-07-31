
YEAR=$1

output_dir='output/echam-6.3.05p2-wiso/pi'
expid='pi_m_402_4.7'

echo '#---------------- Processing output of ' ${expid}

cd $WORK

cd ${output_dir}/${expid}

echo '#---- activate conda env training'
source /home/ollie/qigao001/miniconda3/bin/activate training
which cdo


# for (( YEAR=${yrstart}; YEAR <= ${yrend}; YEAR++ )); do
for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ${YEAR} ${MONTH}
    # echo 'calculate monthly values'
    # cdo -timmean unknown/${expid}_${YEAR}${MONTH}.01_echam.nc outdata/echam/${expid}_${YEAR}${MONTH}_monthly.01_echam.nc
    # cdo -timmean unknown/${expid}_${YEAR}${MONTH}.01_wiso.nc outdata/echam/${expid}_${YEAR}${MONTH}_monthly.01_wiso.nc

    # echo 'extract daily values'
    # cdo -selname,aprl,aprc,aprs unknown/${expid}_${YEAR}${MONTH}.01_echam.nc outdata/echam/${expid}_${YEAR}${MONTH}_daily.01_echam.nc
    # cdo -selname,wisoaprl,wisoaprc,wisoaprs unknown/${expid}_${YEAR}${MONTH}.01_wiso.nc outdata/echam/${expid}_${YEAR}${MONTH}_daily.01_wiso.nc

    echo 'get uvq at plev'
    cdo -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 -dv2uv -selname,sd,svo,aps,q,xl,xi outdata/echam/${expid}_${YEAR}${MONTH}_monthly.01_echam.nc outdata/echam/${expid}_${YEAR}${MONTH}_monthly_uvq_plev.01_echam.nc
done
# done

cd $WORK


    # echo 'extract 6 hourly values'
    # cdo -selname,aprl,aprc,aprs unknown/${expid}_${YEAR}${MONTH}.01_echam.nc outdata/echam/${expid}_${YEAR}${MONTH}_6hourly.01_echam.nc
    # cdo -selname,wisoaprl,wisoaprc,wisoaprs unknown/${expid}_${YEAR}${MONTH}.01_wiso.nc outdata/echam/${expid}_${YEAR}${MONTH}_6hourly.01_wiso.nc
    # echo 'calculate daily values'
    # cdo -daymean -selname,aprl,aprc,aprs unknown/${expid}_${YEAR}${MONTH}.01_echam.nc outdata/echam/${expid}_${YEAR}${MONTH}_daily.01_echam.nc
    # cdo -daymean -selname,wisoaprl,wisoaprc,wisoaprs unknown/${expid}_${YEAR}${MONTH}.01_wiso.nc outdata/echam/${expid}_${YEAR}${MONTH}_daily.01_wiso.nc

