
YEAR=$1

output_dir='output/echam-6.3.05p2-wiso/pi'
expid='pi_m_402_4.7'

# yrstart=2009
# yrend=2029

echo '#---------------- Processing output of ' ${expid}

cd $WORK

cd ${output_dir}

echo '#---- activate conda env training'
source /home/ollie/qigao001/miniconda3/bin/activate training
which cdo


# for (( YEAR=${yrstart}; YEAR <= ${yrend}; YEAR++ )); do
for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ${YEAR} ${MONTH}
    cdo -timmean ${expid}/unknown/${expid}_${YEAR}${MONTH}.01_echam.nc ${expid}/outdata/echam/${expid}_${YEAR}${MONTH}_monthly.01_echam.nc
    cdo -timmean ${expid}/unknown/${expid}_${YEAR}${MONTH}.01_wiso.nc ${expid}/outdata/echam/${expid}_${YEAR}${MONTH}_monthly.01_wiso.nc
done
# done

cd $WORK

