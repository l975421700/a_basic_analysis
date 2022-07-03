

output_dir='/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi'
expid='pi_echam6_1y_213_3.60'

echo '################ Processing output of ' ${expid}

echo '#### activate conda env training'
source /home/ollie/qigao001/miniconda3/bin/activate training
which cdo


echo '################ Get echam files'

cdo -mergetime ${output_dir}/${expid}/unknown/${expid}_200*.01_echam.nc ${output_dir}/${expid}/analysis/echam/${expid}.01_echam.nc


echo '################ Get wiso files'

cdo -mergetime ${output_dir}/${expid}/unknown/${expid}_200*.01_wiso.nc ${output_dir}/${expid}/analysis/echam/${expid}.01_wiso.nc


echo 'job done'
