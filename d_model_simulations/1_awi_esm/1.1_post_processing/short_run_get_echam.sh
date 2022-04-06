

output_dir='/work/ollie/qigao001/output/awiesm-2.1-wiso/pi_final'
expid='pi_final_qg_tag5_1y_7_qgtest5'
yrstart=2000


echo '#### activate conda env training'
source /home/ollie/qigao001/miniconda3/bin/activate training
which cdo


echo '################ Get echam files'

echo '#### combine grib files'
cdo -mergetime ${output_dir}/${expid}/outdata/echam/${expid}_*.01_echam ${output_dir}/${expid}/analysis/echam/${expid}.01_echam

echo '#### grib2nc'
cdo -f nc -t ${output_dir}/${expid}/outdata/echam/${expid}_${yrstart}01.01_echam.codes copy ${output_dir}/${expid}/analysis/echam/${expid}.01_echam ${output_dir}/${expid}/analysis/echam/${expid}.01_echam.nc

rm ${output_dir}/${expid}/analysis/echam/${expid}.01_echam


echo '################ Get wiso files'

echo '#### combine grib files'
cdo -mergetime ${output_dir}/${expid}/outdata/echam/${expid}_*.01_wiso ${output_dir}/${expid}/analysis/echam/${expid}.01_wiso

echo '#### grib2nc'
cdo -f nc -t ${output_dir}/${expid}/outdata/echam/${expid}_${yrstart}01.01_wiso.codes copy ${output_dir}/${expid}/analysis/echam/${expid}.01_wiso ${output_dir}/${expid}/analysis/echam/${expid}.01_wiso.nc

rm ${output_dir}/${expid}/analysis/echam/${expid}.01_wiso


echo 'job done'
