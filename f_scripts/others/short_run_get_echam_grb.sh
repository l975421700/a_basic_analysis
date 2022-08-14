

output_dir='/work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi'
expid='pi_echam6_1d_174_3.60'
yrstart=2000

echo '#---------------- Processing output of ' ${expid}

echo '#---- activate conda env training'
source /home/ollie/qigao001/miniconda3/bin/activate training
which cdo


echo '#---------------- Get echam files'

echo '#### combine grib files'
cdo -mergetime ${output_dir}/${expid}/outdata/echam/${expid}_*.01_echam ${output_dir}/${expid}/analysis/echam/${expid}.01_echam

echo '#### grib2nc'
cdo -b 64 -f nc -t ${output_dir}/${expid}/outdata/echam/${expid}_${yrstart}01.01_echam.codes copy ${output_dir}/${expid}/analysis/echam/${expid}.01_echam ${output_dir}/${expid}/analysis/echam/${expid}.01_echam.nc

rm ${output_dir}/${expid}/analysis/echam/${expid}.01_echam


echo '#---------------- Get wiso files'

echo '#### combine grib files'
cdo -mergetime ${output_dir}/${expid}/outdata/echam/${expid}_*.01_wiso ${output_dir}/${expid}/analysis/echam/${expid}.01_wiso

echo '#### grib2nc'
cdo -b 64 -f nc -t ${output_dir}/${expid}/outdata/echam/${expid}_${yrstart}01.01_wiso.codes copy ${output_dir}/${expid}/analysis/echam/${expid}.01_wiso ${output_dir}/${expid}/analysis/echam/${expid}.01_wiso.nc

rm ${output_dir}/${expid}/analysis/echam/${expid}.01_wiso


echo 'job done'
