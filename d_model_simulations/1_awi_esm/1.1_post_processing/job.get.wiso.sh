
expid=$1
yrstart=$2
yrend=$3
# {}
# Set file path
cdo="/global/AWIsoft/cdo/1.9.2/bin/cdo"
output_dir='/work/ollie/qigao001/output/awiesm-2.1-wiso'

echo '################ Get wiso file'

echo '#### combine grib files'
${cdo} -mergetime ${output_dir}/${expid}/outdata/echam/${expid}_*.01_wiso ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso

echo '#### grib2nc'
${cdo} -f nc -t ${output_dir}/${expid}/outdata/echam/${expid}_${yrstart}01.01_wiso.codes copy ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.nc

echo '#### mon2am'
${cdo} -timmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.am.nc

echo '#### mon2ann'
${cdo} -yearmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.ann.nc

rm ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso

echo 'job done'
