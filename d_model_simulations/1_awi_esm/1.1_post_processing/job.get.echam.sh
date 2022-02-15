
expid=$1
yrstart=$2
yrend=$3
# {}
# Set file path
cdo="/global/AWIsoft/cdo/1.9.2/bin/cdo"
output_dir='/work/ollie/qigao001/output/awiesm-2.1-wiso'

echo '################ Combining echam file'

${cdo} -mergetime ${output_dir}/${expid}/outdata/echam/${expid}_*.01_echam ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam

${cdo} -f nc -t ${output_dir}/${expid}/outdata/echam/${expid}_${yrstart}01.01_echam.codes copy ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.nc

${cdo} -timmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.am.nc

${cdo} -yearmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.ann.nc

rm ${output_dir}/${expid}/outdata/echam/${expid}_*.01_echam ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam

echo 'job done'
