
expid=$1
yrstart=$2
yrend=$3
# {}
# Set file path
cdo="/global/AWIsoft/cdo/1.9.2/bin/cdo"
output_dir='/work/ollie/qigao001/output/awiesm-2.1-wiso'
calc_wiso_d='/work/ollie/qigao001/a_basic_analysis/d_model_simulations/1_awi_esm/1.1_post_processing/calc_wiso_d.grb.sh'

echo '################ Get wiso_d file'

echo '#### echam + wiso ->  wiso_d'
${calc_wiso_d} ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01

echo '#### mon2am'
${cdo} -timmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso_d.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso_d.am.nc

echo '#### mon2ann'
${cdo} -yearmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso_d.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso_d.ann.nc

echo 'job done'
