echo $'\n################ Get wiso_d files'

output_dir=$1
expid=$2
yrstart=$3
yrend=$4

echo '#### activate conda env training'
source /home/ollie/qigao001/miniconda3/bin/activate training
which cdo

calc_wiso_d='/work/ollie/qigao001/a_basic_analysis/d_model_simulations/1_awi_esm/1.1_post_processing/calc_wiso_d.grb.sh'

echo '#### mon echam + wiso ->  wiso_d'
${calc_wiso_d} ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01

echo '#### am echam + wiso ->  wiso_d'
${calc_wiso_d} ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01 '.am'

echo '#### ann echam + wiso ->  wiso_d'
${calc_wiso_d} ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01 '.ann'

echo '#### mm echam + wiso ->  wiso_d'
${calc_wiso_d} ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01 '.mm'

# calculate arithmetic mean of wiso_d
echo '#### mon2am'
cdo -timmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso_d.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso_d.am_uw.nc

echo '#### mon2ann'
cdo -yearmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso_d.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso_d.ann_uw.nc

echo '#### mon2mm'
cdo -ymonmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso_d.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso_d.mm_uw.nc

# calculate prec.weighted mean of wiso_d


echo 'job done'
