echo $'\n################ Get wiso files'

output_dir=$1
expid=$2
yrstart=$3
yrend=$4

echo '#### activate conda env training'
source /home/ollie/qigao001/miniconda3/bin/activate training
which cdo

echo '#### combine grib files'
cdo -mergetime ${output_dir}/${expid}/outdata/echam/${expid}_*.01_wiso ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso

echo '#### grib2nc'
cdo -f nc -t ${output_dir}/${expid}/outdata/echam/${expid}_${yrstart}01.01_wiso.codes copy ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.nc

echo '#### mon2am'
cdo -timmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.am.nc

echo '#### mon2ann'
cdo -yearmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.ann.nc

echo '#### mon2mm'
cdo -ymonmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso.mm.nc

rm ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_wiso

echo 'job done'
