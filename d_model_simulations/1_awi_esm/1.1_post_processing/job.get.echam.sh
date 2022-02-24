echo $'\n################ Get echam files'

output_dir=$1
expid=$2
yrstart=$3
yrend=$4

echo '#### activate conda env training'
source /home/ollie/qigao001/miniconda3/bin/activate training
which cdo

echo '#### combine grib files'
cdo -mergetime ${output_dir}/${expid}/outdata/echam/${expid}_*.01_echam ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam

echo '#### grib2nc'
cdo -f nc -t ${output_dir}/${expid}/outdata/echam/${expid}_${yrstart}01.01_echam.codes copy ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.nc

echo '#### mon2am'
cdo -timmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.am.nc

echo '#### mon2ann'
cdo -yearmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.ann.nc

echo '#### mon2mm'
cdo -ymonmean ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.nc ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam.mm.nc

rm ${output_dir}/${expid}/analysis/echam/${expid}_${yrstart}_${yrend}.01_echam

echo 'job done'
