echo $'\n################ calculate monthly, seasonal, and annual statistics'

monthly_data=$1
season_ini=$2
season_end=$3

echo '#### calculate monthly mean'
cdo -P 4 -ymonmean ${monthly_data}.nc ${monthly_data}_mm.nc

echo '#### calculate year mean'
cdo -P 4 -yearmean ${monthly_data}.nc ${monthly_data}_ann.nc

echo '#### calculate annual mean'
cdo -P 4 -timmean ${monthly_data}.nc ${monthly_data}_am.nc

echo '#### calculate seasonal value'
cdo -P 4 -seasmean -seltimestep,${season_ini}/${season_end} ${monthly_data}.nc ${monthly_data}_sea.nc

echo '#### calculate seasonal mean'
cdo -P 4 -yseasmean ${monthly_data}.nc ${monthly_data}_sm.nc
