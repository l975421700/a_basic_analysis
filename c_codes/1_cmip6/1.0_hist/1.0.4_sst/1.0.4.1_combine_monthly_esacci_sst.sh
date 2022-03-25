
source /home/users/qino/miniconda3/bin/activate deepice
which cdo

YEAR=$1

echo 'mergetime monthly data'

cdo -P 4 -mergetime /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0_${YEAR}_*_monthly.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0_${YEAR}_yearly_monthly.nc


# # check
# export YEAR=1982
# export MONTH=01
# cdo -P 4 -mergetime /neodc/esacci/sst/data/CDR_v2/Analysis/L4/v2.0/${YEAR}/${MONTH}/*/*.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.0-v02.0-fv01.0_${YEAR}_${MONTH}_daily.nc
# cdo -P 4 -monmean /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.0-v02.0-fv01.0_${YEAR}_${MONTH}_daily.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.0-v02.0-fv01.0_${YEAR}_${MONTH}_monthly.nc


