
source /home/users/qino/miniconda3/bin/activate deepice
which cdo

YEAR=$1

for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ${YEAR} ${MONTH} 'mergetime'
    cdo -P 4 -mergetime /neodc/esacci/sst/data/CDR_v2/Analysis/L4/v2.1/${YEAR}/${MONTH}/*/*ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0_${YEAR}_${MONTH}_daily.nc
    echo 'monmean'
    cdo -P 4 -monmean /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0_${YEAR}_${MONTH}_daily.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0_${YEAR}_${MONTH}_monthly.nc
    echo 'done'
done

# run in serial

# for (( YEAR=1982; YEAR <= 2016; YEAR++ )); do
#     for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
#         echo ${YEAR} ${MONTH}
#         cdo -P 4 -mergetime /neodc/esacci/sst/data/CDR_v2/Analysis/L4/v2.0/${YEAR}/${MONTH}/*/*.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.0-v02.0-fv01.0_${YEAR}_${MONTH}_daily.nc

#         cdo -P 4 -monmean /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.0-v02.0-fv01.0_${YEAR}_${MONTH}_daily.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.0-v02.0-fv01.0_${YEAR}_${MONTH}_monthly.nc
#     done
# done


# # check
# export YEAR=1982
# export MONTH=01
# cdo -P 4 -mergetime /neodc/esacci/sst/data/CDR_v2/Analysis/L4/v2.0/${YEAR}/${MONTH}/*/*.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.0-v02.0-fv01.0_${YEAR}_${MONTH}_daily.nc
# cdo -P 4 -monmean /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.0-v02.0-fv01.0_${YEAR}_${MONTH}_daily.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/sst/esacci/ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.0-v02.0-fv01.0_${YEAR}_${MONTH}_monthly.nc


