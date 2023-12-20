

echo '#-------- basic settings'
YEAR=$1

echo '#-------- activate conda env'
cdo="/albedo/soft/sw/spack-sw/cdo/2.2.0-7hyzlde/bin/cdo"

echo '#-------- processing output'
for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ${YEAR} ${MONTH}

    cdo -dv2uv -selname,sd,svo /albedo/work/projects/paleo_work/paleodyn_from_work_ollie_projects/paleodyn/nudging/ERA5/atmos/T63/era5T63L47_${YEAR}${MONTH}.nc /albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/nudged_703_6.0_k52/forcing/echam/era5_uv_${YEAR}${MONTH}.nc

done

echo 'job done'
