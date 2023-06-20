

echo '#-------- basic settings'

YEAR=$1
output_dir=$2
expid=$3
cd ${output_dir}/${expid}

echo 'settings: ' ${YEAR} ${output_dir} ${expid}


echo '#-------- activate conda env'

source ${HOME}/miniconda3/bin/activate deepice

cdo="/albedo/soft/sw/spack-sw/cdo/2.2.0-7hyzlde/bin/cdo"

echo '#-------- processing monthly output'

for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ${YEAR} ${MONTH}

    mkdir tmp_${YEAR}${MONTH}
    cd tmp_${YEAR}${MONTH}

    cdo -selname,aps ../unknown/${expid}_${YEAR}${MONTH}.01_g3b_1m.nc aps
    cdo -selname,geosp ../unknown/${expid}_${YEAR}${MONTH}.01_echam.nc geosp
    cdo -selname,q ../unknown/${expid}_${YEAR}${MONTH}.01_gl_1m.nc q
    cdo -sp2gp -selname,st ../unknown/${expid}_${YEAR}${MONTH}.01_sp_1m.nc t
    cdo -gheight -merge aps geosp q t geop
    cdo -merge geop t ../outdata/echam/${expid}_${YEAR}${MONTH}.monthly_geop_t.nc
    cd ..
    rm -rf tmp_${YEAR}${MONTH}
done

echo 'job done'


