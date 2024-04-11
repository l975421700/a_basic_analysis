

echo '#-------- basic settings'

YEAR=$1
output_dir=$2
expid=$3
cd ${output_dir}/${expid}

echo 'settings: ' ${YEAR} ${output_dir} ${expid}


echo '#-------- activate conda env'

source ${HOME}/miniconda3/bin/activate deepice
which cdo
which python

cdo="/albedo/soft/sw/spack-sw/cdo/2.2.0-7hyzlde/bin/cdo"

echo '#-------- processing daily output'

for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ${YEAR} ${MONTH}

    mkdir tmp_${YEAR}${MONTH}
    cd tmp_${YEAR}${MONTH}

    cdo -selname,aps ../unknown/${expid}_${YEAR}${MONTH}.01_g3b_1d.nc aps
    cdo -selname,geosp ../unknown/${expid}_${YEAR}${MONTH}.01_echam.nc geosp
    cdo -selname,q ../unknown/${expid}_${YEAR}${MONTH}.01_gl_1d.nc q
    cdo -sp2gp -selname,st ../unknown/${expid}_${YEAR}${MONTH}.01_sp_1d.nc t
    cdo -gheight -merge aps geosp q t geop
    cdo -selname,zh -ml2pl,100000,97500,95000,92500,90000,87500,85000,82500,80000,77500,75000,70000,65000,60000,55000,50000,45000,40000,35000,30000,25000,22500,20000,17500,15000,12500,10000,7000,5000,3000,2000,1000,700,500,300,200,100 -merge t aps geop zh
    cdo -sealevelpressure -merge aps geosp t psl
    cdo -merge psl zh ../outdata/echam/${expid}_${YEAR}${MONTH}.daily_psl_zh.nc

    cd ..
    rm -rf tmp_${YEAR}${MONTH}
done

echo 'job done'



