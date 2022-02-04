#!/bin/bash

module load cdo

cat << EOF > namelist.after2
  &SELECT
    CODE = 151
    TYPE = 70,
    FORMAT = 2
  &END
EOF


# ***** time extents *****
for (( expid=3; expid <= 3; expid++ ))
do

   if [ $expid -eq 1 ];  then
       export BEGIN_YEAR=2000
       export END_YEAR=2900
       export EXPID=PI_ctrl_4xCO2_awiesm-2.1_LR
       export DATA_DIR=/work/ba1066/a270061/esm_experiments_v4/$EXPID/outdata/echam/
   elif [ $expid -eq 2 ];  then
       export BEGIN_YEAR=2492
       export END_YEAR=2800
       export EXPID=PI_ctrl_awiesm-2.1_LR
       export DATA_DIR=/work/ba1066/a270061/esm_experiments_v4/$EXPID/outdata/echam/
fi

# *****  Base Directories **********
mkdir /pf/a/a270098/ab0246/AWIESM/
export DES_DIR=/pf/a/a270098/ab0246/AWIESM/$EXPID
mkdir $DES_DIR


for (( YEAR=${BEGIN_YEAR}; YEAR <= ${END_YEAR}; YEAR++ ))
do
    echo $YEAR
    cdo mergetime ${DATA_DIR}/${EXPID}_$YEAR*.01_echam ${DES_DIR}/$YEAR.grb
#    cdo select,name=var178,var179,var103,var169,var142,var143,var210,var103,var165,var166,var182,var184 ${DES_DIR}/$YEAR.grb ${DES_DIR}/${YEAR}_target.grb
#    cdo -f nc copy ${DES_DIR}/${YEAR}_target.grb ${DES_DIR}/${YEAR}_target.nc
    cdo -f nc copy ${DES_DIR}/${YEAR}.grb ${DES_DIR}/${YEAR}_target.nc
    cdo after ${DES_DIR}/${YEAR}.grb  ${DES_DIR}/echam_${YEAR}_after.nc < namelist.after2
    cdo merge ${DES_DIR}/echam_${YEAR}_after.nc ${DES_DIR}/${YEAR}_target.nc ${DES_DIR}/${YEAR}_final.nc
    rm ${DES_DIR}/${YEAR}_target.grb ${DES_DIR}/$YEAR.grb ${DES_DIR}/echam_${YEAR}_after.nc ${DES_DIR}/${YEAR}_target.nc
done
done
exit



