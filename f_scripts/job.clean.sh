
output_dir='/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi'
expid='pi_610_5.8'

echo $'\n#-------- clean model output: ' ${expid}


cd ${output_dir}/${expid}/


echo '#---- clean config files'
rm config/${expid}_filelist_*
rm config/echam/${expid}_filelist_*
rm config/hdmodel/${expid}_filelist_*
rm config/jsbach/${expid}_filelist_*

echo '#---- clean log files'
rm log/*observe_compute*.log
rm log/*prepcompute*.log
rm log/*tidy*.log
rm log/run_folders.log

# echo '#---- clean echam6 output'
# rm outdata/echam/${expid}_*.01_accw*
# rm outdata/echam/${expid}_*.01_aclcim*
# rm outdata/echam/${expid}_*.01_g3b1hi*
# rm outdata/echam/${expid}_*.01_g3bday*
# rm outdata/echam/${expid}_*.01_g3bid*
# rm outdata/echam/${expid}_*.01_g3bim*
# rm outdata/echam/${expid}_*.01_glday*
# rm outdata/echam/${expid}_*.01_glim*
# rm outdata/echam/${expid}_*.01_jsbid*
# rm outdata/echam/${expid}_*.01_sp6h*
# rm outdata/echam/${expid}_*.01_spim*
# rm outdata/echam/${expid}_*.01_co2*
# rm outdata/echam/${expid}_*.01_ma*
# rm outdata/echam/${expid}_*.01_surf*

# echo '#---- clean jsbach output'
# rm outdata/jsbach/${expid}_*.01_js_wiso*
# rm outdata/jsbach/${expid}_*.01_jsbach*
# rm outdata/jsbach/${expid}_*.01_la_wiso*
# rm outdata/jsbach/${expid}_*.01_land*

echo '#---- clean restart files'

for MONTH in 01 02 03 04 05 06 07 08 09 10 11; do
    rm restart/echam/restart_${expid}_*_????${MONTH}??.nc
    rm restart/jsbach/restart_${expid}_????${MONTH}*.nc
done

echo '#---- clean runfolders'
rm -rf run_*


echo '#---- clean unknown output'
# rm unknown/${expid}_*.01_co2*
# rm unknown/${expid}_*.01_ma*
# rm unknown/${expid}_*.01_sf_wiso*
# rm unknown/${expid}_*.01_surf*
# rm unknown/${expid}_*.01_aclcim*
# rm unknown/${expid}_*.01_g3b1hi*
# rm unknown/${expid}_*.01_g3bday*
# rm unknown/${expid}_*.01_g3bid*
# rm unknown/${expid}_*.01_g3bim*
# rm unknown/${expid}_*.01_glday*
# rm unknown/${expid}_*.01_glim*
# rm unknown/${expid}_*.01_jsbid*
# rm unknown/${expid}_*.01_sp6h*
# rm unknown/${expid}_*.01_spim*
rm unknown/restart*


echo 'job done'
# cd $WORK



# echo '#### clean code files except first year'
# for (( year=(${yrstart}+1); year <= ${yrend}; year++ )); do
#     echo $year
#     # delete codes files but not of the first year
#     rm outdata/echam/${expid}_${year}*.01_*.codes
#     rm outdata/jsbach/${expid}_${year}*.01_*.codes
# done


# echo '#### clean part of the fesom output'
# for (( year=${yrstart}; year <= ${yrend}; year++ )); do
#     echo $year
#      rm outdata/fesom/Av.fesom.${year}*.nc       # Vertical mixing A
#      rm outdata/fesom/bolus_u.fesom.${year}*.nc  # GM bolus velocity U
#      rm outdata/fesom/bolus_v.fesom.${year}*.nc  # GM bolus velocity V
#      rm outdata/fesom/bolus_w.fesom.${year}*.nc  # GM bolus velocity W
#      rm outdata/fesom/Kv.fesom.${year}*.nc       # Vertical mixing K
#      rm outdata/fesom/N2.fesom.${year}*.nc       # brunt väisälä
#      rm outdata/fesom/Redi_K.fesom.${year}*.nc   # Redi diffusion coefficient
#      rm outdata/fesom/momix_length.fesom.${year}*.nc   # Monin-Obukov mixing length
# done







