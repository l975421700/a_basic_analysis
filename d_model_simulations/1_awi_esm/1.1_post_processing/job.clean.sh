echo $'\n################ clean extra model output files'

output_dir='/work/ollie/qigao001/output/backup'
expid='pi_final_qg_tag5_1m_4_wisocode_tag'
yrstart=2000
yrend=2000

cd ${output_dir}/${expid}/


echo '#### clean config files'
rm config/*31
rm config/*/*31


echo '#### clean log files'
rm log/*31.log
rm log/*/*31


echo '#### clean code files except first year'
for (( year=(${yrstart}+1); year <= ${yrend}; year++ )); do
    echo $year
    # delete codes files but not of the first year
    rm outdata/echam/${expid}_${year}*.01_*.codes
    rm outdata/jsbach/${expid}_${year}*.01_*.codes
done


echo '#### clean part of the echam6 output'
for (( year=${yrstart}; year <= ${yrend}; year++ )); do
    echo $year
    rm outdata/echam/${expid}_${year}*.01_accw*
    rm outdata/echam/${expid}_${year}*.01_accw_wiso*
    rm outdata/echam/${expid}_${year}*.01_aclcim*
    rm outdata/echam/${expid}_${year}*.01_co2*
    rm outdata/echam/${expid}_${year}*.01_g3b1hi*
    rm outdata/echam/${expid}_${year}*.01_g3bday*
    rm outdata/echam/${expid}_${year}*.01_g3bid*
    rm outdata/echam/${expid}_${year}*.01_g3bim*
    rm outdata/echam/${expid}_${year}*.01_glday*
    rm outdata/echam/${expid}_${year}*.01_glim*
    rm outdata/echam/${expid}_${year}*.01_sp6h*
    rm outdata/echam/${expid}_${year}*.01_spim*
done


echo '#### clean part of the fesom output'
for (( year=${yrstart}; year <= ${yrend}; year++ )); do
    echo $year
     rm outdata/fesom/Av.fesom.${year}*.nc       # Vertical mixing A
     rm outdata/fesom/bolus_u.fesom.${year}*.nc  # GM bolus velocity U
     rm outdata/fesom/bolus_v.fesom.${year}*.nc  # GM bolus velocity V
     rm outdata/fesom/bolus_w.fesom.${year}*.nc  # GM bolus velocity W
     rm outdata/fesom/Kv.fesom.${year}*.nc       # Vertical mixing K
     rm outdata/fesom/N2.fesom.${year}*.nc       # brunt väisälä
     rm outdata/fesom/Redi_K.fesom.${year}*.nc   # Redi diffusion coefficient
     rm outdata/fesom/momix_length.fesom.${year}*.nc   # Monin-Obukov mixing length
done


echo '#### clean part of the jsbach output'
for (( year=${yrstart}; year <= ${yrend}; year++ )); do
    echo $year
     rm outdata/jsbach/${expid}_${year}*.01_jsbid*
done


echo '#### clean unknown output'
rm unknown/*









