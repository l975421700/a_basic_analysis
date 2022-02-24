echo $'\n################ clean extra model output files'

output_dir=$1
expid=$2
yrstart=$3
yrend=$4

echo '#### clean code files except first year'
for (( year=(${yrstart}+1); year <= ${yrend}; year++ )); do
    echo $year
    # delete codes files but not of the first year
    rm ${output_dir}/${expid}/outdata/echam/${expid}_${year}*.01_*.codes
    rm ${output_dir}/${expid}/outdata/jsbach/${expid}_${year}*.01_*.codes
done

# echo '#### clean unknown files'
# rm ${output_dir}/${expid}/unknown/*

echo '#### clean config files'
rm ${output_dir}/${expid}/config/*31
rm ${output_dir}/${expid}/config/*/*31

echo '#### clean log files'
rm ${output_dir}/${expid}/log/*31.log
rm ${output_dir}/${expid}/log/*/*31
