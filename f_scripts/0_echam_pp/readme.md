
File directories (8 files: 642M):

echam:      monthly standdard output.   35M

g3b_1m:     monthly timmean output.     7.2M
            q2m, rh2m, aps, wind10, seaice, relhum, tpot, tsw, temp2

gl_1m:      monthly timmean output.     10M
            q, xl, xi

sf_wiso:    monthly timmean output.     2M
            wisoevap

sp_1m:      monthly timmean output.     2.3M
            st, svo, sd

wiso_q_1m:  monthly timmean output.     278M
            q*, xl*, xi*

wiso_qvi_1d: daily timmean output.      184M
            wisoqvi, wisoxlvi, wisoxivi

wiso:       daily timmean output.       123M
            wisoaprl, wisoaprc



<!-- Template for post processing -->
<!--
echo '#-------- basic settings'

YEAR=$1
output_dir=$2
expid=$3
cd ${WORK}/${output_dir}/${expid}

echo 'settings: ' ${YEAR} ${output_dir} ${expid}


echo '#-------- activate conda env'

source /home/ollie/qigao001/miniconda3/bin/activate training
which cdo
which python


echo '#-------- processing monthly output'

for MONTH in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo ${YEAR} ${MONTH}
done

echo 'job done'
-->
