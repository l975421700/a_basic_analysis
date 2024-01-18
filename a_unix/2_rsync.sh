
#-------------------------------- download figures

rsync -avzP xfer1:/home/users/qino/figures /Users/gao/OneDrive\ -\ University\ of\ Cambridge/research/

rsync -avzP albedo:/albedo/work/user/qigao001/figures /Users/gao/OneDrive\ -\ University\ of\ Cambridge/research/


#-------------------------------- sync from albedo to jasmin

rsync -avzLP /albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/nudged_711_6.0_S6 xfer1:/gws/nopw/j04/bas_palaeoclim/qino/output/echam-6.3.05p2-wiso/pi/ &
# hist_700_6.0
# nudged_706_6.0_k52_88
# nudged_701_5.0
# nudged_707_6.0_k43
# nudged_702_6.0_spinup
# nudged_708_6.0_I01
# nudged_703_6.0_k52
# nudged_709_6.0_I03
# nudged_704_6.0_spinup_reduced
# nudged_710_6.0_S3
# nudged_705_6.0
# nudged_711_6.0_S6

#-------------------------------- backup jasmin to lacie

rsync -avzLP sci3:/gws/nopw/j04/bas_palaeoclim/qino/* /Volumes/LaCie/jasmin/qino/bas_palaeoclim_qino/



