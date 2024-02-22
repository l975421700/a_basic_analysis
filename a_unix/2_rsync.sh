
#-------------------------------- download figures

rsync -avzP xfer1:/home/users/qino/figures /Users/gao/OneDrive\ -\ University\ of\ Cambridge/research/

rsync -avzP albedo:/albedo/work/user/qigao001/figures /Users/gao/OneDrive\ -\ University\ of\ Cambridge/research/


#-------------------------------- sync from albedo to jasmin

rsync -avLP /albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/hist_700_6.0 xfer1:/home/users/qino/output/echam-6.3.05p2-wiso/pi/ &
# nudged_703_6.0_k52                doing
# nudged_705_6.0                    doing
# nudged_707_6.0_k43                doing
# nudged_708_6.0_I01                doing
# nudged_709_6.0_I03                doing
# nudged_710_6.0_S3                 doing
# nudged_711_6.0_S6                 doing

#-------------------------------- backup jasmin to lacie
rsync -avzLP sci3:/gws/nopw/j04/bas_palaeoclim/qino/* /Volumes/LaCie/jasmin/qino/bas_palaeoclim_qino/


#-------------------------------- copy from scratch to projects

cp -rvuL /albedo/work/user/qigao001/albedo_scratch/output/echam-6.3.05p2-wiso/pi/hist_700_6.0/* /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi/hist_700_6.0/ &
# hist_700_6.0


