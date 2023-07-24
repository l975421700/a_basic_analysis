
#-------------------------------- download figures

rsync -avzP xfer1:/home/users/qino/figures /Users/gao/OneDrive\ -\ University\ of\ Cambridge/research/

rsync -avzP albedo:/albedo/work/user/qigao001/figures /Users/gao/OneDrive\ -\ University\ of\ Cambridge/research/


#-------------------------------- sync from albedo to jasmin

rsync -avzLP /albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/* xfer1:/gws/nopw/j04/bas_palaeoclim/qino/output/echam-6.3.05p2-wiso/pi/ &

rsync -avzLP xfer1:/gws/nopw/j04/bas_palaeoclim/qino/output/echam-6.3.05p2-wiso/pi/pi_600_5.0 /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi/ &

# pi_600_5.0; pi_601_5.1; pi_602_5.2; pi_603_5.3; pi_604_5.4

#-------------------------------- backup jasmin to lacie

rsync -avzLP sci3:/gws/nopw/j04/bas_palaeoclim/qino/* /Volumes/LaCie/jasmin/qino/bas_palaeoclim_qino/


#-------------------------------- copy from $SCRATCH to $WORK on albedo

cp -rnL /albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/* /albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi/ &


