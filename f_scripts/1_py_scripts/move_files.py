
import os
import glob
import shutil

os.chdir('/albedo/work/user/qigao001/output/echam-6.3.05p2-wiso/pi/pi_600_5.0/analysis/echam')

for ifile in glob.glob('*'):
    print(ifile)
    
    shutil.move(ifile, '20y_' + ifile)

