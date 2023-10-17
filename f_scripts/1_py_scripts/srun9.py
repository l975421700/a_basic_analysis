

exp_odir = '/albedo/scratch/user/qigao001/output/echam-6.3.05p2-wiso/pi/'
expid = [
    # 'pi_600_5.0',
    # 'pi_601_5.1',
    # 'pi_602_5.2',
    # 'pi_605_5.5',
    # 'pi_606_5.6',
    # 'pi_609_5.7',
    # 'pi_610_5.8',
    # 'hist_700_5.0',
    'nudged_701_5.0',
    ]
i = 0

ifile_start = 12 #0 #120
ifile_end   = 516 #1740 #840

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import pickle
import warnings
warnings.filterwarnings('ignore')
import os
import sys  # print(sys.path)
sys.path.append('/albedo/work/user/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import output

exp_org_o = {}
exp_org_o[expid[i]] = {}

filenames_sf_wiso = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_sf_wiso.nc'))
exp_org_o[expid[i]]['sf_wiso'] = xr.open_mfdataset(
    filenames_sf_wiso[ifile_start:ifile_end],
    )


'''
#-------- check evap
filenames_sf_wiso = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_sf_wiso.nc'))
filenames_echam = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

ifile = 1000
ncfile1 = xr.open_dataset(filenames_sf_wiso[ifile_start:ifile_end][ifile])
ncfile2 = xr.open_dataset(filenames_echam[ifile_start:ifile_end][ifile])

np.max(abs(ncfile1.wisoevap[0, 0] - ncfile2.evap[0])).values

'''
# endregion
# -----------------------------------------------------------------------------


#SBATCH --time=00:30:00
# -----------------------------------------------------------------------------
# region calculate mon/sea/ann evap


wisoevap = {}
wisoevap[expid[i]] = (exp_org_o[expid[i]]['sf_wiso'].wisoevap[:, :3]).copy().compute()

wisoevap[expid[i]] = wisoevap[expid[i]].rename('wisoevap')

wisoevap_alltime = {}
wisoevap_alltime[expid[i]] = mon_sea_ann(var_monthly = wisoevap[expid[i]])

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl',
          'wb') as f:
    pickle.dump(wisoevap_alltime[expid[i]], f)


'''
#-------- check
wisoevap_alltime = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoevap_alltime.pkl', 'rb') as f:
    wisoevap_alltime[expid[i]] = pickle.load(f)

filenames_sf_wiso = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_sf_wiso.nc'))
filenames_echam = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_echam.nc'))

ifile = -2
ncfile1 = xr.open_dataset(filenames_sf_wiso[ifile_start:ifile_end][ifile])
ncfile2 = xr.open_dataset(filenames_echam[ifile_start:ifile_end][ifile])

(wisoevap_alltime[expid[i]]['mon'][ifile, 0] == ncfile2.evap[0]).all().values
(wisoevap_alltime[expid[i]]['mon'][ifile, :] == ncfile1.wisoevap[0, :3]).all().values

'''
# endregion
# -----------------------------------------------------------------------------


