

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = ['pi_m_416_4.9',]
i = 0
print('#-------- ' + expid[i])

ifile_start = 120
ifile_end =   132 #1080

# -----------------------------------------------------------------------------
# region basic settings

ntags = [0, 0, 0, 0, 0,   3, 0, 3, 3, 3,   7, 3, 3, 0]

# ntags = [0, 0, 0, 0, 0,   3, 3, 3, 3, 3,   7]
# ntags = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   7, 3, 3, 0]


var_name  = 'sst'
itag      = 7
min_sf    = 268.15
max_sf    = 318.15

# var_name  = 'lat'
# itag      = 5
# min_sf    = -90
# max_sf    = 90

# var_name  = 'rh2m'
# itag      = 8
# min_sf    = 0
# max_sf    = 1.6

# var_name  = 'wind10'
# itag      = 9
# min_sf    = 0
# max_sf    = 28

# var_name  = 'sinlon'
# itag      = 11
# min_sf    = -1
# max_sf    = 1

# var_name  = 'coslon'
# itag      = 12
# min_sf    = -1
# max_sf    = 1

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import sys  # print(sys.path)
sys.path.append('/work/ollie/qigao001')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": True})
import pickle
from scipy import stats

from a_basic_analysis.b_module.source_properties import (
    source_properties,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann,
)

from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region import data


fl_wiso_q_1m = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_wiso_q_1m.nc'
        ))

exp_out_wiso_q_1m = xr.open_mfdataset(
    fl_wiso_q_1m[ifile_start:ifile_end],
    data_vars='minimal', coords='minimal', parallel=True)


'''
#-------- check with echam output

fl_gl_1m = sorted(glob.glob(
    exp_odir + expid[i] + '/unknown/' + expid[i] + '_??????.01_gl_1m.nc'
        ))

ifile = -1
fl_wiso_q_1m[ifile_start:ifile_end][ifile]
fl_gl_1m[ifile_start:ifile_end][ifile]
ncfile = xr.open_dataset(fl_gl_1m[ifile_start:ifile_end][ifile])

(ncfile.xi.squeeze().values == exp_out_wiso_q_1m.xi16o[ifile, ].values).all()
# np.max(abs(ncfile.xi.squeeze().values - exp_out_wiso_q_1m.xi16o[ifile, ].values))

ncfile2 = xr.open_dataset(fl_wiso_q_1m[ifile_start:ifile_end][ifile])
(ncfile2.xi_24.squeeze().values == exp_out_wiso_q_1m.xi_24[ifile, ].values).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region set indices

kwiso2 = 0

kstart = kwiso2 + sum(ntags[:itag])

str_ind1 = str(kstart + 2)
str_ind2 = str(kstart + 3)

if (len(str_ind1) == 1):
    str_ind1 = '0' + str_ind1
if (len(str_ind2) == 1):
    str_ind2 = '0' + str_ind2

print(str_ind1); print(str_ind2)


'''
exp_out_wiso_q_1m['q_' + str_ind1]


kend   = kwiso2 + sum(ntags[:(itag+1)])
print(kend)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate atmospheric source var

#-------- aggregate atmospheric water
ocean_q = (exp_out_wiso_q_1m['q_' + str_ind1] + \
    exp_out_wiso_q_1m['q_' + str_ind2] + \
        exp_out_wiso_q_1m['xl_' + str_ind1] + \
            exp_out_wiso_q_1m['xl_' + str_ind2] + \
                exp_out_wiso_q_1m['xi_' + str_ind1] + \
                    exp_out_wiso_q_1m['xi_' + str_ind2]
        ).compute()

var_scaled_q = (exp_out_wiso_q_1m['q_' + str_ind1] + \
    exp_out_wiso_q_1m['xl_' + str_ind1] + \
        exp_out_wiso_q_1m['xi_' + str_ind1]
        ).compute()



stats.describe(exp_out_wiso_q_1m['q16o'].values, axis=None)
stats.describe(ocean_q.sel(lev=slice()), axis=None)

# endregion
# -----------------------------------------------------------------------------




