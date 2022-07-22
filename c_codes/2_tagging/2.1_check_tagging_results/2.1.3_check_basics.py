

# -----------------------------------------------------------------------------
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import pickle
import os

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})
from scipy import stats
import xesmf as xe
import pandas as pd

# plot
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
plt.rcParams['pcolor.shading'] = 'auto'
mpl.rc('font', family='Times New Roman', size=10)
plt.rcParams.update({"mathtext.fontset": "stix"})
mpl.rcParams['figure.dpi'] = 300

# self defined
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,
    hemisphere_plot,
    rb_colormap,
    quick_var_plot,
    mesh2plot,
)

from a_basic_analysis.b_module.basic_calculations import (
    mon_sea_ann_average,
)

from a_basic_analysis.b_module.namelist import (
    month,
    seasons,
    hours,
    months,
    month_days,
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_403_4.7',
    'pi_m_404_4.7',
    ]

# region import output

exp_org_o = {}

for i in range(len(expid)):
    # i=0
    print('#-------- ' + expid[i])
    exp_org_o[expid[i]] = {}
    
    
    file_exists = os.path.exists(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
    
    if (file_exists):
        exp_org_o[expid[i]]['echam'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc')
        exp_org_o[expid[i]]['wiso'] = xr.open_dataset(
            exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc')
    else:
        filenames_echam = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_echam.nc'))
        filenames_wiso = sorted(glob.glob(exp_odir + expid[i] + '/outdata/echam/' + expid[i] + '*monthly.01_wiso.nc'))
        exp_org_o[expid[i]]['echam'] = xr.open_mfdataset(filenames_echam, data_vars='minimal', coords='minimal', parallel=True)
        exp_org_o[expid[i]]['wiso'] = xr.open_mfdataset(filenames_wiso, data_vars='minimal', coords='minimal', parallel=True)

# endregion
# -----------------------------------------------------------------------------


itag = 13 # 0-13
# ntags = [0, 0, 0, 0, 0,   3, 3, 3, 3, 3,   7]
# ntags = [2, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0]
# ntags = [0, 17, 0, 0, 0,   0, 0, 0, 0, 0,   7]
# ntags = [0, 0, 13, 0, 0,   0, 0, 0, 0, 0,   7]
# ntags = [0, 0, 0, 18, 0,   0, 0, 0, 0, 0,   7]
# ntags = [0, 0, 0, 0, 19,   0, 0, 0, 0, 0,   7]
# ntags = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   7,   3, 3, 0]
ntags = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   7,   0, 0, 37]

# -----------------------------------------------------------------------------
# region set indices for specific set of tracers

kwiso2 = 3

if (itag == 0):
    kstart = kwiso2 + 0
    kend   = kwiso2 + ntags[0]
else:
    kstart = kwiso2 + sum(ntags[:itag])
    kend   = kwiso2 + sum(ntags[:(itag+1)])

print(kstart); print(kend)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check evaporation conservation

i = 0
expid[i]

#---------------- check tagmap conservation
np.max(abs(exp_org_o[expid[i]]['wiso'].tagmap[:, kstart:kend, :, :].sum(axis=1).values - 1))

#---------------- check evap conservation
echam_evap = exp_org_o[expid[i]]['echam'].evap[:, :, :]
wiso_evap = exp_org_o[expid[i]]['wiso'].wisoevap[:, kstart:kend, :, :].sum(axis=1)
diff_evap = echam_evap - wiso_evap
np.max(abs(diff_evap.values))


#---------------- check evap conservation in detail
i1, i2, i3 = np.where(abs(diff_evap) == np.max(abs(diff_evap)))
diff_evap[i1[0], i2[0], i3[0]].values
echam_evap[i1[0], i2[0], i3[0]].values
wiso_evap[i1[0], i2[0], i3[0]].values


'''
# global 1
np.max(abs(exp_org_o[expid[i]]['wiso'].wisoevap[:, 3, :, :] - exp_org_o[expid[i]]['wiso'].wisoevap[:, 0, :, :]))
np.max(abs(exp_org_o[expid[i]]['wiso'].wisoevap[:, 3, :, :] - exp_org_o[expid[i]]['echam'].evap[:, :, :]))
# global 0
np.max(abs(exp_org_o[expid[i]]['wiso'].wisoevap[:, 4, :, :]))
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check humidity conservation - q, xl, xi

#---------------- check humidity coservation

i = 0
expid[i]
nsets = 2

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_sum_humidity.pkl', 'rb') as f:
    sum_humidity = pickle.load(f)

diff_q = sum_humidity['q'] - nsets * exp_org_o[expid[i]]['wiso'].q16o
# diff_q.values[exp_org_o[expid[i]]['wiso'].q16o.values < 0] = np.nan
np.max(abs(diff_q[:, :, :, :]))
# diff_q.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_diff_q.nc')

diff_xl = sum_humidity['xl'] - nsets * exp_org_o[expid[i]]['wiso'].xl16o
# diff_xl.values[exp_org_o[expid[i]]['wiso'].xl16o.values < 0] = np.nan
np.max(abs(diff_xl[:, :, :, :]))
# diff_xl.to_netcdf('scratch/test/diff_xl.nc')

diff_xi = sum_humidity['xi'] - nsets * exp_org_o[expid[i]]['wiso'].xi16o
# diff_xi.values[exp_org_o[expid[i]]['wiso'].xi16o.values < 0] = np.nan
np.max(abs(diff_xi[:, :, :, :]))
# diff_xi.to_netcdf('scratch/test/diff_xi.nc')


#---------------- check humidity coservation in detail
i1, i2, i3, i4 = np.where(abs(diff_q[:, :, :, :]) == np.max(abs(diff_q[:, :, :, :])))

np.max(abs(diff_q[:, :, :, :]))
diff_q[i1[0], i2[0], i3[0], i4[0]].values
sum_humidity['q'][i1[0], i2[0], i3[0], i4[0]].values
nsets * exp_org_o[expid[i]]['wiso'].q16o[i1[0], i2[0], i3[0], i4[0]].values


i1, i2, i3, i4 = np.where(abs(diff_xl[:, :, :, :]) == np.max(abs(diff_xl[:, :, :, :])))

np.max(abs(diff_xl[:, :, :, :]))
diff_xl[i1[0], i2[0], i3[0], i4[0]].values
sum_humidity['xl'][i1[0], i2[0], i3[0], i4[0]].values
nsets * exp_org_o[expid[i]]['wiso'].xl16o[i1[0], i2[0], i3[0], i4[0]].values

i1, i2, i3, i4 = np.where(abs(diff_xi[:, :, :, :]) == np.max(abs(diff_xi[:, :, :, :])))

np.max(abs(diff_xi[:, :, :, :]))
diff_xi[i1[0], i2[0], i3[0], i4[0]].values
sum_humidity['xi'][i1[0], i2[0], i3[0], i4[0]].values
nsets * exp_org_o[expid[i]]['wiso'].xi16o[i1[0], i2[0], i3[0], i4[0]].values


'''
#---- check negative q

neg_q_01 = exp_org_o[expid[i]]['wiso'].q_01.values[exp_org_o[expid[i]]['wiso'].q_01 < 0]
neg_q16o = exp_org_o[expid[i]]['wiso'].q16o.values[exp_org_o[expid[i]]['wiso'].q16o < 0]

test = exp_org_o[expid[i]]['wiso'].q_01
test.values[test > -1e-18] = 0
test.to_netcdf('scratch/test/test.nc')
(test < -1e-15).sum()

#---------------- check global 0 works as expected

np.max(abs(exp_org_o[expid[i]]['wiso'].q_02))
np.max(abs(exp_org_o[expid[i]]['wiso'].xl_02))
np.max(abs(exp_org_o[expid[i]]['wiso'].xi_02))

#---------------- check global 1

(exp_org_o[expid[i]]['echam'].q == exp_org_o[expid[i]]['wiso'].q_01).all()
(exp_org_o[expid[i]]['echam'].xl == exp_org_o[expid[i]]['wiso'].xl_01).all()
(exp_org_o[expid[i]]['echam'].xi == exp_org_o[expid[i]]['wiso'].xi_01).all()

np.max(abs(exp_org_o[expid[i]]['echam'].q - exp_org_o[expid[i]]['wiso'].q_01))
np.max(abs(exp_org_o[expid[i]]['echam'].xl - exp_org_o[expid[i]]['wiso'].xl_01))
np.max(abs(exp_org_o[expid[i]]['echam'].xi - exp_org_o[expid[i]]['wiso'].xi_01))

diff_xl = exp_org_o[expid[i]]['echam'].xl - exp_org_o[expid[i]]['wiso'].xl_01
where_max_diff_xl = np.where(abs(diff_xl) == np.max(abs(diff_xl)))
diff_xl.values[where_max_diff_xl]
exp_org_o[expid[i]]['echam'].xl.values[where_max_diff_xl]
exp_org_o[expid[i]]['wiso'].xl_01.values[where_max_diff_xl]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check precipitation conservation


#-------------------------------- check precipitation conservation

i = 0
expid[i]

echam_apr = exp_org_o[expid[i]]['echam'].aprl + exp_org_o[expid[i]]['echam'].aprc
wiso_apr = exp_org_o[expid[i]]['wiso'].wisoaprl[:, kstart:kend].sum(axis=1) + exp_org_o[expid[i]]['wiso'].wisoaprc[:, kstart:kend].sum(axis=1)
diff_apr = wiso_apr - echam_apr
# diff_apr.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.diff_apr.nc')
np.max(abs(diff_apr))
wheremax = np.where(abs(diff_apr) == np.max(abs(diff_apr)))
diff_apr.values[wheremax]
wiso_apr.values[wheremax]
echam_apr.values[wheremax]


rel_diff_apr = diff_apr / echam_apr
rel_diff_apr.values[echam_apr.values < 1e-9] = np.nan
# rel_diff_apr.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.rel_diff_apr.nc')
stats.describe(rel_diff_apr, axis=None, nan_policy='omit')

wheremax = np.where(abs(rel_diff_apr) == np.max(abs(rel_diff_apr)))
rel_diff_apr.values[wheremax]
diff_apr.values[wheremax]
wiso_apr.values[wheremax]
echam_apr.values[wheremax]


#-------------------------------- check l. scale & conv. precipitation

echam_aprl = exp_org_o[expid[i]]['echam'].aprl
wiso_aprl = exp_org_o[expid[i]]['wiso'].wisoaprl[:, kstart:kend].sum(axis=1)
diff_aprl = wiso_aprl - echam_aprl
stats.describe(abs(diff_aprl), axis=None)

echam_aprc = exp_org_o[expid[i]]['echam'].aprc
wiso_aprc = exp_org_o[expid[i]]['wiso'].wisoaprc[:, kstart:kend].sum(axis=1)
diff_aprc = wiso_aprc - echam_aprc
where_max_diff_aprc = np.where(abs(diff_aprc) == np.max(abs(diff_aprc)))
diff_aprc.values[where_max_diff_aprc]
echam_aprc.values[where_max_diff_aprc]
wiso_aprc.values[where_max_diff_aprc]
stats.describe(abs(diff_aprc), axis=None)

echam_aprs = exp_org_o[expid[i]]['echam'].aprs
wiso_aprs = exp_org_o[expid[i]]['wiso'].wisoaprs[:, kstart:kend].sum(axis=1)
diff_aprs = wiso_aprs - echam_aprs
stats.describe(abs(diff_aprs), axis=None)


#-------------------------------- plot tracer precipitation percentage deviation

pltlevel = np.arange(-4, 4.001, 0.01)
pltticks = np.arange(-4, 4.001, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    exp_org_o[expid[i]]['echam'].lon,
    exp_org_o[expid[i]]['echam'].lat,
    rel_diff_apr[-1, :, :] * 10**3,
    # abs(rel_diff_apr[:, :, :]).mean(axis=0) * 10**2,
    # rel_diff_apr_mean * 10**3,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Differences between tracer and normal precipitation, as fraction of normal precipitation [$10^{-3}$]\n' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(exp_odir+expid[i]+'/viz/echam/'+expid[i]+'.rel_diff_apr.png')


#-------------------------------- plot tracer precipitation deviation

pltlevel = np.arange(-2, 2.001, 0.01)
pltticks = np.arange(-2, 2.001, 0.4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PRGn', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    exp_org_o[expid[i]]['echam'].lon,
    exp_org_o[expid[i]]['echam'].lat,
    diff_apr[-1, :, :] * 10**8,
    # diff_apr_mean * 10**7,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Differences between tracer and normal precipitation [$10^{-8} \; mm \; s^{-1}$]\n' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(exp_odir+expid[i]+'/viz/echam/'+expid[i]+'.diff_apr.png')



'''
#---------------- global 0 works as expected

aprl_global0 = exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4, :, :]
np.max(abs(aprl_global0))
aprc_global0 = exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4, :, :]
np.max(abs(aprc_global0))

#---------------- global 1

echam_apr_global1 = exp_org_o[expid[i]]['echam'].aprl[:, :, :] + exp_org_o[expid[i]]['echam'].aprc[:, :, :]
wiso_apr_global1 = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 3, :, :])
diff_apr_global1 = wiso_apr_global1 - echam_apr_global1
stats.describe(abs(diff_apr_global1), axis=None)

where_max = np.where(abs(diff_apr_global1) == np.max(abs(diff_apr_global1)))
diff_apr_global1.values[where_max]
echam_apr_global1.values[where_max]
wiso_apr_global1.values[where_max]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check negative evaporation with tagmap

i = 0
expid[i]

#---- over water
post_wisoevap = exp_org_o[expid[i]]['echam'].evapwac.values[1:, None, :, :] * exp_org_o[expid[i]]['wiso'].tagmap[:-1, 3:, :, :]
diff_wisoevap = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevapwac[1:, 3:, :, :].values
diff_wisoevap.values[np.where(exp_org_o[expid[i]]['wiso'].wisoevapwac[1:, 3:, :, :].values >= 0)] = 0
# stats.describe(diff_wisoevap[:, :, :, :], axis=None)
np.max(abs(diff_wisoevap))

wheremax = np.where(abs(diff_wisoevap) == np.max(abs(diff_wisoevap)))
diff_wisoevap.values[wheremax]
post_wisoevap.values[wheremax]
exp_org_o[expid[i]]['wiso'].wisoevapwac[1:, 3:, :, :].values[wheremax]

#---- over land
post_wisoevap = exp_org_o[expid[i]]['echam'].evaplac.values[1:, None, :, :] * exp_org_o[expid[i]]['wiso'].tagmap[:-1, 3:, :, :]
diff_wisoevap = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevaplac[1:, 3:, :, :].values
diff_wisoevap.values[np.where(exp_org_o[expid[i]]['wiso'].wisoevaplac[1:, 3:, :, :].values >= 0)] = 0
# stats.describe(diff_wisoevap[:, :, :, :], axis=None)
np.max(abs(diff_wisoevap[:, :, :, :]))

wheremax = np.where(abs(diff_wisoevap) == np.max(abs(diff_wisoevap)))
diff_wisoevap.values[wheremax]
post_wisoevap.values[wheremax]
exp_org_o[expid[i]]['wiso'].wisoevaplac[1:, 3:, :, :].values[wheremax]


#---- over ice
post_wisoevap = exp_org_o[expid[i]]['echam'].evapiac.values[1:, None, :, :] * exp_org_o[expid[i]]['wiso'].tagmap[:-1, 3:, :, :]
diff_wisoevap = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevapiac[1:, 3:, :, :].values
diff_wisoevap.values[np.where(exp_org_o[expid[i]]['wiso'].wisoevapiac[1:, 3:, :, :].values >= 0)] = 0
# stats.describe(diff_wisoevap[:, :, :, :], axis=None)
np.max(abs(diff_wisoevap[:, :, :, :]))

wheremax = np.where(abs(diff_wisoevap) == np.max(abs(diff_wisoevap)))
diff_wisoevap.values[wheremax]
post_wisoevap.values[wheremax]
exp_org_o[expid[i]]['wiso'].wisoevapiac[1:, 3:, :, :].values[wheremax]



'''
#---- overall
post_wisoevap = exp_org_o[expid[i]]['echam'].evap.values[1:, None, :, :] * exp_org_o[expid[i]]['wiso'].tagmap[:-1, 3:, :, :]
test = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevap[1:, 3:, :, :].values
test.values[np.where(exp_org_o[expid[i]]['wiso'].wisoevap[1:, 3:, :, :].values >= 0)] = 0

stats.describe(test[:, :, :, :], axis=None)
(test == 0).sum() # 97.88%
(test < 1e-10).sum() # 99.80%
# test.to_netcdf('scratch/test/test.nc')


np.where(test[0, :, :, :] == np.max(test[0, :, :, :]))
test[0, 1, 89, 174]

# evap
exp_org_o[expid[i]]['echam'].evap[1, 89, 174].values
exp_org_o[expid[i]]['echam'].evapiac[1, 89, 174].values + \
exp_org_o[expid[i]]['echam'].evaplac[1, 89, 174].values + \
exp_org_o[expid[i]]['echam'].evapwac[1, 89, 174].values

# wisoevap
exp_org_o[expid[i]]['wiso'].wisoevap[1, 4, 89, 174].values
exp_org_o[expid[i]]['wiso'].wisoevapiac[1, 4, 89, 174].values + \
exp_org_o[expid[i]]['wiso'].wisoevaplac[1, 4, 89, 174].values + \
exp_org_o[expid[i]]['wiso'].wisoevapwac[1, 4, 89, 174].values

exp_org_o[expid[i]]['wiso'].wisoevap[1, 3, 89, 174].values
exp_org_o[expid[i]]['wiso'].wisoevapiac[1, 3, 89, 174].values + \
exp_org_o[expid[i]]['wiso'].wisoevaplac[1, 3, 89, 174].values + \
exp_org_o[expid[i]]['wiso'].wisoevapwac[1, 3, 89, 174].values

# ztagfac
exp_org_o[expid[i]]['wiso'].ztag_fac_ice[1, 4, 89, 174].values
exp_org_o[expid[i]]['wiso'].ztag_fac_water[1, 4, 89, 174].values
exp_org_o[expid[i]]['wiso'].ztag_fac_land[1, 4, 89, 174].values

exp_org_o[expid[i]]['wiso'].ztag_fac_ice[1, 3, 89, 174].values
exp_org_o[expid[i]]['wiso'].ztag_fac_water[1, 3, 89, 174].values
exp_org_o[expid[i]]['wiso'].ztag_fac_land[1, 3, 89, 174].values


# post_wisoevap
post_wisoevap[0, 1, 89, 174].values


#---- over ice
exp_org_o[expid[i]]['echam'].evapiac[1, 89, 174].values * exp_org_o[expid[i]]['wiso'].ztag_fac_ice[1, 4, 89, 174].values
exp_org_o[expid[i]]['wiso'].wisoevapiac[1, 4, 89, 174].values

# more check
exp_org_o[expid[i]]['echam'].evapiac[1, 89, 174].values

exp_org_o[expid[i]]['wiso'].tagmap[0, 3, 89, 174].values
exp_org_o[expid[i]]['wiso'].tagmap[0, 4, 89, 174].values

exp_org_o[expid[i]]['wiso'].ztag_fac_ice[1, 3, 89, 174].values
exp_org_o[expid[i]]['wiso'].ztag_fac_ice[1, 4, 89, 174].values

exp_org_o[expid[i]]['wiso'].wisoevapiac[1, 3, 89, 174].values
exp_org_o[expid[i]]['wiso'].wisoevapiac[1, 4, 89, 174].values


#---- over land
exp_org_o[expid[i]]['echam'].evaplac[1, 89, 174].values * exp_org_o[expid[i]]['wiso'].ztag_fac_land[1, 4, 89, 174].values
exp_org_o[expid[i]]['wiso'].wisoevaplac[1, 4, 89, 174].values

exp_org_o[expid[i]]['echam'].evaplac[1, 89, 174].values * exp_org_o[expid[i]]['wiso'].ztag_fac_land[1, 3, 89, 174].values
exp_org_o[expid[i]]['wiso'].wisoevaplac[1, 3, 89, 174].values

exp_org_o[expid[i]]['echam'].slf[1, 89, 174]



# more check
exp_org_o[expid[i]]['echam'].evaplac[1, 89, 174].values # -1.86264515e-09

exp_org_o[expid[i]]['wiso'].tagmap[0, 3, 89, 174].values # 0
exp_org_o[expid[i]]['wiso'].tagmap[0, 4, 89, 174].values # 1

exp_org_o[expid[i]]['wiso'].ztag_fac_land[1, 3, 89, 174].values # 0
exp_org_o[expid[i]]['wiso'].ztag_fac_land[1, 4, 89, 174].values # 0

exp_org_o[expid[i]]['wiso'].wisoevaplac[1, 0, 89, 174].values # -1.86264515e-09
exp_org_o[expid[i]]['wiso'].wisoevaplac[1, 3, 89, 174].values # -1.86264515e-09
exp_org_o[expid[i]]['wiso'].wisoevaplac[1, 4, 89, 174].values # 1.36788003e-09


#---- over ocean
exp_org_o[expid[i]]['echam'].evapwac[1, 89, 174].values * exp_org_o[expid[i]]['wiso'].ztag_fac_water[1, 4, 89, 174].values
exp_org_o[expid[i]]['wiso'].wisoevapwac[1, 4, 89, 174].values

# more check
exp_org_o[expid[i]]['echam'].evapwac[1, 89, 174].values

exp_org_o[expid[i]]['wiso'].tagmap[0, 3, 89, 174].values
exp_org_o[expid[i]]['wiso'].tagmap[0, 4, 89, 174].values

exp_org_o[expid[i]]['wiso'].ztag_fac_water[1, 3, 89, 174].values
exp_org_o[expid[i]]['wiso'].ztag_fac_water[1, 4, 89, 174].values

exp_org_o[expid[i]]['wiso'].wisoevapwac[1, 3, 89, 174].values
exp_org_o[expid[i]]['wiso'].wisoevapwac[1, 4, 89, 174].values



'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check all evaporation with ztag_fac_*

i = 0
expid[i]


#---- over water
post_wisoevap = exp_org_o[expid[i]]['echam'].evapwac.values[:, None, :, :] * exp_org_o[expid[i]]['wiso'].ztag_fac_water[:, 3:, :, :]
diff_wisoevap = post_wisoevap[1:] - exp_org_o[expid[i]]['wiso'].wisoevapwac[1:, 3:, :, :].values
np.max(abs(diff_wisoevap))

wheremax = np.where(abs(diff_wisoevap) == np.max(abs(diff_wisoevap)))
diff_wisoevap.values[wheremax]
post_wisoevap.values[1:][wheremax]
exp_org_o[expid[i]]['wiso'].wisoevapwac[1:, 3:, :, :].values[wheremax]


#---- over land
post_wisoevap = exp_org_o[expid[i]]['echam'].evaplac.values[:, None, :, :] * exp_org_o[expid[i]]['wiso'].ztag_fac_land[:, 3:, :, :]
diff_wisoevap = post_wisoevap[1:] - exp_org_o[expid[i]]['wiso'].wisoevaplac[1:, 3:, :, :].values
np.max(abs(diff_wisoevap))

wheremax = np.where(abs(diff_wisoevap) == np.max(abs(diff_wisoevap)))
diff_wisoevap.values[wheremax]
post_wisoevap.values[1:][wheremax]
exp_org_o[expid[i]]['wiso'].wisoevaplac[1:, 3:, :, :].values[wheremax]


#---- over ice
post_wisoevap = exp_org_o[expid[i]]['echam'].evapiac.values[:, None, :, :] * exp_org_o[expid[i]]['wiso'].ztag_fac_ice[:, 3:, :, :]
diff_wisoevap = post_wisoevap[1:] - exp_org_o[expid[i]]['wiso'].wisoevapiac[1:, 3:, :, :].values
np.max(abs(diff_wisoevap))

wheremax = np.where(abs(diff_wisoevap) == np.max(abs(diff_wisoevap)))
diff_wisoevap.values[wheremax]
post_wisoevap.values[1:][wheremax]
exp_org_o[expid[i]]['wiso'].wisoevapiac[1:, 3:, :, :].values[wheremax]


'''
#---- check over water
tagmap = xr.open_dataset('startdump/tagging/tagmap/tagmap_ls_15_0.nc')
post_wisoevap = exp_org_o[expid[i]]['echam'].evapwac.values[:, None, :, :] * exp_org_o[expid[i]]['wiso'].ztag_fac_water[:, 3:, :, :]
diff_wisoevap = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevapwac[:, 3:, :, :].values
# diff_wisoevap.to_netcdf('scratch/test/test.nc')

stats.describe(diff_wisoevap[1:, :, :, :], axis=None)
np.max(abs(diff_wisoevap[1:, :, :, :]))

np.max(abs(diff_wisoevap[0, :, :, :]))
np.where(diff_wisoevap[0, :, :, :] == np.max(diff_wisoevap[0, :, :, :]))

diff_wisoevap[0, 1, 27, 77]

exp_org_o[expid[i]]['echam'].evapwac[0, 27, 77].values

tagmap.tagmap[3, 27, 77].values
tagmap.tagmap[4, 27, 77].values

exp_org_o[expid[i]]['wiso'].ztag_fac_water[0, 3, 27, 77].values
exp_org_o[expid[i]]['wiso'].ztag_fac_water[0, 4, 27, 77].values

exp_org_o[expid[i]]['wiso'].wisoevapwac[0, 4, 27, 77].values
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check Martin's corrections - shorter version

# expid = ['pi_echam6_1d_154_3.59',]
i = 0
expid[i]
exp_odir = 'output/echam-6.3.05p2-wiso/pi/'

cor_prefix = ['phy_tagqte_c_', 'phy_tagxlte_c_', 'phy_tagxite_c_']
org_prefix = ['q_', 'xl_', 'xi_']
level = [47, 41, 34]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_org_var.pkl',
          'rb') as f:
    org_var = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_cor_var.pkl',
          'rb') as f:
    cor_var = pickle.load(f)

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_rel_corr.pkl',
          'rb') as f:
    rel_corr = pickle.load(f)

l = 0
i1, i2, i3, i4 = np.where(abs(cor_var[org_prefix[l][:-1]][:, :, 0, :, :]) == np.nanmax(abs(cor_var[org_prefix[l][:-1]][:, :, 0, :, :])))
org_var[org_prefix[l][:-1]][i1[0], i2[0], 0, i3[0], i4[0]]
cor_var[org_prefix[l][:-1]][i1[0], i2[0], 0, i3[0], i4[0]] * 900
cor_var[org_prefix[l][:-1]][i1[0], i2[0] + 1, 0, i3[0], i4[0]] * 900

# loop along q/xl/xi
for l in range(len(cor_prefix)):
    # l = len(cor_prefix) - 1
    print('#----------------------------------------------------------------')
    print('#-------------------------------- ' + cor_prefix[l])

    # If a correction is positive and the original variable is still not positive, it will not be considered.
    where_neg_org = np.where((cor_var[org_prefix[l][:-1]] > 0) & (org_var[org_prefix[l][:-1]]<=1e-20))
    cor_var[org_prefix[l][:-1]][where_neg_org] = np.nan
    org_var[org_prefix[l][:-1]][where_neg_org] = np.nan
    
    # Do not account for grid cells where absolute corrections is less than 1e-18.
    # where_small_cor = np.where(abs(cor_var[org_prefix[l][:-1]]) < 1e-18)
    org_var[org_prefix[l][:-1]][
        abs(cor_var[org_prefix[l][:-1]]) < 1e-18] = np.nan
    cor_var[org_prefix[l][:-1]][
        abs(cor_var[org_prefix[l][:-1]]) < 1e-18] = np.nan
    
    print('#---- Mean original fields')
    print(np.nanmean(abs(org_var[org_prefix[l][:-1]])))
    print('#---- Mean corrections')
    print(np.nanmean(abs(cor_var[org_prefix[l][:-1]])) * 900)
    
    where_max_corr = np.where(abs(cor_var[org_prefix[l][:-1]]) == np.nanmax(abs(cor_var[org_prefix[l][:-1]])))
    print('#---- Maximum corrections')
    print(cor_var[org_prefix[l][:-1]][where_max_corr] * 900)
    print('#---- Original values at maximum corrections')
    print(org_var[org_prefix[l][:-1]][where_max_corr])
    
    rel_corr[org_prefix[l][:-1]] = abs(cor_var[org_prefix[l][:-1]] * 900 / org_var[org_prefix[l][:-1]])
    
    # If absolute original variable is less than 1e-18, relative corrections will not be considered.
    rel_corr[org_prefix[l][:-1]][abs(org_var[org_prefix[l][:-1]]) < 1e-15] = np.nan
    print('#---- Mean relative corrections')
    print(np.nanmean(rel_corr[org_prefix[l][:-1]]))
    print('#---- Maximum relative corrections')
    print(np.nanmax(rel_corr[org_prefix[l][:-1]]))



#---------------- detailed checks

l = 0
var_names = np.array([var for var in exp_org_o[expid[i]]['wiso'].data_vars])
org_names = var_names[[var.startswith(org_prefix[l]) for var in var_names]]
cor_names = var_names[[var.startswith(cor_prefix[l]) for var in var_names]]

where_max_rel = np.where(rel_corr[org_prefix[l][:-1]] == np.nanmax(rel_corr[org_prefix[l][:-1]]))
rel_corr[org_prefix[l][:-1]][where_max_rel]
org_var[org_prefix[l][:-1]][where_max_rel]
cor_var[org_prefix[l][:-1]][where_max_rel] * 900
ij, ik, i3, i4, i5 = where_max_rel
exp_org_o[expid[i]]['wiso'][org_names[ik[0]]][i3[0], level[ij[0]]-1, i4[0], i5[0]].values
exp_org_o[expid[i]]['wiso'][cor_names[ij[0]]][i3[0], 3+ik[0], i4[0], i5[0]].values * 900

qm1 = [
    exp_org_o[expid[i]]['wiso'][org_names[0]][i3[0]-1, level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[1]][i3[0]-1, level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[2]][i3[0]-1, level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[3]][i3[0]-1, level[ij[0]]-1, i4[0], i5[0]].values,
]
q = [
    exp_org_o[expid[i]]['wiso'][org_names[0]][i3[0], level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[1]][i3[0], level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[2]][i3[0], level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[3]][i3[0], level[ij[0]]-1, i4[0], i5[0]].values,
]

lname = 'phy_tagqte_34'
org_qte = exp_org_o[expid[i]]['wiso'][lname][i3[0], 3:, i4[0], i5[0]].values * 900
cor_qte = exp_org_o[expid[i]]['wiso'][cor_names[ij[0]]][i3[0], 3:, i4[0], i5[0]].values * 900
final_qte = org_qte + cor_qte

org_q = qm1 + org_qte
cor_q = qm1 + final_qte

exp_org_o[expid[i]]['wiso'].q16o[i3[0]-1, level[ij[0]]-1, i4[0], i5[0]].values
exp_org_o[expid[i]]['wiso'].q16o[i3[0], level[ij[0]]-1, i4[0], i5[0]].values


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check Martin's corrections - plot

i = 0
expid[i]

cor_prefix = ['phy_tagqte_c_', 'phy_tagxlte_c_', 'phy_tagxite_c_']
org_prefix = ['q_', 'xl_', 'xi_']
level = [47, 41, 34]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_org_var.pkl',
          'rb') as f:
    org_var = pickle.load(f)
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_cor_var.pkl',
          'rb') as f:
    cor_var = pickle.load(f)
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_rel_corr.pkl',
          'rb') as f:
    rel_corr = pickle.load(f)

# loop along q/xl/xi
for l in range(len(cor_prefix)):
    # l = len(cor_prefix) - 1
    print('#----------------------------------------------------------------')
    print('#-------------------------------- ' + cor_prefix[l])

    # If a correction is positive and the original variable is still not positive, it will not be considered.
    where_neg_org = np.where((cor_var[org_prefix[l][:-1]] > 0) & (org_var[org_prefix[l][:-1]]<=1e-20))
    cor_var[org_prefix[l][:-1]][where_neg_org] = np.nan
    org_var[org_prefix[l][:-1]][where_neg_org] = np.nan
    
    # Do not account for grid cells where absolute corrections is less than 1e-18.
    org_var[org_prefix[l][:-1]][
        abs(cor_var[org_prefix[l][:-1]]) < 1e-18] = np.nan
    cor_var[org_prefix[l][:-1]][
        abs(cor_var[org_prefix[l][:-1]]) < 1e-18] = np.nan
    
    rel_corr[org_prefix[l][:-1]] = abs(cor_var[org_prefix[l][:-1]] * 900 / org_var[org_prefix[l][:-1]])
    
    # If absolute original variable is less than 1e-15, relative corrections will not be considered.
    rel_corr[org_prefix[l][:-1]][abs(org_var[org_prefix[l][:-1]]) < 1e-15] = np.nan



np.nanmean(abs(org_var['q']))
np.nanmean(abs(cor_var['q']) * 900)


stats.describe(np.nanmean(abs(org_var['q']), axis=(0, 1, 2)) * 10**3, axis=None, nan_policy='omit')


pltlevel = np.arange(0, 16.001, 0.001)
pltticks = np.arange(0, 16.001, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Blues', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    exp_org_o[expid[i]]["wiso"].lon,
    exp_org_o[expid[i]]["wiso"].lat,
    np.nanmean(abs(org_var['q']), axis=(0, 1, 2)) * 10**3,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax,
    orientation="horizontal", pad=0.02, fraction=0.14, shrink=0.6,
    aspect=40, anchor=(0.5, 0.7), ticks=pltticks, extend="max",)

cbar.ax.set_xlabel(
    u'q [$10^{-3} \; kg \; kg^{-1}$] with corrections in ' + expid[i] + '\nAveraged over three vertical levels, among all water tracers, and all time steps',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig( exp_odir + expid[i] + '/viz/echam/' + expid[i] + '_org_var_q.png')


stats.describe(np.nanmean(abs(cor_var['q']) * 900, axis=(0, 1, 2)) * 10**6, axis=None, nan_policy='omit')

pltlevel = np.arange(0, 12.001, 0.001)
pltticks = np.arange(0, 12.001, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Purples', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    exp_org_o[expid[i]]["wiso"].lon,
    exp_org_o[expid[i]]["wiso"].lat,
    np.nanmean(abs(cor_var['q']) * 900, axis=(0, 1, 2)) * 10**6,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax,
    orientation="horizontal", pad=0.02, fraction=0.14, shrink=0.6,
    aspect=40, anchor=(0.5, 0.7), ticks=pltticks, extend="max",)

cbar.ax.set_xlabel(
    u'Corrections to q [$10^{-6} \; kg \; kg^{-1}$] in ' + expid[i] + '\nAveraged over three vertical levels, among all water tracers, and all time steps',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig( exp_odir + expid[i] + '/viz/echam/' + expid[i] + '_org_var_corrections to q.png')


stats.describe(np.nanmean(abs(rel_corr['q']), axis=(0, 1, 2)) * 10**3, axis=None, nan_policy='omit')


pltlevel = np.arange(0, 30.001, 0.001)
pltticks = np.arange(0, 30.001, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('Reds', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    exp_org_o[expid[i]]["wiso"].lon,
    exp_org_o[expid[i]]["wiso"].lat,
    np.nanmean(abs(rel_corr['q']), axis=(0, 1, 2)) * 10**3,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax,
    orientation="horizontal", pad=0.02, fraction=0.14, shrink=0.6,
    aspect=40, anchor=(0.5, 0.7), ticks=pltticks, extend="max",)

cbar.ax.set_xlabel(
    u'Relative corrections to q [$10^{-3}$] in ' + expid[i] + '\nAveraged over three vertical levels, among all water tracers, and all time steps',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig( exp_odir + expid[i] + '/viz/echam/' + expid[i] + '_org_var_relative corrections to q.png')


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check Martin's corrections preliminary analysis

#---------------- comparing tendencies with corrections

org_fields = [
    'phy_tagqte_41',
    # 'phy_tagqte_47', 'phy_tagqte_41', 'phy_tagqte_34',
    # 'phy_tagxlte_47', 'phy_tagxlte_41', 'phy_tagxlte_34',
    # 'phy_tagxite_47', 'phy_tagxite_41', 'phy_tagxite_34',
    ]
cor_fields = [
    'phy_tagqte_c_41',
    # 'phy_tagqte_c_47', 'phy_tagqte_c_41', 'phy_tagqte_c_34',
    # 'phy_tagxlte_c_47', 'phy_tagxlte_c_41', 'phy_tagxlte_c_34',
    # 'phy_tagxite_c_47', 'phy_tagxite_c_41', 'phy_tagxite_c_34',
    ]

i = 0
expid[i]

rel_corr = {}

# 1, Mean absolute corrections should be several magnitudes smaller than mean absolute values.

for j in range(len(org_fields)):
    # j = 0
    print('#----------------------------------------------------------------')
    print('#---------------- ' + org_fields[j] + ' and ' + cor_fields[j])

    print('#---- Mean tendencies')
    print(np.mean(abs(exp_org_o[expid[i]]['wiso'][org_fields[j]][:, 3:, :, :])).values)

    print('#---- Mean correction')
    print(np.mean(abs(exp_org_o[expid[i]]['wiso'][cor_fields[j]][:, 3:, :, :])).values)
    
    where_max_corr = np.where(abs(exp_org_o[expid[i]]['wiso'][cor_fields[j]][:, 3:, :, :]) == np.max(abs(exp_org_o[expid[i]]['wiso'][cor_fields[j]][:, 3:, :, :])))
    
    print('#---- Maximum correction')
    print(exp_org_o[expid[i]]['wiso'][cor_fields[j]][:, 3:, :, :].values[where_max_corr])
    
    print('#---- Original values at maximum correction')
    print(exp_org_o[expid[i]]['wiso'][org_fields[j]][:, 3:, :, :].values[where_max_corr])


j = 0
i1, i2, i3, i4 = np.where(abs(exp_org_o[expid[i]]['wiso'][cor_fields[j]][:, 3:, :, :]) == np.max(abs(exp_org_o[expid[i]]['wiso'][cor_fields[j]][:, 3:, :, :])))

# i1, i2, i3, i4 = [[40], [30], [40], [50]]
org_qte = exp_org_o[expid[i]]['wiso'][org_fields[j]][i1[0], 3:, i3[0], i4[0]].values
cor_qte = exp_org_o[expid[i]]['wiso'][cor_fields[j]][i1[0], 3:, i3[0], i4[0]].values
final_qte = org_qte + cor_qte

qm1 = np.array([
    exp_org_o[expid[i]]['wiso'].q_01[i1[0] - 1, 40, i3[0], i4[0]].values,
    exp_org_o[expid[i]]['wiso'].q_02[i1[0] - 1, 40, i3[0], i4[0]].values,
    exp_org_o[expid[i]]['wiso'].q_03[i1[0] - 1, 40, i3[0], i4[0]].values,
    exp_org_o[expid[i]]['wiso'].q_04[i1[0] - 1, 40, i3[0], i4[0]].values])
q = np.array([
    exp_org_o[expid[i]]['wiso'].q_01[i1[0], 40, i3[0], i4[0]].values,
    exp_org_o[expid[i]]['wiso'].q_02[i1[0], 40, i3[0], i4[0]].values,
    exp_org_o[expid[i]]['wiso'].q_03[i1[0], 40, i3[0], i4[0]].values,
    exp_org_o[expid[i]]['wiso'].q_04[i1[0], 40, i3[0], i4[0]].values])

org_q = qm1 + 900*org_qte
cor_q = qm1 + 900*final_qte

cor_qte*900 / q
cor_qte*900 / org_q


#---------------- comparing actual values with corrections in tendencies

i = 0
expid[i]

cor_fields = [
    'phy_tagqte_c_41',
    ]
org_fields = [
    'q_01', 'q_02', 'q_03', 'q_04',
    ]
level = [47,]
rel_corr = {}

j = 0
cor_fields[j]
rel_corr[cor_fields[j]] = {}
level[j]
for k in range(4):
    print('#----------------------------------------------------------------')
    # k = 1
    org_var = exp_org_o[expid[i]]['wiso'][org_fields[4*j + k]][:, level[j]-1, :, :]
    cor_var = exp_org_o[expid[i]]['wiso'][cor_fields[j]][:, 3+k, :, :]
    
    # if a correction is positive and org_var is 0, it's correct
    where_neg_org = np.where((cor_var > 0) & (org_var<=0))
    cor_var.values[where_neg_org] = np.nan
    org_var.values[where_neg_org] = np.nan
    
    # do not account for grids where there is no corrections
    where_small_cor = np.where(abs(cor_var) < 1e-18)
    cor_var.values[where_small_cor] = np.nan
    org_var.values[where_small_cor] = np.nan
    
    print('#---- Mean original fields')
    print(np.mean(abs(org_var)).values)
    print('#---- Mean corrections')
    print(np.mean(abs(cor_var)).values * 900)
    
    where_max_corr = np.where(abs(cor_var) == np.max(abs(cor_var)))
    print('#---- Maximum corrections')
    print(cor_var.values[where_max_corr] * 900)
    print('#---- Original values at maximum corrections')
    print(org_var.values[where_max_corr])
    
    rel_corr[cor_fields[j]][org_fields[4*j + k]] = abs(cor_var * 900) / abs(org_var)
    
    rel_corr[cor_fields[j]][org_fields[4*j + k]].values[abs(org_var) < 1e-18] = np.nan
    # stats.describe(rel_corr[cor_fields[j]][org_fields[4*j + k]], axis=None, nan_policy='omit')
    print('#---- Mean relative corrections')
    print(np.mean(rel_corr[cor_fields[j]][org_fields[4*j + k]]).values)
    print('#---- Maximum relative corrections')
    print(np.max(rel_corr[cor_fields[j]][org_fields[4*j + k]]).values)

# where is the largest corrections?

k = 1

i1, i2, i3 = np.where(abs(cor_var) == np.max(abs(cor_var)))
cor_var.values[i1[0], i2[0], i3[0]] * 900
org_var.values[i1[0], i2[0], i3[0]]

cor_qte = exp_org_o[expid[i]]['wiso'][cor_fields[j]][i1[0], 3:, i2[0], i3[0]].values
cor_qte * 900
org_qte = exp_org_o[expid[i]]['wiso']['phy_tagqte_41'][i1[0], 3:, i2[0], i3[0]].values
final_qte = org_qte + cor_qte

qm1 = np.array([
    exp_org_o[expid[i]]['wiso'].q_01[i1[0] - 1, level[j]-1, i2[0], i3[0]].values,
    exp_org_o[expid[i]]['wiso'].q_02[i1[0] - 1, level[j]-1, i2[0], i3[0]].values,
    exp_org_o[expid[i]]['wiso'].q_03[i1[0] - 1, level[j]-1, i2[0], i3[0]].values,
    exp_org_o[expid[i]]['wiso'].q_04[i1[0] - 1, level[j]-1, i2[0], i3[0]].values])
q = np.array([
    exp_org_o[expid[i]]['wiso'].q_01[i1[0], level[j]-1, i2[0], i3[0]].values,
    exp_org_o[expid[i]]['wiso'].q_02[i1[0], level[j]-1, i2[0], i3[0]].values,
    exp_org_o[expid[i]]['wiso'].q_03[i1[0], level[j]-1, i2[0], i3[0]].values,
    exp_org_o[expid[i]]['wiso'].q_04[i1[0], level[j]-1, i2[0], i3[0]].values])

org_q = qm1 + 900*org_qte
cor_q = qm1 + 900*final_qte


# where is the largest relative corrections?
i1, i2, i3 = np.where(abs(rel_corr[cor_fields[j]][org_fields[4*j + k]]) == np.max(abs(rel_corr[cor_fields[j]][org_fields[4*j + k]])))
np.max(abs(rel_corr[cor_fields[j]][org_fields[4*j + k]])).values
rel_corr[cor_fields[j]][org_fields[4*j + k]][i1[0], i2[0], i3[0]].values
cor_var[i1[0], i2[0], i3[0]].values * 900
org_var[i1[0], i2[0], i3[0]].values
exp_org_o[expid[i]]['wiso'][cor_fields[j]][i1[0], 3+k, i2[0], i3[0]].values * 900
exp_org_o[expid[i]]['wiso'][org_fields[4*j + k]][i1[0], level[j]-1, i2[0], i3[0]].values







'''
    # print('#---------------- No correction in the first three fields')
    # print(np.max(abs(exp_org_o[expid[i]]['wiso'][cor_fields[j]][:, :3, :, :])).values)

    where_max_rel_corr = np.where(abs(rel_corr[org_fields[j]]) == np.nanmax(abs(rel_corr[org_fields[j]])))
    print('#--------maximum relative correction')
    print(rel_corr[org_fields[j]].values[where_max_rel_corr])
    
    print('#----original values at maximum relative correction')
    print(exp_org_o[expid[i]]['wiso'][org_fields[j]][:, 3:, :, :].values[where_max_rel_corr])
    print('#----corrections at maximum relative correction')
    print(exp_org_o[expid[i]]['wiso'][cor_fields[j]][:, 3:, :, :].values[where_max_rel_corr])
    # exp_org_o[expid[i]]['wiso'][cor_fields[j]][:, 3:, :, :].values[where_max_rel_corr] / exp_org_o[expid[i]]['wiso'][org_fields[j]][:, 3:, :, :].values[where_max_rel_corr]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check Martin's corrections

i = 0
expid[i]

var_names = np.array([var for var in exp_org_o[expid[i]]['wiso'].data_vars])

cor_prefix = ['phy_tagqte_c_', 'phy_tagxlte_c_', 'phy_tagxite_c_']
org_prefix = ['q_', 'xl_', 'xi_']

level = [47, 41, 34]

org_var = {}
cor_var = {}
rel_corr = {}

# loop along q/xl/xi
for l in range(len(cor_prefix)):
    # l = len(cor_prefix) - 1
    print('#----------------------------------------------------------------')
    print('#-------------------------------- ' + cor_prefix[l])
    
    #---- define variables to store results
    # (klev, kwiso, ktime, klat, klon)
    org_var[org_prefix[l][:-1]] = np.zeros((
        len(level),
        len(exp_org_o[expid[i]]['wiso'].wisotype) - 3,
        len(exp_org_o[expid[i]]['wiso'].time),
        len(exp_org_o[expid[i]]['wiso'].lat),
        len(exp_org_o[expid[i]]['wiso'].lon),
        ), dtype=np.float32)
    cor_var[org_prefix[l][:-1]] = np.zeros(org_var[org_prefix[l][:-1]].shape)
    rel_corr[org_prefix[l][:-1]] = np.zeros(org_var[org_prefix[l][:-1]].shape)
    
    cor_names = var_names[
            [var.startswith(cor_prefix[l]) for var in var_names]]
    
    for j in range(len(cor_names)): # loop along klev
        # j = len(cor_names) - 1
        print('#---------------- ' + cor_names[j])
        
        org_names = var_names[
            [var.startswith(org_prefix[l]) for var in var_names]]
        
        for k in range(len(org_names)): # loop along kwiso
            # k = len(org_names) - 1
            print('#-------- ' + org_names[k])
            
            org_var[org_prefix[l][:-1]][j, k, :, :, :] = exp_org_o[expid[i]]['wiso'][org_names[k]][
                :, level[j]-1, :, :]
            
            cor_var[org_prefix[l][:-1]][j, k, :, :, :] = exp_org_o[expid[i]]['wiso'][cor_names[j]][
                :, 3+k, :, :]
    
    
    # If a correction is positive and the original variable is still not positive, it will not be considered.
    where_neg_org = np.where((cor_var[org_prefix[l][:-1]] > 0) & (org_var[org_prefix[l][:-1]]<=0))
    cor_var[org_prefix[l][:-1]][where_neg_org] = np.nan
    org_var[org_prefix[l][:-1]][where_neg_org] = np.nan
    
    # Do not account for grid cells where absolute corrections is less than 1e-18.
    where_small_cor = np.where(abs(cor_var[org_prefix[l][:-1]]) < 1e-18)
    cor_var[org_prefix[l][:-1]][where_small_cor] = np.nan
    org_var[org_prefix[l][:-1]][where_small_cor] = np.nan
    
    print('#---- Mean original fields')
    print(np.nanmean(abs(org_var[org_prefix[l][:-1]])))
    print('#---- Mean corrections')
    print(np.nanmean(abs(cor_var[org_prefix[l][:-1]])) * 900)
    
    where_max_corr = np.where(abs(cor_var[org_prefix[l][:-1]]) == np.nanmax(abs(cor_var[org_prefix[l][:-1]])))
    print('#---- Maximum corrections')
    print(cor_var[org_prefix[l][:-1]][where_max_corr] * 900)
    print('#---- Original values at maximum corrections')
    print(org_var[org_prefix[l][:-1]][where_max_corr])
    
    rel_corr[org_prefix[l][:-1]] = abs(cor_var[org_prefix[l][:-1]] * 900 / org_var[org_prefix[l][:-1]])
    
    # If absolute original variable is less than 1e-18, relative corrections will not be considered.
    rel_corr[org_prefix[l][:-1]][abs(org_var[org_prefix[l][:-1]]) < 1e-15] = np.nan
    print('#---- Mean relative corrections')
    print(np.nanmean(rel_corr[org_prefix[l][:-1]]))
    print('#---- Maximum relative corrections')
    print(np.nanmax(rel_corr[org_prefix[l][:-1]]))



#---------------- detailed checks

where_max_rel = np.where(rel_corr == np.nanmax(rel_corr))
rel_corr[where_max_rel]
org_var[where_max_rel]
cor_var[where_max_rel] * 900
ij, ik, i3, i4, i5 = where_max_rel
exp_org_o[expid[i]]['wiso'][org_names[ik[0]]][i3[0], level[ij[0]]-1, i4[0], i5[0]].values
exp_org_o[expid[i]]['wiso'][cor_names[ij[0]]][i3[0], 3+ik[0], i4[0], i5[0]].values * 900

qm1 = [
    exp_org_o[expid[i]]['wiso'][org_names[0]][i3[0]-1, level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[1]][i3[0]-1, level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[2]][i3[0]-1, level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[3]][i3[0]-1, level[ij[0]]-1, i4[0], i5[0]].values,
]
q = [
    exp_org_o[expid[i]]['wiso'][org_names[0]][i3[0], level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[1]][i3[0], level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[2]][i3[0], level[ij[0]]-1, i4[0], i5[0]].values,
    exp_org_o[expid[i]]['wiso'][org_names[3]][i3[0], level[ij[0]]-1, i4[0], i5[0]].values,
]

org_qte = exp_org_o[expid[i]]['wiso']['phy_tagxite_34'][i3[0], 3:, i4[0], i5[0]].values * 900
cor_qte = exp_org_o[expid[i]]['wiso'][cor_names[ij[0]]][i3[0], 3:, i4[0], i5[0]].values * 900
final_qte = org_qte + cor_qte

org_q = qm1 + org_qte
cor_q = qm1 + final_qte

exp_org_o[expid[i]]['wiso'].xi16o[i3[0]-1, level[ij[0]]-1, i4[0], i5[0]].values
exp_org_o[expid[i]]['wiso'].xi16o[i3[0], level[ij[0]]-1, i4[0], i5[0]].values
'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region check bit identity

i = 0
j = 1
expid[i] + '   ' + expid[j]


#-------------------------------- normal climate variables

(exp_org_o[expid[i]]['echam'].evap == exp_org_o[expid[j]]['echam'].evap).all()
(exp_org_o[expid[i]]['echam'].aprl == exp_org_o[expid[j]]['echam'].aprl).all()
(exp_org_o[expid[i]]['echam'].temp2 == exp_org_o[expid[j]]['echam'].temp2).all()
(exp_org_o[expid[i]]['echam'].u10 == exp_org_o[expid[j]]['echam'].u10).all()
(exp_org_o[expid[i]]['echam'].q2m == exp_org_o[expid[j]]['echam'].q2m).all()
(exp_org_o[expid[i]]['echam'].q == exp_org_o[expid[j]]['echam'].q).all()
(exp_org_o[expid[i]]['echam'].evapwac == exp_org_o[expid[j]]['echam'].evapwac).all()

#-------------------------------- wiso variables

(exp_org_o[expid[i]]['wiso'].wisoevap[:, :3] == exp_org_o[expid[j]]['wiso'].wisoevap[:, :3]).all()
(exp_org_o[expid[i]]['wiso'].wisoevapwac[:, :3] == exp_org_o[expid[j]]['wiso'].wisoevapwac[:, :3]).all()
(exp_org_o[expid[i]]['wiso'].wisoaprl[:, :3] == exp_org_o[expid[j]]['wiso'].wisoaprl[:, :3]).all()
(exp_org_o[expid[i]]['wiso'].wisoaprc[:, :3] == exp_org_o[expid[j]]['wiso'].wisoaprc[:, :3]).all()
(exp_org_o[expid[i]]['wiso'].wisows[:, :3] == exp_org_o[expid[j]]['wiso'].wisows[:, :3]).all()

(exp_org_o[expid[i]]['wiso'].tagmap[:, :3] == exp_org_o[expid[j]]['wiso'].tagmap[:, :3]).all()


#-------------------------------- check separate VS. combined run

(exp_org_o[expid[i]]['wiso'].wisoevap.values[:, kstart:kend] == exp_org_o[expid[j]]['wiso'].wisoevap.values[:, 3:10]).all()
(exp_org_o[expid[i]]['wiso'].wisoevapwac.values[:, kstart:kend] == exp_org_o[expid[j]]['wiso'].wisoevapwac.values[:, 3:10]).all()
(exp_org_o[expid[i]]['wiso'].wisoaprl.values[:, kstart:kend] == exp_org_o[expid[j]]['wiso'].wisoaprl.values[:, 3:10]).all()
(exp_org_o[expid[i]]['wiso'].wisoaprc.values[:, kstart:kend] == exp_org_o[expid[j]]['wiso'].wisoaprc.values[:, 3:10]).all()
(exp_org_o[expid[i]]['wiso'].wisows.values[:, kstart:kend] == exp_org_o[expid[j]]['wiso'].wisows.values[:, 3:10]).all()

(exp_org_o[expid[i]]['wiso'].tagmap.values[:, kstart:kend] == exp_org_o[expid[j]]['wiso'].tagmap.values[:, 3:10]).all()

np.max(abs((exp_org_o[expid[i]]['wiso'].wisoevap.values[:, kstart:kend] - exp_org_o[expid[j]]['wiso'].wisoevap.values[:, 3:10])))
np.max(abs((exp_org_o[expid[i]]['wiso'].wisoevapwac.values[:, kstart:kend] - exp_org_o[expid[j]]['wiso'].wisoevapwac.values[:, 3:10])))
np.max(abs((exp_org_o[expid[i]]['wiso'].wisoaprl.values[:, kstart:kend] - exp_org_o[expid[j]]['wiso'].wisoaprl.values[:, 3:10])))
np.max(abs((exp_org_o[expid[i]]['wiso'].wisoaprc.values[:, kstart:kend] - exp_org_o[expid[j]]['wiso'].wisoaprc.values[:, 3:10])))
np.max(abs((exp_org_o[expid[i]]['wiso'].wisows.values[:, kstart:kend] - exp_org_o[expid[j]]['wiso'].wisows.values[:, 3:10])))
np.max(abs((exp_org_o[expid[i]]['wiso'].tagmap.values[:, kstart:kend] - exp_org_o[expid[j]]['wiso'].tagmap.values[:, 3:10])))

test = exp_org_o[expid[i]]['wiso'].wisoaprc.values[:, kstart:kend] - exp_org_o[expid[j]]['wiso'].wisoaprc.values[:, 3:10]
where_max = np.where(abs(test) == np.max(abs(test)))
test[where_max]
exp_org_o[expid[i]]['wiso'].wisoaprc.values[:, kstart:kend][where_max]
exp_org_o[expid[j]]['wiso'].wisoaprc.values[:, 3:10][where_max]


#-------------------------------- while lupdate_tagmap = False

i = 0
for j in range(10):
    print((exp_org_o[expid[i]]['wiso'].wisoevap[:, 3:5, :, :].values == exp_org_o[expid[i]]['wiso'].wisoevap[:, (2*j+5):(2*j+7), :, :].values).all())
    print((exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3:5, :, :].values == exp_org_o[expid[i]]['wiso'].wisoaprl[:, (2*j+5):(2*j+7), :, :].values).all())
    print((exp_org_o[expid[i]]['wiso'].wisows[:, 3:5, :, :].values == exp_org_o[expid[i]]['wiso'].wisows[:, (2*j+5):(2*j+7), :, :].values).all())

j = 9
np.max(abs(exp_org_o[expid[i]]['wiso'].wisoevap[:, 3:5, :, :].values - exp_org_o[expid[i]]['wiso'].wisoevap[:, (2*j+5):(2*j+7), :, :].values))
np.max(abs(exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3:5, :, :].values - exp_org_o[expid[i]]['wiso'].wisoaprl[:, (2*j+5):(2*j+7), :, :].values))

test = exp_org_o[expid[i]]['wiso'].wisoevap[:, 3:5, :, :].values - exp_org_o[expid[i]]['wiso'].wisoevap[:, (2*j+5):(2*j+7), :, :].values
wheremax = np.where(abs(test) == np.max(abs(test)))
test[wheremax]
exp_org_o[expid[i]]['wiso'].wisoevap[:, 3:5, :, :].values[wheremax]
exp_org_o[expid[i]]['wiso'].wisoevap[:, (2*j+5):(2*j+7), :, :].values[wheremax]


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check precipitation from land/ocean

i = 0
expid[i]


pre_land = {}
pre_all = {}
pre_land_fraction = {}

pre_land[expid[i]] = exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 3, :, :]
pre_all[expid[i]] = exp_org_o[expid[i]]['echam'].aprl[:, :, :] + exp_org_o[expid[i]]['echam'].aprc[:, :, :]
pre_land_fraction[expid[i]] = pre_land[expid[i]] / pre_all[expid[i]]

pre_land_fraction[expid[i]].values[pre_all[expid[i]].values < 1e-9] = np.nan
stats.describe(pre_land_fraction[expid[i]], axis=None, nan_policy='omit')

pre_land_fraction[expid[i]].to_netcdf('scratch/test/test00.nc')



'''
# pre_ocean = {}
# pre_ocean[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4:, :, :]).sum(axis=1)

(pre_land[expid[i]] < 0).sum()
pre_land[expid[i]].values[pre_land[expid[i]] < 0]
pre_all[expid[i]].values[pre_land[expid[i]] < 0]

(pre_all[expid[i]] < 1e-9).sum()
'''
# endregion
# -----------------------------------------------------------------------------

