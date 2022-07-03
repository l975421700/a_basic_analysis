

# =============================================================================
# region import packages

# management
import glob
import warnings
warnings.filterwarnings('ignore')
import pickle

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
# =============================================================================


# =============================================================================
# =============================================================================
# region import orignal model output

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'

expid = [
    # 'pi_echam6_1d_200_3.60',
    # 'pi_echam6_1d_206_3.64',
    # 'pi_echam6_1d_207_3.65',
    # 'pi_echam6_1d_208_3.66',
    # 'pi_echam6_1d_209_3.67',
    # 'pi_echam6_1d_210_3.68',
    # 'pi_echam6_1d_211_3.69',
    # 'pi_echam6_1d_212_3.69',
    'pi_echam6_1d_213_3.69',
    # 'pi_echam6_1d_201_3.60',
    # 'pi_echam6_1d_202_3.60',
    # 'pi_echam6_1d_203_3.60',
    # 'pi_echam6_1d_204_3.60',
    
    # 'pi_echam6_1y_201_3.60',
    # 'pi_echam6_1y_202_3.60',
    # 'pi_echam6_1y_203_3.60',
    # 'pi_echam6_1y_204_3.60',
    # 'pi_echam6_1y_205_3.60',
    # 'pi_echam6_1y_206_3.60',
    # 'pi_echam6_1y_207_3.60',
    # 'pi_echam6_1y_208_3.60',
    # 'pi_echam6_1y_209_3.60',
    # 'pi_echam6_1y_210_3.60',
    # 'pi_echam6_1y_211_3.60',
    # 'pi_echam6_1y_212_3.60',
    # 'pi_echam6_1y_213_3.60',
    # 'pi_echam6_1y_214_3.60',
    # 'pi_echam6_1y_215_3.60',
    # 'pi_echam6_1y_216_3.60',
    # 'pi_echam6_1y_217_3.60',
    # 'pi_echam6_1y_218_3.65',
    # 'pi_echam6_1y_219_3.66',
    # 'pi_echam6_1y_220_3.69',
    ]

exp_org_o = {}

for i in range(len(expid)):
    # i=0
    
    exp_org_o[expid[i]] = {}
    
    # echam
    exp_org_o[expid[i]]['echam'] = xr.open_dataset( exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_echam.nc' )
    
    # wiso
    exp_org_o[expid[i]]['wiso'] = xr.open_dataset( exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.01_wiso.nc' )
    
    print(str(i) + '/' + str(len(expid) - 1))

'''
# (exp_org_o[expid[i]]['echam'].hyam + exp_org_o[expid[i]]['echam'].hybm * 101325)[46]
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check evaporation conservation

i = 0
expid[i]

#---------------- check tagmap conservation
stats.describe(exp_org_o[expid[i]]['wiso'].tagmap[:, 3:, :, :].sum(axis=1), axis=None)

#---------------- check evap conservation
echam_evap = exp_org_o[expid[i]]['echam'].evap[:, :, :]
(echam_evap == exp_org_o[expid[i]]['wiso'].wisoevap[:, 0, :, :]).all()
wiso_evap = exp_org_o[expid[i]]['wiso'].wisoevap[:, 3:, :, :].sum(axis=1)
diff_evap = echam_evap - wiso_evap
# stats.describe(diff_evap, axis=None)
np.max(abs(diff_evap))


#---------------- check evap conservation in detail
i1, i2, i3 = np.where(abs(diff_evap) == np.max(abs(diff_evap)))
diff_evap[i1[0], i2[0], i3[0]].values
echam_evap[i1[0], i2[0], i3[0]].values
wiso_evap[i1[0], i2[0], i3[0]].values






'''
# diff_evap.to_netcdf('scratch/test/test.nc')

i=2
# global 1
np.max(abs(exp_org_o[expid[i]]['wiso'].wisoevap[:, 3, :, :] - exp_org_o[expid[i]]['wiso'].wisoevap[:, 0, :, :]))
# global 0
np.max(abs(exp_org_o[expid[i]]['wiso'].wisoevap[:, 4, :, :]))
'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
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
# =============================================================================


# =============================================================================
# =============================================================================
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
# =============================================================================


# =============================================================================
# =============================================================================
# region check humidity conservation - q, xl, xi

#---------------- check humidity coservation

i = 0
expid[i]

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_sum_humidity.pkl', 'rb') as f:
    sum_humidity = pickle.load(f)

diff_q = sum_humidity['q'] - exp_org_o[expid[i]]['wiso'].q16o
# diff_q.values[exp_org_o[expid[i]]['wiso'].q16o.values < 0] = np.nan
np.max(abs(diff_q[:, :, :, :]))
# np.max(abs(diff_q[3, :, :, :]))
# diff_q.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_diff_q.nc')
# exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_diff_q.nc'

# diff_q_mean = diff_q.mean(axis=(0, 1))
# diff_q_mean.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '_diff_q_mean.nc')
# del diff_q

diff_xl = sum_humidity['xl'] - exp_org_o[expid[i]]['wiso'].xl16o
# diff_xl.values[exp_org_o[expid[i]]['wiso'].xl16o.values < 0] = np.nan
np.max(abs(diff_xl))
# np.max(abs(diff_xl[1, :, :, :]))
# diff_xl.to_netcdf('scratch/test/diff_xl.nc')

# del diff_xl

diff_xi = sum_humidity['xi'] - exp_org_o[expid[i]]['wiso'].xi16o
# diff_xi.values[exp_org_o[expid[i]]['wiso'].xi16o.values < 0] = np.nan
np.max(abs(diff_xi))
# np.max(abs(diff_xi[0, :, :, :]))
# diff_xi.to_netcdf('scratch/test/diff_xi.nc')

# del diff_xi


#---------------- check humidity coservation in detail
i1, i2, i3, i4 = np.where(abs(diff_q[:, :, :, :]) == np.max(abs(diff_q[:, :, :, :])))
# i1, i2, i3, i4 = np.where(abs(diff_q[:, :, :, :]) > 0)
# i1[0:5]
# i2[0:5]
# i3[0:5]
# i4[0:5]

np.max(abs(diff_q[:, :, :, :]))
diff_q[i1[0], i2[0], i3[0], i4[0]].values
sum_humidity['q'][i1[0], i2[0], i3[0], i4[0]].values
exp_org_o[expid[i]]['wiso'].q16o[i1[0], i2[0], i3[0], i4[0]].values

# q_wiso = np.array((
#     exp_org_o[expid[i]]['wiso'].q_01[i1[0], i2[0], i3[0], i4[0]].values,
#     exp_org_o[expid[i]]['wiso'].q_02[i1[0], i2[0], i3[0], i4[0]].values,
#     # exp_org_o[expid[i]]['wiso'].q_03[i1[0], i2[0], i3[0], i4[0]].values,
#     # exp_org_o[expid[i]]['wiso'].q_04[i1[0], i2[0], i3[0], i4[0]].values,
#     ))
# q_wiso.sum()


i1, i2, i3, i4 = np.where(abs(diff_xl[:, :, :, :]) == np.max(abs(diff_xl[:, :, :, :])))
# i1, i2, i3, i4 = np.where(abs(diff_xl[:, :, :, :]) > 0)
# (i1 == 1).sum()
# i1[0:5]
# i2[0:5]
# i3[0:5]
# i4[0:5]

np.max(abs(diff_xl[:, :, :, :]))
diff_xl[i1[0], i2[0], i3[0], i4[0]].values
sum_humidity['xl'][i1[0], i2[0], i3[0], i4[0]].values
exp_org_o[expid[i]]['wiso'].xl16o[i1[0], i2[0], i3[0], i4[0]].values

# np.max(sum_humidity['xl'].values[np.where(abs(diff_xl[:, :, :, :]) > 0)])
# np.max(exp_org_o[expid[i]]['wiso'].xl16o.values[np.where(abs(diff_xl[:, :, :, :]) > 0)])

i1, i2, i3, i4 = np.where(abs(diff_xi[:, :, :, :]) == np.max(abs(diff_xi[:, :, :, :])))
# i1, i2, i3, i4 = np.where(abs(diff_xi[:, :, :, :]) > 0)
# (i1 == 1).sum()
# i1[0:29]
# i2[0:29]
# i3[0:29]
# i4[0:29]

diff_xi[i1[0], i2[0], i3[0], i4[0]].values
sum_humidity['xi'][i1[0], i2[0], i3[0], i4[0]].values
exp_org_o[expid[i]]['wiso'].xi16o[i1[0], i2[0], i3[0], i4[0]].values

# xi_wiso = np.array((
#     exp_org_o[expid[i]]['wiso'].xi_01[i1[0], i2[0], i3[0], i4[0]].values,
#     exp_org_o[expid[i]]['wiso'].xi_02[i1[0], i2[0], i3[0], i4[0]].values,
#     # exp_org_o[expid[i]]['wiso'].xi_03[i1[0], i2[0], i3[0], i4[0]].values,
#     # exp_org_o[expid[i]]['wiso'].xi_04[i1[0], i2[0], i3[0], i4[0]].values,
#     ))
# xi_wiso.sum()


'''
exp_org_o[expid[i]]['echam'].xl
exp_org_o[expid[i]]['echam'].xi

i = 0
np.max(abs(exp_org_o[expid[i]]['echam'].q - exp_org_o[expid[i]]['wiso'].q16o.values))

# check humidity conservation
diff_q = (exp_org_o[expid[i]]['wiso'].q_01 + exp_org_o[expid[i]]['wiso'].q_02 + exp_org_o[expid[i]]['wiso'].q_03 + exp_org_o[expid[i]]['wiso'].q_04) - exp_org_o[expid[i]]['wiso'].q16o
diff_q.values[exp_org_o[expid[i]]['wiso'].q16o < 0] = np.nan
print(np.max(abs(diff_q)))

diff_xl = (exp_org_o[expid[i]]['wiso'].xl_01 + exp_org_o[expid[i]]['wiso'].xl_02 + exp_org_o[expid[i]]['wiso'].xl_03 + exp_org_o[expid[i]]['wiso'].xl_04) - exp_org_o[expid[i]]['wiso'].xl16o
print(np.max(abs(diff_xl)))
diff_xi = (exp_org_o[expid[i]]['wiso'].xi_01 + exp_org_o[expid[i]]['wiso'].xi_02 + exp_org_o[expid[i]]['wiso'].xi_03 + exp_org_o[expid[i]]['wiso'].xi_04) - exp_org_o[expid[i]]['wiso'].xi16o
print(np.max(abs(diff_xi)))


# check negative q
neg_q_01 = exp_org_o[expid[i]]['wiso'].q_01.values[exp_org_o[expid[i]]['wiso'].q_01 < 0]
neg_q16o = exp_org_o[expid[i]]['wiso'].q16o.values[exp_org_o[expid[i]]['wiso'].q16o < 0]

test = exp_org_o[expid[i]]['wiso'].q_01
test.values[test > -1e-18] = 0
test.to_netcdf('scratch/test/test.nc')
(test < -1e-15).sum()

#---------------- check global 0 works as expected

i = 0
expid[i]

q_global0 = exp_org_o[expid[i]]['wiso'].q_02
np.max(abs(q_global0))
# stats.describe(abs(q_global0), axis=None)
# q_global0.to_netcdf('scratch/test/q_02.nc')

where_max_q = np.where(abs(q_global0) == np.max(abs(q_global0)))
q_global0[where_max_q]
exp_org_o[expid[i]]['wiso'].q_01[where_max_q].values
exp_org_o[expid[i]]['echam'].q[where_max_q].values

np.max(abs(exp_org_o[expid[i]]['wiso'].xl_02))
np.max(abs(exp_org_o[expid[i]]['wiso'].xi_02))

np.max(abs(exp_org_o[expid[i]]['wiso'].wisosnsic[:, 4, :, :]))


#---------------- check global 1

i = 2
expid[i]

q_echam = exp_org_o[expid[i]]['echam'].q
# q_wiso = exp_org_o[expid[i]]['wiso'].q16o
# (q_echam == q_wiso).all()
q_global1 = exp_org_o[expid[i]]['wiso'].q_01
(q_echam == q_global1).all()

(exp_org_o[expid[i]]['echam'].q == exp_org_o[expid[i]]['wiso'].q_01).all()
(exp_org_o[expid[i]]['echam'].xl == exp_org_o[expid[i]]['wiso'].xl_01).all()
(exp_org_o[expid[i]]['echam'].xi == exp_org_o[expid[i]]['wiso'].xi_01).all()

np.max(abs(exp_org_o[expid[i]]['echam'].xl - exp_org_o[expid[i]]['wiso'].xl_01))
np.max(abs(exp_org_o[expid[i]]['echam'].xi - exp_org_o[expid[i]]['wiso'].xi_01))

diff_xl = exp_org_o[expid[i]]['echam'].xl - exp_org_o[expid[i]]['wiso'].xl_01
where_max_diff_xl = np.where(abs(diff_xl) == np.max(abs(diff_xl)))
diff_xl.values[where_max_diff_xl]
exp_org_o[expid[i]]['echam'].xl.values[where_max_diff_xl]
exp_org_o[expid[i]]['wiso'].xl_01.values[where_max_diff_xl]

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
# region check precipitation conservation

#-------------------------------- check precipitation conservation

i = 0
expid[i]

echam_apr = exp_org_o[expid[i]]['echam'].aprl[:, :, :] + exp_org_o[expid[i]]['echam'].aprc[:, :, :]
echam_apr.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.echam_apr.nc')
# np.max(abs(echam_apr - (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 0, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 0, :, :])))
wiso_apr = exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3:, :, :].sum(axis=1) + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 3:, :, :].sum(axis=1)
wiso_apr.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wiso_apr.nc')
diff_apr = wiso_apr - echam_apr
diff_apr.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.diff_apr.nc')
stats.describe(abs(diff_apr), axis=None)

# diff_apr_mean = diff_apr.mean(axis=0)
# diff_apr_mean.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.diff_apr_mean.nc')
# stats.describe(abs(echam_apr[:, :, :]), axis=None)

rel_diff_apr = diff_apr / echam_apr
rel_diff_apr.values[echam_apr.values < 1e-9] = np.nan
rel_diff_apr.to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.rel_diff_apr.nc')
stats.describe(rel_diff_apr, axis=None, nan_policy='omit')

#-------------------------------- check precipitation conservation in detail
i1 = 62
i2 = 1
i3 = 126
rel_diff_apr[i1, i2, i3]
echam_apr[i1, i2, i3]
wiso_apr[i1, i2, i3]

where_max_diff_apr = np.where(abs(diff_apr) == np.max(abs(diff_apr)))
diff_apr.values[where_max_diff_apr]
wiso_apr.values[where_max_diff_apr]
echam_apr.values[where_max_diff_apr]

where_max_rel_diff_apr = np.where(abs(rel_diff_apr) == np.max(abs(rel_diff_apr)))
rel_diff_apr.values[where_max_rel_diff_apr]
wiso_apr.values[where_max_rel_diff_apr]
echam_apr.values[where_max_rel_diff_apr]


#-------------------------------- check l. scale & conv. precipitation

echam_aprl = exp_org_o[expid[i]]['echam'].aprl[:, :, :]
wiso_aprl = exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3:, :, :].sum(axis=1)
diff_aprl = wiso_aprl - echam_aprl
stats.describe(abs(diff_aprl[:, :, :]), axis=None)

echam_aprc = exp_org_o[expid[i]]['echam'].aprc[:, :, :]
wiso_aprc = exp_org_o[expid[i]]['wiso'].wisoaprc[:, 3:, :, :].sum(axis=1)
diff_aprc = wiso_aprc - echam_aprc
stats.describe(abs(diff_aprc[:, :, :]), axis=None)


#-------------------------------- plot tracer precipitation percentage deviation
pltlevel = np.arange(-5, 5.001, 0.01)
pltticks = np.arange(-5, 5.001, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('BrBG', len(pltlevel))

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    exp_org_o[expid[i]]['echam'].lon,
    exp_org_o[expid[i]]['echam'].lat,
    rel_diff_apr[-1, :, :] * 100,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Differences between tracer and normal precipitation, as percentage of normal precipitation [$\%$]\n' + expid[i],
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
    diff_apr[-1, :, :] * 10**7,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal",  pad=0.02,
    fraction=0.14, shrink=0.6, aspect=40, anchor=(0.5, 0.7),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    u'Differences between tracer and normal precipitation [$10^{-7} \; mm \; s^{-1}$]\n' + expid[i],
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(exp_odir+expid[i]+'/viz/echam/'+expid[i]+'.diff_apr.png')



'''

# check large-scale precipitation conservation
diff_aprl = exp_org_o[expid[i]]['wiso'].wisoaprl[:, 0, :, :] - exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3:, :, :].sum(axis=1)

# stats.describe(abs(diff_aprl), axis=None)
np.max(abs(diff_aprl))
np.mean(abs(diff_aprl))

i1, i2, i3 = np.where(abs(diff_aprl) == np.max(abs(diff_aprl)))

diff_aprl[i1[0], i2[0], i3[0]].values
exp_org_o[expid[i]]['echam'].aprl[i1[0], i2[0], i3[0]].values
exp_org_o[expid[i]]['wiso'].wisoaprl[i1[0], 3:, i2[0], i3[0]].values.sum()


# check convective precipitation conservation
diff_aprc = exp_org_o[expid[i]]['echam'].aprc[:, :, :] - exp_org_o[expid[i]]['wiso'].wisoaprc[:, 3:, :, :].sum(axis=1)
stats.describe(abs(diff_aprc), axis=None)
np.max(abs(diff_aprc))

i1, i2, i3 = np.where(abs(diff_aprc) == np.max(abs(diff_aprc)))

diff_aprc[i1[0], i2[0], i3[0]].values
exp_org_o[expid[i]]['echam'].aprc[i1[0], i2[0], i3[0]].values
exp_org_o[expid[i]]['wiso'].wisoaprc[i1[0], 3:, i2[0], i3[0]].values.sum()

#-------------------------------- check precipitation conservation in detail

i = 0
expid[i]

#---------------- global 0 works as expected

aprl_global0 = exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4, :, :]
np.max(abs(aprl_global0))
# aprl_global0.to_netcdf('scratch/test/test.nc')
# where_nonzero_aprl = np.where(abs(aprl_global0) > 1e-10)
# aprl_global0.values[where_nonzero_aprl]
aprc_global0 = exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4, :, :]
np.max(abs(aprc_global0))
# aprc_global0.to_netcdf('scratch/test/test1.nc')
# where_nonzero_aprc = np.where(abs(aprc_global0) > 1e-10)
# aprc_global0.values[where_nonzero_aprc]

#---------------- global 1

echam_apr_global1 = exp_org_o[expid[i]]['echam'].aprl[:, :, :] + exp_org_o[expid[i]]['echam'].aprc[:, :, :]
wiso_apr_global1 = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 3, :, :])
diff_apr_global1 = wiso_apr_global1 - echam_apr_global1
# diff_apr_global1.to_netcdf('scratch/test/test2.nc')
stats.describe(abs(diff_apr_global1), axis=None)

where_max = np.where(abs(diff_apr_global1) == np.max(abs(diff_apr_global1)))
diff_apr_global1.values[where_max]
echam_apr_global1.values[where_max]
wiso_apr_global1.values[where_max]

# where_large_diff = np.where(abs(diff_apr_global1) > 1e-10)
# diff_apr_global1.values[where_large_diff]
# echam_apr_global1.values[where_large_diff]
# wiso_apr_global1.values[where_large_diff]


diff_aprl_global1 = exp_org_o[expid[i]]['echam'].aprl[:, :, :] - exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3, :, :]
np.max(diff_aprl_global1)
(abs(diff_aprl_global1) > 0).sum()
# diff_aprl_global1_1 = exp_org_o[expid[i]]['wiso'].wisoaprl[:, 0, :, :] - exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3, :, :]
# np.max(diff_aprl_global1_1)
# diff_aprl_global1.to_netcdf('scratch/test/test3.nc')

diff_aprc_global1 = exp_org_o[expid[i]]['echam'].aprc[:, :, :] - exp_org_o[expid[i]]['wiso'].wisoaprc[:, 3, :, :]
np.max(diff_aprc_global1)
# diff_aprc_global1.to_netcdf('scratch/test/test4.nc')

diff_aprs_global1 = exp_org_o[expid[i]]['echam'].aprs[:, :, :] - exp_org_o[expid[i]]['wiso'].wisoaprs[:, 3, :, :]
np.max(diff_aprs_global1)

'''
# endregion
# =============================================================================


# =============================================================================
# =============================================================================
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
# =============================================================================


# =============================================================================
# =============================================================================
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
# =============================================================================


# =============================================================================
# region check negative evaporation with tagmap

i = 0
expid[i]

#---- over water
post_wisoevap = exp_org_o[expid[i]]['echam'].evapwac.values[1:, None, :, :] * exp_org_o[expid[i]]['wiso'].tagmap[:-1, 3:, :, :]
diff_wisoevap = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevapwac[1:, 3:, :, :].values
diff_wisoevap.values[np.where(exp_org_o[expid[i]]['wiso'].wisoevapwac[1:, 3:, :, :].values >= 0)] = 0
# stats.describe(diff_wisoevap[:, :, :, :], axis=None)
np.max(abs(diff_wisoevap[:, :, :, :]))


#---- over land
post_wisoevap = exp_org_o[expid[i]]['echam'].evaplac.values[1:, None, :, :] * exp_org_o[expid[i]]['wiso'].tagmap[:-1, 3:, :, :]
diff_wisoevap = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevaplac[1:, 3:, :, :].values
diff_wisoevap.values[np.where(exp_org_o[expid[i]]['wiso'].wisoevaplac[1:, 3:, :, :].values >= 0)] = 0
# stats.describe(diff_wisoevap[:, :, :, :], axis=None)
np.max(abs(diff_wisoevap[:, :, :, :]))


#---- over ice
post_wisoevap = exp_org_o[expid[i]]['echam'].evapiac.values[1:, None, :, :] * exp_org_o[expid[i]]['wiso'].tagmap[:-1, 3:, :, :]
diff_wisoevap = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevapiac[1:, 3:, :, :].values
diff_wisoevap.values[np.where(exp_org_o[expid[i]]['wiso'].wisoevapiac[1:, 3:, :, :].values >= 0)] = 0
# stats.describe(diff_wisoevap[:, :, :, :], axis=None)
np.max(abs(diff_wisoevap[:, :, :, :]))




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
# =============================================================================


# =============================================================================
# region check all evaporation with ztag_fac_*

i = 0
expid[i]


#---- over water
post_wisoevap = exp_org_o[expid[i]]['echam'].evapwac.values[:, None, :, :] * exp_org_o[expid[i]]['wiso'].ztag_fac_water[:, 3:, :, :]
diff_wisoevap = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevapwac[:, 3:, :, :].values
# stats.describe(diff_wisoevap[:, :, :, :], axis=None)
np.max(abs(diff_wisoevap[1:, :, :, :]))


#---- over land
post_wisoevap = exp_org_o[expid[i]]['echam'].evaplac.values[:, None, :, :] * exp_org_o[expid[i]]['wiso'].ztag_fac_land[:, 3:, :, :]
diff_wisoevap = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevaplac[:, 3:, :, :].values
# stats.describe(diff_wisoevap[1:, :, :, :], axis=None)
np.max(abs(diff_wisoevap[1:, :, :, :]))


#---- over ice
post_wisoevap = exp_org_o[expid[i]]['echam'].evapiac.values[:, None, :, :] * exp_org_o[expid[i]]['wiso'].ztag_fac_ice[:, 3:, :, :]
diff_wisoevap = post_wisoevap - exp_org_o[expid[i]]['wiso'].wisoevapiac[:, 3:, :, :].values
# stats.describe(diff_wisoevap[:, :, :, :], axis=None)
np.max(abs(diff_wisoevap[1:, :, :, :]))



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
# =============================================================================


# =============================================================================
# =============================================================================
# region check bit identity

i = 4
j = 5
expid[i] + '   ' + expid[j]



# normal climate variables
(exp_org_o[expid[i]]['echam'].evap == exp_org_o[expid[j]]['echam'].evap).all()
(exp_org_o[expid[i]]['echam'].aprl == exp_org_o[expid[j]]['echam'].aprl).all()
(exp_org_o[expid[i]]['echam'].temp2 == exp_org_o[expid[j]]['echam'].temp2).all()
(exp_org_o[expid[i]]['echam'].u10 == exp_org_o[expid[j]]['echam'].u10).all()
(exp_org_o[expid[i]]['echam'].q2m == exp_org_o[expid[j]]['echam'].q2m).all()
(exp_org_o[expid[i]]['echam'].q == exp_org_o[expid[j]]['echam'].q).all()
(exp_org_o[expid[i]]['echam'].evapwac == exp_org_o[expid[j]]['echam'].evapwac).all()

(exp_org_o[expid[i]]['wiso'].tagmap.values == exp_org_o[expid[j]]['wiso'].tagmap.values).all()


# wiso variables
(exp_org_o[expid[i]]['wiso'].wisoevap.values == exp_org_o[expid[j]]['wiso'].wisoevap.values).all()
(exp_org_o[expid[i]]['wiso'].wisoevapwac.values == exp_org_o[expid[j]]['wiso'].wisoevapwac.values).all()
(exp_org_o[expid[i]]['wiso'].wisoaprl.values == exp_org_o[expid[j]]['wiso'].wisoaprl.values).all()
(exp_org_o[expid[i]]['wiso'].wisoaprc.values == exp_org_o[expid[j]]['wiso'].wisoaprc.values).all()
(exp_org_o[expid[i]]['wiso'].wisows.values == exp_org_o[expid[j]]['wiso'].wisows.values).all()

(exp_org_o[expid[i]]['wiso'].tagmap.values == exp_org_o[expid[j]]['wiso'].tagmap.values).all()



'''
np.max(abs((exp_org_o[expid[i]]['wiso'].wisoevap.values - exp_org_o[expid[j]]['wiso'].wisoevap.values)))

i = 0
m=21
n=23
(exp_org_o[expid[i]]['wiso'].wisoevap[:, 3:5, :, :].values == exp_org_o[expid[i]]['wiso'].wisoevap[:, m:n, :, :].values).all()
(exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3:5, :, :].values == exp_org_o[expid[i]]['wiso'].wisoaprl[:, m:n, :, :].values).all()
(exp_org_o[expid[i]]['wiso'].wisows[:, 3:5, :, :].values == exp_org_o[expid[i]]['wiso'].wisows[:, m:n, :, :].values).all()


m = 18
n = 21
(exp_org_o[expid[i]]['wiso'].wisoevap[:, m:n, :, :].values == exp_org_o[expid[j]]['wiso'].wisoevap[:, 3:6, :, :].values).all()
(exp_org_o[expid[i]]['wiso'].wisoaprl[:, m:n, :, :].values == exp_org_o[expid[j]]['wiso'].wisoaprl[:, 3:6, :, :].values).all()
(exp_org_o[expid[i]]['wiso'].wisows[:, m:n, :, :].values == exp_org_o[expid[j]]['wiso'].wisows[:, 3:6, :, :].values).all()



np.nanmax(abs(exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3:5, :, :].values - exp_org_o[expid[i]]['wiso'].wisoaprl[:, m:n, :, :].values))

test = exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3:5, :, :] - exp_org_o[expid[i]]['wiso'].wisoaprl[:, m:n, :, :]
test.to_netcdf('scratch/test/test00.nc')

'''
# endregion
# =============================================================================


# =============================================================================
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
# =============================================================================

