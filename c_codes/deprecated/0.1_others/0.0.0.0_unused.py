

# =============================================================================
# =============================================================================
# region refined calculation of seasonal average

#-------------------------------- scaled SST

minsst = {}
maxsst = {}
minsst['pi_echam6_1y_204_3.60'] = 260
maxsst['pi_echam6_1y_204_3.60'] = 310
i = 0
expid[i]

ocean_pre = {}
sst_scaled_pre = {}
pre_weighted_tsw = {}
ocean_pre_sea = {}
sst_scaled_pre_sea = {}
pre_weighted_tsw_sea = {}
ocean_pre_am = {}
sst_scaled_pre_am = {}
pre_weighted_tsw_am = {}

ocean_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4:, :, :] +  exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4:, :, :]).sum(axis=1)
sst_scaled_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4, :, :])


#---------------- seasonal values

# spin up: one year
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)
sst_scaled_pre_sea[expid[i]] = sst_scaled_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_tsw_sea[expid[i]] = sst_scaled_pre_sea[expid[i]] / ocean_pre_sea[expid[i]] * (maxsst[expid[i]] - minsst[expid[i]]) + minsst[expid[i]] - zerok
pre_weighted_tsw_sea[expid[i]].values[np.where(ocean_pre_sea[expid[i]] < 1e-9)] = np.nan
pre_weighted_tsw_sea[expid[i]] = pre_weighted_tsw_sea[expid[i]].rename('pre_weighted_tsw_sea')


#---------------- annual mean values

# spin up: one year
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)
sst_scaled_pre_am[expid[i]] = sst_scaled_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)

pre_weighted_tsw_am[expid[i]] = sst_scaled_pre_am[expid[i]] / ocean_pre_am[expid[i]] * (maxsst[expid[i]] - minsst[expid[i]]) + minsst[expid[i]] - zerok
pre_weighted_tsw_am[expid[i]].values[np.where(ocean_pre_am[expid[i]] < 1e-9)] = np.nan
pre_weighted_tsw_am[expid[i]] = pre_weighted_tsw_am[expid[i]].rename('pre_weighted_tsw_am')


#-------------------------------- binned SST

i = 1
expid[i]

sstbins = np.concatenate((np.array([-100]), np.arange(0, 28.1, 2), np.array([100])))
sstbins_mid = np.arange(-1, 29.1, 2)

sst_binned_pre = {}
sst_binned_pre_sea = {}
sst_binned_pre_am = {}


sst_binned_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4:, :, :])
ocean_pre[expid[i]] = sst_binned_pre[expid[i]].sum(axis=1)

#---------------- seasonal values
# spin up: one year

sst_binned_pre_sea[expid[i]] = sst_binned_pre[expid[i]][60:, :, :, :].groupby('time.season').sum(dim="time", skipna=True)
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][60:, :, :].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_tsw_sea[expid[i]] = ( sst_binned_pre_sea[expid[i]] * sstbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre_sea[expid[i]]
pre_weighted_tsw_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_sea[expid[i]] = pre_weighted_tsw_sea[expid[i]].rename('pre_weighted_tsw_sea')


#---------------- annual mean values

# spin up: one year

sst_binned_pre_am[expid[i]] = sst_binned_pre[expid[i]][60:, :, :, :].mean(dim="time", skipna=True)
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][60:, :, :].mean(dim="time", skipna=True)

pre_weighted_tsw_am[expid[i]] = ( sst_binned_pre_am[expid[i]] * sstbins_mid[:, None, None]).sum(axis=0) / ocean_pre_am[expid[i]]
pre_weighted_tsw_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_am[expid[i]] = pre_weighted_tsw_am[expid[i]].rename('pre_weighted_tsw_am')

#-------- basic set
i = 0
j = 1
lon = pre_weighted_tsw_am[expid[i]].lon
lat = pre_weighted_tsw_am[expid[i]].lat
print('#-------- ' + expid[i] + ' & '+ expid[j])
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2_8' + expid[i] + '_and_' + expid[j] + '_pre_weighted_tsw_compare.png'
cbar_label1 = 'Precipitation-weighted source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted source SST [$°C$]'

pltlevel = np.arange(0, 32.01, 2)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()
# pltcmp = rb_colormap(pltlevel, right_c = 'Blues_r', left_c = 'Reds')

pltlevel2 = np.arange(-2, 2.01, 0.25)
pltticks2 = np.arange(-2, 2.01, 0.5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)).reversed()
# pltcmp2 = rb_colormap(pltlevel, right_c = 'Blues_r', left_c = 'Reds')

nrow = 3
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = globe_plot(ax_org = axs[irow, jcol],
                                     add_grid_labels=False)

#-------- annual mean values
axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[i]],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[j]],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[i]] - \
        pre_weighted_tsw_am[expid[j]],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- DJF values
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[j]].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].sel(season='DJF') - pre_weighted_tsw_sea[expid[j]].sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- JJA values
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[j]].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].sel(season='JJA') - pre_weighted_tsw_sea[expid[j]].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)


plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'DJF', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'JJA', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'Scaling tag map with SST', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'Partitioning tag map based on SST', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'Differences', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='vertical')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, -0.2), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = fm_bottom, top = 0.96)
fig.savefig(output_png)



# endregion
# =============================================================================


# =============================================================================
# region create tagmap_nhsh_qg1

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 2

tagmap_nhsh_qg1 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_nhsh_qg1.tagmap.sel(level=slice(1, 3))[:, :] = 1

# sh
tagmap_nhsh_qg1.tagmap.sel(level=4, lat=slice(0, -90))[:, :] = 1

# nh
tagmap_nhsh_qg1.tagmap.sel(level=5, lat=slice(90, 0))[:, :] = 1

tagmap_nhsh_qg1.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nhsh_qg1.nc', mode='w')


'''
tagmap_nhsh_qg1 = xr.open_dataset('/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nhsh_qg1.nc')

# check
np.max(tagmap_nhsh_qg1.tagmap[3:5, :, :].sum(axis=0))
np.min(tagmap_nhsh_qg1.tagmap[3:5, :, :].sum(axis=0))
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_nhsh_qg2

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 4

tagmap_nhsh_qg2 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# sh
tagmap_nhsh_qg2.tagmap.sel(level=5, lat=slice(0, -90))[:, :] = 1

# nh
tagmap_nhsh_qg2.tagmap.sel(level=6, lat=slice(90, 0))[:, :] = 1

tagmap_nhsh_qg2.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nhsh_qg2.nc', mode='w')


'''
tagmap_nhsh_qg0 = xr.open_dataset('/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nhsh_qg0.nc')

# check
np.max(tagmap_nhsh_qg0.tagmap[3:5, :, :].sum(axis=0))
np.min(tagmap_nhsh_qg0.tagmap[3:5, :, :].sum(axis=0))
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_nhsh_qg3

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 3

tagmap_nhsh_qg3 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_nhsh_qg3.tagmap.sel(level=slice(1, 3))[:, :] = 1

# sh
tagmap_nhsh_qg3.tagmap.sel(level=4, lat=slice(0, -90))[:, :] = 1

# nh
tagmap_nhsh_qg3.tagmap.sel(level=5, lat=slice(90, 0))[:, :] = 1

tagmap_nhsh_qg3.tagmap.sel(level=6)[:, :] = 1

tagmap_nhsh_qg3.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nhsh_qg3.nc', mode='w')

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_nhsh_qg4

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 3

tagmap_nhsh_qg4 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_nhsh_qg4.tagmap.sel(level=slice(1, 3))[:, :] = 1

tagmap_nhsh_qg4.tagmap.sel(level=4)[:, :] = 1

# sh
tagmap_nhsh_qg4.tagmap.sel(level=5, lat=slice(0, -90))[:, :] = 1

# nh
tagmap_nhsh_qg4.tagmap.sel(level=6, lat=slice(90, 0))[:, :] = 1

tagmap_nhsh_qg4.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nhsh_qg4.nc', mode='w')

'''
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g_nhsh_p1

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 8

tagmap_g_nhsh_p1 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# globe
tagmap_g_nhsh_p1.tagmap.sel(level=4)[:, :] = 1

# sh
tagmap_g_nhsh_p1.tagmap.sel(level=5, lat=slice(0, -90))[:, :] = 1

# nh
tagmap_g_nhsh_p1.tagmap.sel(level=6, lat=slice(90, 0))[:, :] = 1

# sh + 1
tagmap_g_nhsh_p1.tagmap.sel(level=7)[:, :] = tagmap_g_nhsh_p1.tagmap.sel(level=5) + 1

# nh + 1
tagmap_g_nhsh_p1.tagmap.sel(level=8)[:, :] = tagmap_g_nhsh_p1.tagmap.sel(level=6) + 1

# globe + 1
tagmap_g_nhsh_p1.tagmap.sel(level=9)[:, :] = tagmap_g_nhsh_p1.tagmap.sel(level=4) + 1

# globe -0.5
tagmap_g_nhsh_p1.tagmap.sel(level=10)[:, :] = tagmap_g_nhsh_p1.tagmap.sel(level=4) -0.5

# globe
tagmap_g_nhsh_p1.tagmap.sel(level=11)[:, :] = tagmap_g_nhsh_p1.tagmap.sel(level=4)



tagmap_g_nhsh_p1.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g_nhsh_p1.nc')




'''
# check
stats.describe(tagmap_g_nhsh_ew.tagmap.sel(level=4), axis=None)
stats.describe(tagmap_g_nhsh_ew.tagmap.sel(level=5) + tagmap_g_nhsh_ew.tagmap.sel(level=6), axis=None)
stats.describe(tagmap_g_nhsh_ew.tagmap.sel(level=7) + tagmap_g_nhsh_ew.tagmap.sel(level=8), axis=None)
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g_nhsh_1p

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 7

tagmap_g_nhsh_1p = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# globe
tagmap_g_nhsh_1p.tagmap.sel(level=4)[:, :] = 1

# sh
tagmap_g_nhsh_1p.tagmap.sel(level=5, lat=slice(0, -90))[:, :] = 1

# nh
tagmap_g_nhsh_1p.tagmap.sel(level=6, lat=slice(90, 0))[:, :] = 1

# globe but 1 point
tagmap_g_nhsh_1p.tagmap.sel(level=7)[:, :] = 1
tagmap_g_nhsh_1p.tagmap.sel(level=7)[48, 92] = 0

# 1 point
tagmap_g_nhsh_1p.tagmap.sel(level=8)[48, 92] = 1

# zero fields
tagmap_g_nhsh_1p.tagmap.sel(level=9)[:, :] = 0

# globe
tagmap_g_nhsh_1p.tagmap.sel(level=10)[:, :] = 1

tagmap_g_nhsh_1p.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g_nhsh_1p.nc')

'''
# check
stats.describe(tagmap_g_nhsh_1p.tagmap.sel(level=4), axis=None)
stats.describe(tagmap_g_nhsh_1p.tagmap.sel(level=5) + tagmap_g_nhsh_1p.tagmap.sel(level=6), axis=None)
stats.describe(tagmap_g_nhsh_1p.tagmap.sel(level=7) + tagmap_g_nhsh_1p.tagmap.sel(level=8), axis=None)
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_nh_s

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 1

tagmap_nh_s = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# nh sea
tagmap_nh_s.tagmap.sel(level=4, lat=slice(90, 0))[:, :] = \
    1 - slm.sel(lat=slice(90, 0)).values

tagmap_nh_s.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nh_s.nc', mode='w')


'''
cdo --reduce_dim -selvar,slm /work/ollie/qigao001/output/awiesm-2.1-wiso/pi_final/pi_final_qg_tag4_1y_0/analysis/echam/pi_final_qg_tag4_1y_0_2000_2003.01_echam.am.nc /work/ollie/qigao001/output/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc

'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_nh_l

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 1

tagmap_nh_l = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# nh land
tagmap_nh_l.tagmap.sel(level=4, lat=slice(90, 0))[:, :] = \
    slm.sel(lat=slice(90, 0)).values

tagmap_nh_l.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_nh_l.nc', mode='w')


'''
'''
# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g_nhsh_sl

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 5

tagmap_g_nhsh_sl = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

# globe
tagmap_g_nhsh_sl.tagmap.sel(level=4)[:, :] = 1

# sh land
tagmap_g_nhsh_sl.tagmap.sel(level=5, lat=slice(0, -90))[:, :] = \
    slm.sel(lat=slice(0, -90)).values

# sh sea
tagmap_g_nhsh_sl.tagmap.sel(level=6, lat=slice(0, -90))[:, :] = \
    1 - slm.sel(lat=slice(0, -90)).values

# nh land
tagmap_g_nhsh_sl.tagmap.sel(level=7, lat=slice(90, 0))[:, :] = \
    slm.sel(lat=slice(90, 0)).values

# nh sea
tagmap_g_nhsh_sl.tagmap.sel(level=8, lat=slice(90, 0))[:, :] = \
    1 - slm.sel(lat=slice(90, 0)).values



tagmap_g_nhsh_sl.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g_nhsh_sl.nc')


# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 1

tagmap_g = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_g.tagmap.sel(level=4)[:, :] = 1

tagmap_g.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g.nc')


# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g2

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 2

tagmap_g2 = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_g2.tagmap.sel(level=4)[:, :] = 1
tagmap_g2.tagmap.sel(level=5)[:, :] = 0.5

tagmap_g2.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g2.nc')


# endregion
# =============================================================================


# =============================================================================
# region create tagmap_g2i

echam_t63_slm = xr.open_dataset(
    '/work/ollie/qigao001/scratch/others/land_sea_masks/ECHAM6_T63_slm.nc')
lon = echam_t63_slm.lon.values
lat = echam_t63_slm.lat.values
slm = echam_t63_slm.slm.squeeze()


ntag = 2

tagmap_g2i = xr.Dataset(
    {"tagmap": (
        ("level", "lat", "lon"),
        np.zeros((ntag+3, len(lat), len(lon)), dtype=np.double)),
     },
    coords={
        "level": np.arange(1, ntag+3+1, 1, dtype='int32'),
        "lat": lat,
        "lon": lon,
    }
)

tagmap_g2i.tagmap.sel(level=4)[:, :] = 1
tagmap_g2i.tagmap.sel(level=5)[:, :] = 1

tagmap_g2i.to_netcdf(
    '/home/ollie/qigao001/startdump/tagging/tagmap3/tagmap_g2i.nc')


# endregion
# =============================================================================


# =============================================================================
# region attempt to plot Fesom2 original mesh

import pyfesom2 as pf
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import xarray as xr
from a_basic_analysis.b_module.mapplot import (
    hemisphere_plot,
)
import cartopy.crs as ccrs

#### trial 1 does not work
mesh = pf.load_mesh('startdump/mesh_CORE2_finaltopo_mean', abg=[50, 15, -90])
# mesh = pf.load_mesh('startdump/mesh_CORE2_finaltopo_mean/')
sst_fesom_pi_final_qg = xr.open_dataset('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/sst.fesom.200001_202912.nc')

pf.tplot(
    mesh, sst_fesom_pi_final_qg.sst[0, ],
    ptype='tri', box=[-180, 180, 60, 90], mapproj='np', lw=0.5)
plt.savefig('figures/0_test/trial0.png')


#### trial 2 it works
mesh = pf.load_mesh('core2/')
datapath = "/work/ollie/qigao001/"
data = pf.get_data(datapath, "temp", 1950, mesh)
# pf.plot(mesh, data[:,0])
pf.tplot(
    mesh, data[:,0],
    ptype='tri', box=[-180, 180, 60, 90], mapproj='np', lw=0.5)
plt.savefig('figures/0_test/trial1.png')



#### trial 3 it works
mesh = pf.load_mesh('startdump/core2/')
sst_fesom_pi_final_qg = xr.open_dataset('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/sst.fesom.200001_202912.nc')

def cut_region(mesh, box):
    """
    Return mesh elements (triangles) and their indices corresponding to a bounding box.
    Parameters
    ----------
    mesh : object
        FESOM mesh object
    box : list
        Coordinates of the box in [-180 180 -90 90] format.
        Default set to [13, 30, 53, 66], Baltic Sea.
    Returns
    -------
    elem_no_nan : array
        elements that belong to the region defined by `box`.
    no_nan_triangles: array
        boolean array of element size with True for elements
        that belong to the `box`.
    """
    left, right, down, up = box

    selection = (
        (mesh.x2 >= left)
        & (mesh.x2 <= right)
        & (mesh.y2 >= down)
        & (mesh.y2 <= up)
    )

    elem_selection = selection[mesh.elem]

    no_nan_triangles = np.all(elem_selection, axis=1)

    elem_no_nan = mesh.elem[no_nan_triangles]

    return elem_no_nan, no_nan_triangles

def get_no_cyclic(mesh, elem_no_nan):
    """Compute non cyclic elements of the mesh."""
    d = mesh.x2[elem_no_nan].max(axis=1) - mesh.x2[elem_no_nan].min(axis=1)
    no_cyclic_elem = np.argwhere(d < 100)
    return no_cyclic_elem.ravel()

box=[-180, 180, 60, 90]
box_mesh = [box[0] - 1, box[1] + 1, box[2] - 1, box[3] + 1]

elem_no_nan, no_nan_triangles = cut_region(mesh, box_mesh)
no_cyclic_elem2 = get_no_cyclic(mesh, elem_no_nan)

data_to_plot = sst_fesom_pi_final_qg.sst[0, ].copy()
data_to_plot[data_to_plot == 0] = np.nan
elem_to_plot = elem_no_nan[no_cyclic_elem2]

fig, ax = hemisphere_plot(
    southextent=60,
    add_grid_labels=False, plot_scalebar=False, grid_color='black',
    fm_left=0.06, fm_right=0.94, fm_bottom=0.04, fm_top=0.98,
    )
ax.tripcolor(
    mesh.x2, mesh.y2, elem_to_plot, data_to_plot,
    transform=ccrs.PlateCarree(), cmap=cm.get_cmap('viridis'),
    edgecolors="k", lw=0.05, alpha=1,)

fig.savefig('figures/0_test/trial2.png')


#### trial 4 it works
mesh = pf.load_mesh('startdump/core2/')
sst_fesom_pi_final_qg = xr.open_dataset('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/sst.fesom.200001_202912.nc')
pf.tplot(
    mesh, sst_fesom_pi_final_qg.sst[0,:],
    ptype='tri', box=[-180, 180, 60, 90], mapproj='np', lw=0.5)
plt.savefig('figures/0_test/trial4.png')


#### trial 5 it does not work
mesh = pf.load_mesh('startdump/core2/')
sst_fesom_pi_final_qg = xr.open_dataset('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/sst.fesom.200001_202912.nc')
fig, ax = hemisphere_plot(
    southextent=60,
    add_grid_labels=False, plot_scalebar=False, grid_color='black',
    fm_left=0.06, fm_right=0.94, fm_bottom=0.04, fm_top=0.98,
    )
ax.tripcolor(
    mesh.x2, mesh.y2, mesh.elem, sst_fesom_pi_final_qg.sst[0,:],
    transform=ccrs.PlateCarree(), cmap=cm.get_cmap('viridis'),
    edgecolors="k", lw=0.05, alpha=1,)

fig.savefig('figures/0_test/trial5.png')


# endregion
# =============================================================================


# =============================================================================
# region Don't run Annual mean siconc in AWI-CM-1-1-MR, historical, r1i1p1f1

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'AWI/'
source = 'AWI-CM-1-1-MR/'
experiment = 'historical/'
member = 'r1i1p1f1/'
table = 'SIday/'
variable = 'siconc/'
grid = 'gn/'
version = 'v20181218/'

siconc_awc_mr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))

siconc_awc_mr_hi_r1 = xr.open_mfdataset(
    siconc_awc_mr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_siconc_awc_mr_hi_r1 = xr.Dataset(
    data_vars={
        'siconc': (('ncells'), np.zeros((len(siconc_awc_mr_hi_r1.ncells)))),
        'lat_bnds': (('ncells', 'vertices'), siconc_awc_mr_hi_r1.lat_bnds.data),
        'lon_bnds': (('ncells', 'vertices'), siconc_awc_mr_hi_r1.lon_bnds.data),
    },
    coords={
        'ncells': siconc_awc_mr_hi_r1.ncells.data,
        'vertices': siconc_awc_mr_hi_r1.vertices.data,
        'lat': siconc_awc_mr_hi_r1.lat.data,
        'lon': siconc_awc_mr_hi_r1.lon.data,
    },
    attrs=siconc_awc_mr_hi_r1.attrs
)

am_siconc_awc_mr_hi_r1_80 = xr.Dataset(
    data_vars={
        'siconc': (('ncells'), np.zeros((len(siconc_awc_mr_hi_r1.ncells)))),
        'lat_bnds': (('ncells', 'vertices'), siconc_awc_mr_hi_r1.lat_bnds.data),
        'lon_bnds': (('ncells', 'vertices'), siconc_awc_mr_hi_r1.lon_bnds.data),
    },
    coords={
        'ncells': siconc_awc_mr_hi_r1.ncells.data,
        'vertices': siconc_awc_mr_hi_r1.vertices.data,
        'lat': siconc_awc_mr_hi_r1.lat.data,
        'lon': siconc_awc_mr_hi_r1.lon.data,
    },
    attrs=siconc_awc_mr_hi_r1.attrs
)

am_siconc_awc_mr_hi_r1.siconc[:] = siconc_awc_mr_hi_r1.siconc.sel(time=slice(
    '1979-01-01', '2014-12-31')).mean(axis=0).values
am_siconc_awc_mr_hi_r1_80.siconc[:] = siconc_awc_mr_hi_r1.siconc.sel(time=slice(
    '1980-01-01', '2014-12-31')).mean(axis=0).values

# am_siconc_awc_mr_hi_r1 = siconc_awc_mr_hi_r1.siconc.sel(time=slice(
#     '1979-01-01', '2014-12-31')).mean(axis=0).load()
# am_siconc_awc_mr_hi_r1_80 = siconc_awc_mr_hi_r1.siconc.sel(time=slice(
#     '1980-01-01', '2014-12-31')).mean(axis=0).load()


am_siconc_awc_mr_hi_r1.to_netcdf(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc')
am_siconc_awc_mr_hi_r1_80.to_netcdf(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80.nc')


'''
#### cdo commands
cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_cdo_remap.nc
cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80_cdo_remap.nc


# global_2: lonlat (180x90) grid
# global_1: lonlat (360x180) grid
# cdo -P 4 -remapcon,global_1 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231.nc regridding_weights_AWI-CM-1-1-MR_FESOM.nc

#### generate weights does not work
cdo genycon,global_1 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_18500101-18501231.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM
cdo remap,global_1, bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_cdo_remap.nc


# cdo genycon, global_2 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_18500101-18501231.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM
# cdo genycon, global_1 siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-301012.nc regrid2deg_weights_AWI-ESM-1-1-LR

#### slow
from cdo import Cdo
cdo = Cdo()
cdo.remapcon(
    'r360x180',
    input='/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231.nc',
    output='bas_palaeoclim_qino/scratch/cmip6/historical/siconc/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231_cdo_regrid.nc')

# regrid_weights = xr.open_dataset(
#     '/home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/AWI_org/regrid2deg_weights_AWI-ESM-1-1-LR.nc'
# )


stats.describe(am_siconc_awc_mr_hi_r1.siconc, axis=None, nan_policy='omit')
stats.describe(am_siconc_awc_mr_hi_r1_80.siconc, axis=None, nan_policy='omit')
'''
# endregion
# =============================================================================


# =============================================================================
# region Don't run Annual mean siconc in AWI-CM-1-1-MR, historical, r1i1p1f1

# Import data
top_dir = '/badc/cmip6/data/CMIP6/'
mip = 'CMIP/'
institute = 'AWI/'
source = 'AWI-CM-1-1-MR/'
experiment = 'historical/'
member = 'r1i1p1f1/'
table = 'SIday/'
variable = 'siconc/'
grid = 'gn/'
version = 'v20181218/'

siconc_awc_mr_hi_r1_fl = np.array(sorted(glob.glob(
    top_dir + mip + institute + source + experiment + member +
    table + variable + grid + version + '*.nc',
)))

siconc_awc_mr_hi_r1 = xr.open_mfdataset(
    siconc_awc_mr_hi_r1_fl, data_vars='minimal',
    coords='minimal', compat='override',
)

am_siconc_awc_mr_hi_r1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc')

am_siconc_awc_mr_hi_r1_reformat = siconc_awc_mr_hi_r1.copy()

am_siconc_awc_mr_hi_r1_reformat['siconc'] = am_siconc_awc_mr_hi_r1_reformat['siconc'][0, :]

am_siconc_awc_mr_hi_r1_reformat['siconc'][:] = \
    am_siconc_awc_mr_hi_r1.siconc[:].values


am_siconc_awc_mr_hi_r1_reformat.to_netcdf(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_reformat.nc')
# am_siconc_awc_mr_hi_r1_80.to_netcdf(
#     'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80.nc')


'''
# check
am_siconc_awc_mr_hi_r1_reformat = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_reformat.nc'
)
am_siconc_awc_mr_hi_r1 = xr.open_dataset(
    'bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc')
(am_siconc_awc_mr_hi_r1_reformat.siconc == am_siconc_awc_mr_hi_r1.siconc).all()


#### cdo commands
cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_reformat.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_reformat_cdo_remap.nc



cdo -P 4 -remapcon,global_1 bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_80_cdo_remap.nc


# global_2: lonlat (180x90) grid
# global_1: lonlat (360x180) grid
# cdo -P 4 -remapcon,global_1 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231.nc regridding_weights_AWI-CM-1-1-MR_FESOM.nc

#### generate weights does not work
cdo genycon,global_1 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_18500101-18501231.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM
cdo remap,global_1, bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1.nc bas_palaeoclim_qino/scratch/cmip6/historical/siconc/am_siconc_awc_mr_hi_r1_cdo_remap.nc


# cdo genycon, global_2 /badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_18500101-18501231.nc /home/users/qino/bas_palaeoclim_qino/scratch/cmip6/historical/siconc/regridding_weights_AWI-CM-1-1-MR_FESOM
# cdo genycon, global_1 siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-301012.nc regrid2deg_weights_AWI-ESM-1-1-LR

#### slow
from cdo import Cdo
cdo = Cdo()
cdo.remapcon(
    'r360x180',
    input='/badc/cmip6/data/CMIP6/CMIP/AWI/AWI-CM-1-1-MR/historical/r1i1p1f1/SIday/siconc/gn/v20181218/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231.nc',
    output='bas_palaeoclim_qino/scratch/cmip6/historical/siconc/siconc_SIday_AWI-CM-1-1-MR_historical_r1i1p1f1_gn_20110101-20141231_cdo_regrid.nc')

# regrid_weights = xr.open_dataset(
#     '/home/users/rahuls/LOUISE/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/AWI_org/regrid2deg_weights_AWI-ESM-1-1-LR.nc'
# )


stats.describe(am_siconc_awc_mr_hi_r1.siconc, axis=None, nan_policy='omit')
stats.describe(am_siconc_awc_mr_hi_r1_80.siconc, axis=None, nan_policy='omit')
'''
# endregion
# =============================================================================


# =============================================================================
# region Handling CloudSat raw data

# Using deepice_pynio environment
# conda activate deepice_pynio
# python -c "from IPython import start_ipython; start_ipython()" --no-autoindent

# management
from pyhdf.SD import SD, SDC
import warnings
warnings.filterwarnings('ignore')

# data analysis
import numpy as np
import xarray as xr
import dask
dask.config.set({"array.slicing.split_large_chunks": False})

# import Nio
# Nio.option_defaults['TimePeriodSuffix'] = True
# cloudsat2007001 = xr.open_dataset(
#     'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf',
#     engine='pynio',
# )

'''

import Nio
Nio.option_defaults
opt = Nio.option_defaults
opt['CompressionLevel'] = -2

# file = Nio.open_file(
#     'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf', 'r')
# file.dataset()

# from pyhdf.SD import SD, SDC
# hdf = SD(
#     'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf',
#     SDC.READ)
# hdf.datasets()

# import xarray as xr
# cloudsat2007001_nc = xr.open_dataset(
#     'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.nc',
# )

# import pandas
# pandas.read_hdf(
#     'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf'
# )

import rioxarray as rxr
cloudsat2007001 = rxr.open_rasterio(
    'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf',
    masked=True,
)

import rasterio
cloudsat2007001 = rasterio.open(
    'bas_palaeoclim_qino/observations/products/CloudSat/2C-SNOW-PROFILE.P1_R05/2007001005141_03607_CS_2C-SNOW-PROFILE_GRANULE_P1_R05_E02_F00.hdf',
)
'''
# endregion
# =============================================================================

