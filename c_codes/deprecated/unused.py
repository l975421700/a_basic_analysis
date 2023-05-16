

# -----------------------------------------------------------------------------
# region trial of linear regressions


%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(9876789)

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x ** 2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)

X = sm.add_constant(X)

y = np.dot(X, beta) + e

model = sm.OLS(y, X)

results = model.fit()
print(results.summary())

print("Parameters: ", results.params)
print("R2: ", results.rsquared)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi sep sic

output_png = 'figures/7_lig/7.0_sim_rec/7.0.1_sic/7.0.1.1 lig-pi sep sic multiple models 1deg.png'
cbar_label = 'Sep SIC [$\%$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-70, cm_max=20, cm_interval1=10, cm_interval2=10, cmap='PuOr',
    reversed=False, asymmetric=True,)

nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.12, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            northextent=-38, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        # MC
        plt_scatter = axs[irow, jcol].scatter(
            x = lig_recs['MC']['interpolated'].Longitude,
            y = lig_recs['MC']['interpolated'].Latitude,
            c = lig_recs['MC']['interpolated']['sic_anom_hadisst_sep'],
            s=64, lw=0.5, marker='^', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        sep_lig_pi = lig_pi_sic_regrid_alltime[model]['mm'][8].values
        
        plt_mesh = axs[irow, jcol].pcolormesh(
            lon, lat, sep_lig_pi,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        rmse = {}
        for irec in ['MC']:
            # irec = 'MC'
            print(irec)
            
            rmse[irec] = int(np.round(SO_sep_sic_site_values[irec].loc[
                SO_sep_sic_site_values[irec].Model == model
                ]['sim_rec_sep_sic_lig_pi'].mean(), 0))
        
        plt.text(
            0.5, 1.05,
            model + ': ' + str(rmse['MC']),
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_scatter, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.4),)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi am sat

output_png = 'figures/7_lig/7.0_sim_rec/7.0.2_tas/7.0.2.1 lig-pi am tas multiple models 1deg.png'
cbar_label = 'Annual SAT [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

max_size = 80
scale_size = 16

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.12, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            northextent=-60, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        plt_scatter = axs[irow, jcol].scatter(
            x = lig_recs['EC']['AIS_am'].Longitude,
            y = lig_recs['EC']['AIS_am'].Latitude,
            c = lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'],
            s = max_size - scale_size * lig_recs['EC']['AIS_am']['127 ka 2s PIAn [°C]'],
            lw=0.5, marker='o', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        am_lig_pi = lig_pi_tas_regrid_alltime[model]['am'].values[0]
        
        plt_mesh = axs[irow, jcol].pcolormesh(
            lon, lat, am_lig_pi,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        rmse = {}
        for irec in ['EC']:
            # irec = 'EC'
            print(irec)
            
            rmse[irec] = np.round(AIS_ann_tas_site_values[irec].loc[
                AIS_ann_tas_site_values[irec].Model == model
                ]['sim_rec_ann_tas_lig_pi'].mean(), 1)
        
        plt.text(
            0.5, 1.05, model + ': ' + str(rmse['EC']),
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_scatter, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.4),)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi am sst

output_png = 'figures/7_lig/7.0_sim_rec/7.0.0_sst/7.0.0.1 lig-pi am sst multiple models 1deg.png'
cbar_label = r'$\mathit{lig127k}$' + ' vs. ' + r'$\mathit{piControl}$' + ' Annual SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

max_size = 80
scale_size = 16

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.12, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            northextent=-38, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        # JH
        axs[irow, jcol].scatter(
            x = lig_recs['JH']['SO_ann'].Longitude,
            y = lig_recs['JH']['SO_ann'].Latitude,
            c = lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'],
            s = max_size - scale_size * lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'],
            lw=0.5, marker='s', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

        # EC SST
        axs[irow, jcol].scatter(
            x = lig_recs['EC']['SO_ann'].Longitude,
            y = lig_recs['EC']['SO_ann'].Latitude,
            c = lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'],
            s = max_size - scale_size * lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]'],
            lw=0.5, marker='o', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

        # DC
        plt_scatter = axs[irow, jcol].scatter(
            x = lig_recs['DC']['annual_128'].Longitude,
            y = lig_recs['DC']['annual_128'].Latitude,
            c = lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'],
            s = max_size - scale_size * 1,
            lw=0.5, marker='v', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        am_lig_pi = lig_pi_sst_regrid_alltime[model]['am'].values[0]
        # ann_lig = lig_sst_regrid_alltime[model]['ann'].values
        # ann_pi = pi_sst_regrid_alltime[model]['ann'].values
        
        # ttest_fdr_res = ttest_fdr_control(ann_lig, ann_pi,)
        
        plt_mesh = axs[irow, jcol].pcolormesh(
            lon, lat, am_lig_pi,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        # axs[irow, jcol].scatter(
        #     x=lon[(ttest_fdr_res == False) & np.isfinite(am_lig_pi)],
        #     y=lat[(ttest_fdr_res == False) & np.isfinite(am_lig_pi)],
        #     s=0.3, c='k', marker='.', edgecolors='none',
        #     transform=ccrs.PlateCarree(),
        #     )
        
        rmse = {}
        for irec in ['EC', 'JH', 'DC']:
            # irec = 'EC'
            print(irec)
            
            rmse[irec] = np.round(SO_ann_sst_site_values[irec].loc[
                SO_ann_sst_site_values[irec].Model == model
                ]['sim_rec_ann_sst_lig_pi'].mean(), 1)
        
        plt.text(
            0.5, 1.05,
            model + ': ' + \
                str(rmse['EC']) + ', ' + \
                    str(rmse['JH']) + ', ' + \
                        str(rmse['DC']),
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_scatter, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.4),)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig-pi summer sst

output_png = 'figures/7_lig/7.0_sim_rec/7.0.0_sst/7.0.0.1 lig-pi summer sst multiple models 1deg.png'
cbar_label = 'Summer SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='RdBu',)

nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

max_size = 80
scale_size = 16

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.12, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            northextent=-38, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1
        
        # JH
        axs[irow, jcol].scatter(
            x = lig_recs['JH']['SO_jfm'].Longitude,
            y = lig_recs['JH']['SO_jfm'].Latitude,
            c = lig_recs['JH']['SO_jfm']['127 ka SST anomaly (°C)'],
            s = max_size - scale_size * lig_recs['JH']['SO_jfm']['127 ka 2σ (°C)'],
            lw=0.5, marker='s', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # EC SST
        axs[irow, jcol].scatter(
            x = lig_recs['EC']['SO_jfm'].Longitude,
            y = lig_recs['EC']['SO_jfm'].Latitude,
            c = lig_recs['EC']['SO_jfm']['127 ka Median PIAn [°C]'],
            s = max_size - scale_size * lig_recs['EC']['SO_jfm']['127 ka 2s PIAn [°C]'],
            lw=0.5, marker='o', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # MC
        axs[irow, jcol].scatter(
            x = lig_recs['MC']['interpolated'].Longitude,
            y = lig_recs['MC']['interpolated'].Latitude,
            c = lig_recs['MC']['interpolated']['sst_anom_hadisst_jfm'],
            s = max_size - scale_size * 1.09,
            lw=0.5, marker='^', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        
        # DC
        plt_scatter = axs[irow, jcol].scatter(
            x = lig_recs['DC']['JFM_128'].Longitude,
            y = lig_recs['DC']['JFM_128'].Latitude,
            c = lig_recs['DC']['JFM_128']['sst_anom_hadisst_jfm'],
            s = max_size - scale_size * 1,
            lw=0.5, marker='v', edgecolors = 'black', zorder=2,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

for irow in range(nrow):
    for jcol in range(ncol):
        # irow = 0
        # jcol = 0
        model = models[jcol + ncol * irow]
        print(model)
        
        sm_lig_pi = lig_pi_sst_regrid_alltime[model]['sm'][0].values
        # sea_lig = lig_sst_regrid_alltime[model]['sea'][0::4].values
        # sea_pi = pi_sst_regrid_alltime[model]['sea'][0::4].values
        
        # ttest_fdr_res = ttest_fdr_control(sea_lig, sea_pi,)
        
        plt_mesh = axs[irow, jcol].pcolormesh(
            lon, lat, sm_lig_pi,
            norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
        # axs[irow, jcol].scatter(
        #     x=lon[(ttest_fdr_res == False) & np.isfinite(sm_lig_pi)],
        #     y=lat[(ttest_fdr_res == False) & np.isfinite(sm_lig_pi)],
        #     s=0.3, c='k', marker='.', edgecolors='none',
        #     transform=ccrs.PlateCarree(),
        #     )
        
        rmse = {}
        for irec in ['EC', 'JH', 'DC', 'MC']:
            # irec = 'EC'
            print(irec)
            
            rmse[irec] = np.round(SO_jfm_sst_site_values[irec].loc[
                SO_jfm_sst_site_values[irec].Model == model
                ]['sim_rec_jfm_sst_lig_pi'].mean(), 1)
        
        plt.text(
            0.5, 1.05,
            model + ': ' + \
                str(rmse['EC']) + ', ' + \
                    str(rmse['JH']) + ', ' + \
                        str(rmse['DC']) + ', ' + \
                            str(rmse['MC']),
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_scatter, ax=axs, aspect=40, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.4),)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region 1979-2014 ERA5 monthly averaged data on single levels from 1979 to present

# retrieval time: 2022-02-24
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'total_precipitation',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    '/home/users/qino/bas_palaeoclim_qino/scratch/cmip6/hist/pre/ERA5_mon_sl_197901_201412_pre.nc')



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region deprecated 1979-2021 ERA5 monthly averaged data on single levels from 1979 to present
# folder: mon_sl_79_present
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview

import cdsapi
c = cdsapi.Client()

# file: era5_mon_sl_79_21_pre.nc
# Monthly mean Total precipitation from 1979 to 2021
# retrieval time: 2021-11-06
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'total_precipitation',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_pre.nc')


# file: era5_mon_sl_79_21_slp.nc
# Monthly mean sea level pressure from 1979 to 2021
# retrieval time: 2021-11-06
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'mean_sea_level_pressure',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_slp.nc')


# file: era5_mon_sl_79_21_2mtem.nc
# Monthly mean 2m temperature from 1979 to 2021
# retrieval time: 2021-11-06
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': '2m_temperature',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_2mtem.nc')


# file: era5_mon_sl_79_21_sic.nc
# Monthly mean sea ice cover from 1979 to 2021
# retrieval time: 2021-11-06
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'sea_ice_cover',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sic.nc')


# file: era5_mon_sl_20_gph.nc
# Monthly mean geopotential height in 2020
# retrieval time: 2021-11-08
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'variable': 'geopotential',
        'year': '2020',
        'product_type': 'monthly_averaged_reanalysis',
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_20_gph.nc')


# file: era5_mon_sl_79_21_wind.nc
# Monthly mean 10m u, 10m v, 10m wind speed from 1979 to 2021
# retrieval time: 2021-12-04
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed',
        ],
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_10mwind.nc')


# file: era5_mon_sl_79_21_sst.nc
# Monthly mean Sea surface temperature from 1979 to 2021
# retrieval time: 2021-12-04
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'sea_surface_temperature',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_sst.nc')


# file: era5_mon_sl_79_21_evp.nc
# Monthly mean Evaporation from 1979 to 2021
# retrieval time: 2021-12-04
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'evaporation',
        'year': [
            '1979', '1980', '1981',
            '1982', '1983', '1984',
            '1985', '1986', '1987',
            '1988', '1989', '1990',
            '1991', '1992', '1993',
            '1994', '1995', '1996',
            '1997', '1998', '1999',
            '2000', '2001', '2002',
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_evp.nc')


# file: era5_mon_sl_79_21_lsmask.nc
# Land Sea Mask on 2020-01
# retrieval time: 2021-12-04
c.retrieve(
    'reanalysis-era5-single-levels-monthly-means',
    {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'land_sea_mask',
        'year': '2020',
        'month': '01',
        'time': '00:00',
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/mon_sl_79_present/era5_mon_sl_79_21_lsmask.nc')

# endregion
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region deprecated 1979-2021 ERA5 hourly data on single levels from 1979 to present
# folder: hr_sl_79_present
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=overview

import cdsapi
c = cdsapi.Client()

# file: era5_hr_sl_201412_10uv_slp_sst.nc
# Hourly 10m u, 10m v, SST, SLP in 2014
# retrieval time: 2021-12-05
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure',
            'sea_surface_temperature',
        ],
        'year': '2014',
        'month': '12',
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
    },
    'bas_palaeoclim_qino/observations/reanalysis/ERA5/hr_sl_79_present/era5_hr_sl_201412_10uv_slp_sst.nc')


# endregion
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# region plot reconstructions of am sst/sat

with open('scratch/cmip6/lig/sst/sst_regrid_alltime_ens_stats.pkl', 'rb') as f:
    sst_regrid_alltime_ens_stats = pickle.load(f)

with open('scratch/cmip6/lig/tas/tas_regrid_alltime_ens_stats.pkl', 'rb') as f:
    tas_regrid_alltime_ens_stats = pickle.load(f)

lon = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'].lon
lat = sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'].lat
# sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0]


output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_rec/7.0.3.0 rec am sst lig-pi.png'
cbar_label = 'LIG annual SST/SAT anomalies [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-5, cm_max=5, cm_interval1=0.5, cm_interval2=1, cmap='BrBG',)

max_size = 20
scale_size = 4

fig, ax = hemisphere_plot(northextent=-38, loceanarcs=False,)

ax.pcolormesh(
    lon,
    lat,
    tas_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

ax.pcolormesh(
    lon,
    lat,
    sst_regrid_alltime_ens_stats['lig_pi']['am']['mean'][0],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(), zorder=1)

# JH
ax.scatter(
    x = lig_recs['JH']['SO_ann'].Longitude,
    y = lig_recs['JH']['SO_ann'].Latitude,
    c = lig_recs['JH']['SO_ann']['127 ka SST anomaly (°C)'],
    s = max_size - scale_size * lig_recs['JH']['SO_ann']['127 ka 2σ (°C)'],
    lw=0.5, marker='s', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SST
ax.scatter(
    x = lig_recs['EC']['SO_ann'].Longitude,
    y = lig_recs['EC']['SO_ann'].Latitude,
    c = lig_recs['EC']['SO_ann']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['SO_ann']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# EC SAT
ax.scatter(
    x = lig_recs['EC']['AIS_am'].Longitude,
    y = lig_recs['EC']['AIS_am'].Latitude,
    c = lig_recs['EC']['AIS_am']['127 ka Median PIAn [°C]'],
    s = max_size - scale_size * lig_recs['EC']['AIS_am']['127 ka 2s PIAn [°C]'],
    lw=0.5, marker='o', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# DC
plt_scatter = ax.scatter(
    x = lig_recs['DC']['annual_128'].Longitude,
    y = lig_recs['DC']['annual_128'].Latitude,
    c = lig_recs['DC']['annual_128']['sst_anom_hadisst_ann'],
    s = max_size - scale_size * 1,
    lw=0.5, marker='v', edgecolors = 'black', zorder=2,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)


l1 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 1,
    lw=0.5, edgecolors = 'black',)
l2 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 2,
    lw=0.5, edgecolors = 'black',)
l3 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 3,
    lw=0.5, edgecolors = 'black',)
l4 = plt.scatter(
    [],[], c='white', marker='s', s=max_size - scale_size * 4,
    lw=0.5, edgecolors = 'black',)
plt.legend(
    [l1, l2, l3, l4,], ['1', '2', '3', '4 $°C$'], ncol=4, frameon=False,
    loc = (0.1, -0.35), handletextpad=0.05, columnspacing=0.3,)

cbar = fig.colorbar(
    plt_scatter, ax=ax, aspect=30,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.22, format=remove_trailing_zero_pos,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
fig.savefig(output_png)


'''
np.max(jh_sst_rec['SO_ann']['127 ka 2σ (°C)']) - np.min(jh_sst_rec['SO_ann']['127 ka 2σ (°C)'])
np.max(ec_sst_rec['SO_ann']['127 ka 2s PIAn [°C]']) - np.min(ec_sst_rec['SO_ann']['127 ka 2s PIAn [°C]'])

sns.scatterplot(
    x = lig_datasets.Latitude, y = lig_datasets.Longitude,
    size = lig_datasets['two-sigma errors [°C]'],
    style = lig_datasets['Dataset'],
    transform=ccrs.PlateCarree(),
    )


, figsize=np.array([5.8, 7]) / 2.54
\nReconstruction from Capron et al. 2017
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot site locations

output_png = 'figures/test/trial.pdf'
fig, ax = hemisphere_plot(northextent=-38, loceanarcs=True)

# output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_site_names/7.0.3 am sst site names.pdf'
# for isite in range(len(jh_sst_rec['SO_ann'].Longitude)):
#     ax.text(
#         jh_sst_rec['SO_ann'].Longitude.iloc[isite],
#         jh_sst_rec['SO_ann'].Latitude.iloc[isite],
#         jh_sst_rec['SO_ann'].Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     jh_sst_rec['SO_ann'].Longitude, jh_sst_rec['SO_ann'].Latitude, ax, s=10,
#     marker='s', )

# for isite in range(len(ec_sst_rec['SO_ann'].Longitude)):
#     ax.text(
#         ec_sst_rec['SO_ann'].Longitude.iloc[isite],
#         ec_sst_rec['SO_ann'].Latitude.iloc[isite],
#         ec_sst_rec['SO_ann'].Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     ec_sst_rec['SO_ann'].Longitude, ec_sst_rec['SO_ann'].Latitude, ax, s=10,
#     marker='o', )

# output_png = 'figures/7_lig/7.0_sim_rec/7.0.3_site_names/7.0.3 summer sst site names.pdf'
# for isite in range(len(jh_sst_rec['SO_djf'].Longitude)):
#     ax.text(
#         jh_sst_rec['SO_djf'].Longitude.iloc[isite],
#         jh_sst_rec['SO_djf'].Latitude.iloc[isite],
#         jh_sst_rec['SO_djf'].Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     jh_sst_rec['SO_djf'].Longitude, jh_sst_rec['SO_djf'].Latitude, ax, s=10,
#     marker='s', )

# for isite in range(len(ec_sst_rec['SO_djf'].Longitude)):
#     ax.text(
#         ec_sst_rec['SO_djf'].Longitude.iloc[isite],
#         ec_sst_rec['SO_djf'].Latitude.iloc[isite],
#         ec_sst_rec['SO_djf'].Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     ec_sst_rec['SO_djf'].Longitude, ec_sst_rec['SO_djf'].Latitude, ax, s=10,
#     marker='o', )

for isite in range(len(chadwick_interp.lon)):
    ax.text(
        chadwick_interp.lon.iloc[isite],
        chadwick_interp.lat.iloc[isite],
        chadwick_interp.sites.iloc[isite],
        transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
cplot_ice_cores(
    chadwick_interp.lon, chadwick_interp.lat, ax, s=10,
    marker='^', )


fig.savefig(output_png)


'''
# lig_datasets_jh = lig_datasets.loc[
#     lig_datasets.Dataset == 'Hoffman et al. (2017)']
# for isite in range(len(lig_datasets_jh.Longitude)):
#     ax.text(
#         lig_datasets_jh.Longitude.iloc[isite],
#         lig_datasets_jh.Latitude.iloc[isite],
#         lig_datasets_jh.Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     lig_datasets_jh.Longitude, lig_datasets_jh.Latitude, ax, s=10,
#     marker='s', )

# lig_datasets_mc = lig_datasets.loc[
#     lig_datasets.Dataset == 'Chadwick et al. (2021)']
# for isite in range(len(lig_datasets_mc.Longitude)):
#     ax.text(
#         lig_datasets_mc.Longitude.iloc[isite],
#         lig_datasets_mc.Latitude.iloc[isite],
#         lig_datasets_mc.Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     lig_datasets_mc.Longitude, lig_datasets_mc.Latitude, ax, s=10,
#     marker='^', )

# lig_datasets_ec = lig_datasets.loc[
#     lig_datasets.Dataset == 'Capron et al. (2017)']
# for isite in range(len(lig_datasets_ec.Longitude)):
#     ax.text(
#         lig_datasets_ec.Longitude.iloc[isite],
#         lig_datasets_ec.Latitude.iloc[isite],
#         lig_datasets_ec.Station.iloc[isite],
#         transform=ccrs.PlateCarree(), c='gray', fontsize=6,)
# cplot_ice_cores(
#     lig_datasets_ec.Longitude, lig_datasets_ec.Latitude, ax, s=10,
#     marker='o', )

double station entries: ODP-1089, MD88-770
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region extract indices for WOA2018

woa18_sst = {}
woa18_sst['ann'] = xr.open_dataset(
    'data_sources/LIG/WOA_2018/woa18_decav_t00_01.nc', decode_times=False)
woa18_sst['JFM'] = xr.open_dataset(
    'data_sources/LIG/WOA_2018/woa18_decav_t13_01.nc', decode_times=False)
woa18_sst['JAS'] = xr.open_dataset(
    'data_sources/LIG/WOA_2018/woa18_decav_t15_01.nc', decode_times=False)








# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check difference between DJF and JFM

with open('scratch/cmip6/lig/sst/lig_sst_regrid.pkl', 'rb') as f:
    lig_sst_regrid = pickle.load(f)

model = 'AWI-ESM-1-1-LR'

lig_sst_regrid_djf = mon_sea_ann(var_monthly=lig_sst_regrid[model].tos)

lig_sst_regrid_jfm = mon_sea_ann(var_monthly=lig_sst_regrid[model].tos,
                                 seasons = 'Q-MAR',)

data1 = lig_sst_regrid_djf['sm'].sel(season='DJF')
data2 = lig_sst_regrid_jfm['sm'].sel(month=3)
np.max((data1 - data2.values).isel(y=slice(0, 50)))
np.mean(abs((data1 - data2.values).isel(y=slice(0, 50))))
(data1 - data2.values).to_netcdf('scratch/test/test0.nc')

data1.isel(y=slice(0, 50))

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded AMIP SIC

amip_pi_sic = xr.open_dataset(
    'startdump/model_input/pi/alex/T63_amipsic_pcmdi_187001-189912.nc')

# set land points as np.nan
esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
b_slm = np.broadcast_to(
    np.isnan(esacci_echam6_t63_trim.analysed_sst.values),
    amip_pi_sic.sic.shape,)
amip_pi_sic.sic.values[b_slm] = np.nan

amip_pi_sic_regrid = {}
amip_pi_sic_regrid['mm'] = regrid(amip_pi_sic.sic)
amip_pi_sic_regrid['am'] = time_weighted_mean(amip_pi_sic_regrid['mm'])
amip_pi_sic_regrid['sm'] = amip_pi_sic_regrid['mm'].groupby(
    'time.season').map(time_weighted_mean)

with open('scratch/cmip6/lig/amip_pi_sic_regrid.pkl', 'wb') as f:
    pickle.dump(amip_pi_sic_regrid, f)


'''
with open('scratch/cmip6/lig/amip_pi_sic_regrid.pkl', 'rb') as f:
    amip_pi_sic_regrid = pickle.load(f)
amip_pi_sic_regrid['am']
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann sic

with open('scratch/cmip6/lig/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)

models=sorted(lig_sic.keys())

lig_sic_alltime = {}
pi_sic_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    lig_sic_alltime[model] = mon_sea_ann(var_monthly = lig_sic[model].siconc)
    pi_sic_alltime[model] = mon_sea_ann(var_monthly = pi_sic[model].siconc)

with open('scratch/cmip6/lig/lig_sic_alltime.pkl', 'wb') as f:
    pickle.dump(lig_sic_alltime, f)
with open('scratch/cmip6/lig/pi_sic_alltime.pkl', 'wb') as f:
    pickle.dump(pi_sic_alltime, f)


'''
with open('scratch/cmip6/lig/lig_sic.pkl', 'rb') as f:
    lig_sic = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic.pkl', 'rb') as f:
    pi_sic = pickle.load(f)

with open('scratch/cmip6/lig/lig_sic_alltime.pkl', 'rb') as f:
    lig_sic_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sic_alltime.pkl', 'rb') as f:
    pi_sic_alltime = pickle.load(f)

models=sorted(lig_sic.keys())

#-------------------------------- check monthly values
for model in models:
    # model = 'NorESM2-LM'
    print('#-------- ' + model)
    data1 = lig_sic[model].siconc.values
    data2 = lig_sic_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

    data1 = pi_sic[model].siconc.values
    data2 = pi_sic_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


#-------------------------------- check one model
with open('scratch/cmip6/lig/lig_sic_alltime.pkl', 'rb') as f:
    lig_sic_alltime = pickle.load(f)

model = 'NorESM2-LM'
files = glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/siconc_SImon_'+model+'_lig127k_*.nc')
ds=xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
ds_pp=combined_preprocessing(ds.isel(time=slice(-2400,None)))

data1 = ds_pp.siconc.values
data2 = lig_sic_alltime['NorESM2-LM']['mon'].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann tas

with open('scratch/cmip6/lig/lig_tas.pkl', 'rb') as f:
    lig_tas = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas.pkl', 'rb') as f:
    pi_tas = pickle.load(f)

models=sorted(lig_tas.keys())


lig_tas_alltime = {}
pi_tas_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    lig_tas_alltime[model] = mon_sea_ann(var_monthly = lig_tas[model].tas)
    pi_tas_alltime[model] = mon_sea_ann(var_monthly = pi_tas[model].tas)

with open('scratch/cmip6/lig/lig_tas_alltime.pkl', 'wb') as f:
    pickle.dump(lig_tas_alltime, f)
with open('scratch/cmip6/lig/pi_tas_alltime.pkl', 'wb') as f:
    pickle.dump(pi_tas_alltime, f)


'''
for model in models:
    print('#------------------------' + model)
    print(lig_tas[model].lon.shape)
    print(lig_tas[model].lat.shape)
    print(lig_tas[model].tas.shape)


with open('scratch/cmip6/lig/lig_tas.pkl', 'rb') as f:
    lig_tas = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas.pkl', 'rb') as f:
    pi_tas = pickle.load(f)

with open('scratch/cmip6/lig/lig_tas_alltime.pkl', 'rb') as f:
    lig_tas_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_tas_alltime.pkl', 'rb') as f:
    pi_tas_alltime = pickle.load(f)

models=sorted(lig_tas.keys())

#---- check monthly values
for model in models:
    # model = 'NorESM2-LM'
    print('#-------- ' + model)
    data1 = lig_tas[model].tas.values
    data2 = lig_tas_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

    data1 = pi_tas[model].tas.values
    data2 = pi_tas_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())


#---- check 'AWI-ESM-1-1-LR'
ds = xr.open_mfdataset(
    '/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/atmos/tas/tas_Amon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_*.nc', use_cftime=True,parallel=True
)

with open('scratch/cmip6/lig/lig_tas_alltime.pkl', 'rb') as f:
    lig_tas_alltime = pickle.load(f)

data1 = ds.tas.values
data2 = lig_tas_alltime['AWI-ESM-1-1-LR']['mon'].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get regridded AMIP SST

pi_sst_file = 'startdump/model_input/pi/alex/T63_amipsst_pcmdi_187001-189912.nc'
boundary_conditions = {}
boundary_conditions['sst'] = {}
boundary_conditions['sst']['pi'] = xr.open_dataset(pi_sst_file)

# set land points as np.nan
esacci_echam6_t63_trim = xr.open_dataset('startdump/tagging/tagmap/auxiliaries/sst_mon_ESACCI-2.1_198201_201612_am_rg_echam6_t63_slm_trim.nc')
b_slm = np.broadcast_to(
    np.isnan(esacci_echam6_t63_trim.analysed_sst.values),
    boundary_conditions['sst']['pi'].sst.shape,)
boundary_conditions['sst']['pi'].sst.values[b_slm] = np.nan
boundary_conditions['sst']['pi'].sst.values -= zerok

amip_pi_sst_regrid = {}
amip_pi_sst_regrid['mm'] = regrid(boundary_conditions['sst']['pi'].sst)
amip_pi_sst_regrid['am'] = time_weighted_mean(amip_pi_sst_regrid['mm'])
amip_pi_sst_regrid['sm'] = amip_pi_sst_regrid['mm'].groupby(
    'time.season').map(time_weighted_mean)

with open('scratch/cmip6/lig/amip_pi_sst_regrid.pkl', 'wb') as f:
    pickle.dump(amip_pi_sst_regrid, f)

'''
with open('scratch/cmip6/lig/amip_pi_sst_regrid.pkl', 'rb') as f:
    amip_pi_sst_regrid = pickle.load(f)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get mon_sea_ann sst

with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

models=sorted(lig_sst.keys())

lig_sst_alltime = {}
pi_sst_alltime = {}

for model in models:
    # model = 'NorESM2-LM'
    print(model)
    lig_sst_alltime[model] = mon_sea_ann(var_monthly = lig_sst[model].tos)
    pi_sst_alltime[model] = mon_sea_ann(var_monthly = pi_sst[model].tos)

with open('scratch/cmip6/lig/lig_sst_alltime.pkl', 'wb') as f:
    pickle.dump(lig_sst_alltime, f)
with open('scratch/cmip6/lig/pi_sst_alltime.pkl', 'wb') as f:
    pickle.dump(pi_sst_alltime, f)


'''
with open('scratch/cmip6/lig/lig_sst.pkl', 'rb') as f:
    lig_sst = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst.pkl', 'rb') as f:
    pi_sst = pickle.load(f)

with open('scratch/cmip6/lig/lig_sst_alltime.pkl', 'rb') as f:
    lig_sst_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst_alltime.pkl', 'rb') as f:
    pi_sst_alltime = pickle.load(f)

models=sorted(lig_sst.keys())

#---- check monthly values
for model in models:
    # model = 'NorESM2-LM'
    print('#-------- ' + model)
    data1 = lig_sst[model].tos.values
    data2 = lig_sst_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

    data1 = pi_sst[model].tos.values
    data2 = pi_sst_alltime[model]['mon'].values
    print((data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all())

#---- check storage
for model in models:
    # model = 'NorESM2-LM'
    print('#-------- ' + model)
    print(lig_sst[model].tos.nbytes / 2**30)
    
    # print(lig_sst_alltime[model].keys())
    
    for ialltime in lig_sst_alltime[model].keys():
        print(lig_sst_alltime[model][ialltime].nbytes / 2**30)

#---- check 'NorESM2-LM'
with open('scratch/cmip6/lig/lig_sst_alltime.pkl', 'rb') as f:
    lig_sst_alltime = pickle.load(f)

files=glob.glob('/gws/nopw/j04/bas_palaeoclim/rahul/data/PMIP_LIG/ESGF_download/CMIP6/model-output/ocean/tos/tos_Omon_NorESM2-LM_lig127k_*.nc')
ds=xr.open_mfdataset(paths=files,use_cftime=True,parallel=True)
ds_pp=combined_preprocessing(ds.isel(time=slice(-2400,None)))

data1 = ds_pp.tos.values
data2 = lig_sst_alltime['NorESM2-LM']['mon'].values
(data1[np.isfinite(data1)] == data2[np.isfinite(data2)]).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region compare three quantiles [90%, 95%, 99%]

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1 ' + expid[i] + ' compare quantiles frc_source_lat Antarctica.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=0, cm_max=50, cm_interval1=5, cm_interval2=5, cmap='PuOr',
    reversed=False)

pltlevel2 = np.arange(0, 10 + 1e-4, 1)
pltticks2 = np.arange(0, 10 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

nrow = 2
ncol = 3

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.01, 'wspace': 0.01},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            northextent=-50, ax_org = axs[irow, jcol])
        cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat,
                        axs[irow, jcol])
        plt.text(
            0.05, 1, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1

# Contribution to total precipitation
for icount,iqtl in enumerate(quantiles.keys()):
    plt1 = axs[0, icount].pcolormesh(
        lon,
        lat,
        wisoaprt_epe[expid[i]]['frc_aprt']['am'][iqtl] * 100,
        norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)
    
    plt2 = axs[1, icount].pcolormesh(
        lon,
        lat,
        epe_weighted_lat[expid[i]][iqtl]['am'] - pre_weighted_lat[expid[i]]['am'],
        norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)
    ttest_fdr_res = ttest_fdr_control(
        epe_weighted_lat[expid[i]][iqtl]['ann'],
        pre_weighted_lat[expid[i]]['ann'],)
    axs[1, icount].scatter(
        x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
        s=0.5, c='k', marker='.', edgecolors='none',
        transform=ccrs.PlateCarree(),)

for icount,iqtl in enumerate(quantiles.keys()):
    plt.text(
        0.5, 1.05, iqtl,
        transform=axs[0, icount].transAxes,
        ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(-0.2, -0.3), ticks=pltticks)
cbar1.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar1.ax.set_xlabel('Contribution to total precipitation [$\%$]', linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(1.1,-3.8),ticks=pltticks2)
cbar2.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar2.ax.set_xlabel('EPE source latitude anomalies [$°$]', linespacing=2)


fig.subplots_adjust(left=0.01, right = 0.99, bottom = 0.12, top = 0.96)
fig.savefig(output_png)



'''
plt.text(
    -0.05, 0.5, 'Contribution to total precipitation [$\%$]',
    transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')

plt.text(
    -0.05, 0.5, 'EPE source latitude anomalies [$°$]',
    transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot aprt_frc from geo7
stats.describe(
    aprt_frc['NHseaice']['am'] + aprt_frc['SHseaice']['am'] + aprt_frc['NHocean']['am'] + aprt_frc['SHocean']['am'] + aprt_frc['Antarctica']['am'] + aprt_frc['NHland']['am'] + aprt_frc['SHland']['am'],
    axis=None, nan_policy='omit')

#-------- precipitation from geo regions


np.max(aprt_frc['NHseaice']['am'].sel(lat=slice(-50, -90))) # 3e-5
np.max(aprt_frc['NHocean']['am'].sel(lat=slice(-50, -90))) # 0.88
np.max(aprt_frc['SHocean']['am'].sel(lat=slice(-50, -90))) # 98.9
np.max(aprt_frc['SHseaice']['am'].sel(lat=slice(-50, -90))) # 42.5
np.max(aprt_frc['Antarctica']['am'].sel(lat=slice(-50, -90))) # 19.4
np.max(aprt_frc['NHland']['am'].sel(lat=slice(-50, -90))) # 0.25
np.max(aprt_frc['SHland']['am'].sel(lat=slice(-50, -90))) # 16.2

#-------- precipitation from Antarctica

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.0_aprt_frc/6.1.4.0.0 ' + expid[i] + ' aprt_frc am Antarctica.png'

pltlevel = np.arange(0, 10.01, 1)
pltticks = np.arange(0, 10.01, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-50)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)
plt_cmp = ax.pcolormesh(
    lon, lat,
    aprt_frc['Antarctica']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nAntarctica [%]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)

#-------- SH sea ice

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.0_aprt_frc/6.1.4.0.0 ' + expid[i] + ' aprt_frc am SHseaice.png'

pltlevel = np.arange(0, 40.01, 5)
pltticks = np.arange(0, 40.01, 5)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Purples', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-50)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)
plt_cmp = ax.pcolormesh(
    lon, lat,
    aprt_frc['SHseaice']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nSH sea ice covered area [%]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)

#-------- other Land

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.0_aprt_frc/6.1.4.0.0 ' + expid[i] + ' aprt_frc am Otherland.png'

pltlevel = np.arange(0, 10.01, 1)
pltticks = np.arange(0, 10.01, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-50)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)
plt_cmp = ax.pcolormesh(
    lon, lat,
    aprt_frc['Otherland']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nland excl. Antarctica [%]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)


#-------- other Ocean

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.0_aprt_frc/6.1.4.0.0 ' + expid[i] + ' aprt_frc am Otherocean.png'

pltlevel = np.arange(60, 100.01, 4)
pltticks = np.arange(60, 100.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Greens', len(pltlevel)-1)

fig, ax = hemisphere_plot(northextent=-50)
cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)
plt_cmp = ax.pcolormesh(
    lon, lat,
    aprt_frc['Otherocean']['am'],
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='min',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel('Fraction of annual mean precipitation from\nOpen ocean [%]', linespacing=1.5, fontsize=8)
fig.savefig(output_png)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot 95% daily precipitation

iqtl = '95%'

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1 ' + expid[i] + ' daily precipitation percentile_' + iqtl[:2] + ' Antarctica.png'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

fig, ax = hemisphere_plot(
    northextent=-50, figsize=np.array([5.8, 7.8]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    wisoaprt_epe[expid[i]]['quantiles']['95%'] * seconds_per_d,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.2,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    iqtl + ' quantile of\ndaily precipitation [$mm \; day^{-1}$]', linespacing=1.5)
fig.savefig(output_png, dpi=1200)

'''
wisoaprt_epe[expid[i]]['frc_aprt']['am']['99%'].to_netcdf('scratch/test/test.nc')

#---- check fraction of real pre to total pre
tot_pre = (wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1) * seconds_per_d).values
sum_pre = (wisoaprt_epe[expid[i]]['sum_aprt']['am']['original'] / len(wisoaprt_alltime[expid[i]]['daily'].time) * seconds_per_d).values

np.max(abs((tot_pre - sum_pre) / tot_pre))

diff = (tot_pre - sum_pre) / tot_pre
where_max = np.where(abs(diff) == np.max(abs(diff)))
tot_pre[where_max]
sum_pre[where_max]
(tot_pre[where_max] - sum_pre[where_max]) / tot_pre[where_max]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot annual mean precipitation

output_png = 'figures/6_awi/6.1_echam6/6.1.7_epe/6.1.7.1_pre/6.1.7.1 ' + expid[i] + ' aprt am Antarctica.png'

pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14])
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)

fig, ax = hemisphere_plot(
    northextent=-50, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    wisoaprt_alltime[expid[i]]['am'].sel(wisotype=1) * seconds_per_d,
    norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30, format=remove_trailing_zero_pos,
    orientation="horizontal", shrink=1, ticks=pltticks, extend='max',
    pad=0.02, fraction=0.15,
    )
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel(
    'Annual mean precipitation [$mm \; day^{-1}$]', linespacing=1.5)
fig.savefig(output_png, dpi=1200)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region get wisoaprt_epe at ice core sites

wisoaprt_epe = {}
with open(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.wisoaprt_epe.pkl',
    'rb') as f:
    wisoaprt_epe[expid[i]] = pickle.load(f)

quantiles = {'90%': 0.9}

# wisoaprt_epe[expid[i]]['frc_aprt']['am']['90%']

wisoaprt_epe_alltime_icores = {}
wisoaprt_epe_alltime_icores[expid[i]] = {}

for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#--------' + icores)
    wisoaprt_epe_alltime_icores[expid[i]][icores] = {}
    
    for ialltime in wisoaprt_epe[expid[i]]['frc_aprt'].keys():
        print('#----' + ialltime)
        wisoaprt_epe_alltime_icores[expid[i]][icores][ialltime] = {}
        
        for iqtl in quantiles.keys():
            print('#--' + iqtl)
            
            if ialltime in ['mon', 'sea', 'ann', 'mm', 'sm']:
                # ialltime = 'mon'
                wisoaprt_epe_alltime_icores[expid[i]][icores][ialltime][iqtl] = \
                    wisoaprt_epe[expid[i]]['frc_aprt'][ialltime][iqtl][
                        :,
                        t63_sites_indices[icores]['ilat'],
                        t63_sites_indices[icores]['ilon']]
            elif (ialltime == 'am'):
                wisoaprt_epe_alltime_icores[expid[i]][icores][ialltime][iqtl] = \
                    wisoaprt_epe[expid[i]]['frc_aprt'][ialltime][iqtl][
                        t63_sites_indices[icores]['ilat'],
                        t63_sites_indices[icores]['ilon']]


with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_epe_alltime_icores.pkl',
    'wb') as f:
    pickle.dump(wisoaprt_epe_alltime_icores[expid[i]], f)




'''
iqtl = '90%'
for icores in stations_sites.Site:
    # icores = 'EDC'
    print('#----------------' + icores)
    wisoaprt_epe_am_icores = \
        np.round(
            wisoaprt_epe_alltime_icores[expid[i]][icores][
                'am'][iqtl].values * 100, 1)
    wisoaprt_epe_annstd_icores = \
        np.round(
            (wisoaprt_epe_alltime_icores[expid[i]][icores][
                'ann'][iqtl] * 100).std(ddof=1).values,
            1)
    print(str(wisoaprt_epe_am_icores) + ' ± ' + str(wisoaprt_epe_annstd_icores))

#-------------------------------- check
wisoaprt_epe_alltime_icores = {}
with open(
    exp_odir + expid[i] + '/analysis/jsbach/' + expid[i] + '.wisoaprt_epe_alltime_icores.pkl', 'rb') as f:
    wisoaprt_epe_alltime_icores[expid[i]] = pickle.load(f)

for icores in wisoaprt_epe_alltime_icores[expid[i]].keys():
    # icores = 'EDC'
    print('#----------------' + icores)
    print('local lat:  ' + str(t63_sites_indices[icores]['lat']))
    print('grid lat:   ' + \
        str(np.round(wisoaprt_epe_alltime_icores[expid[i]][icores]['am'].lat.values, 2)))
    print('local lon:  ' + str(t63_sites_indices[icores]['lon']))
    print('grid lon:   ' + \
        str(wisoaprt_epe_alltime_icores[expid[i]][icores]['am'].lon.values))

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region check negative evaporation with tagmap

i = 0

evaplac = exp_org_o[expid[i]]['echam'].evaplac.values
evaplac[evaplac < 0] = 0
evapiac = exp_org_o[expid[i]]['echam'].evapiac.values
evapiac[evapiac < 0] = 0
echam_evap = evaplac + evapiac

iwisotype = 4
wisoevaplac = exp_org_o[expid[i]]['wiso'].wisoevaplac.sel(wisotype=iwisotype)



echam_evap = (exp_org_o[expid[i]]['echam'].evaplac + exp_org_o[expid[i]]['echam'].evapiac).values[1:]

wiso_evap = exp_org_o[expid[i]]['wiso'].wisoevap.sel(wisotype=4).values[1:]
np.max(abs(echam_evap[echam_evap < 0] - wiso_evap[echam_evap < 0]))
stats.describe(abs(echam_evap - wiso_evap), axis=None)

np.min(exp_org_o[expid[i]]['wiso'].wisoevapwac.sel(wisotype=4).values)


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
# region plot regridded am sst Antarctica

with open('scratch/cmip6/lig/amip_pi_sst_regrid.pkl', 'rb') as f:
    amip_pi_sst_regrid = pickle.load(f)

output_png = 'figures/0_test/trial.png'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=26, cm_interval1=2, cm_interval2=4, cmap='RdBu',)

fig, ax = hemisphere_plot(northextent=-20, figsize=np.array([5.8, 7.3]) / 2.54,)

plt_mesh1 = ax.pcolormesh(
    amip_pi_sst_regrid['am'].lon,
    amip_pi_sst_regrid['am'].lat,
    amip_pi_sst_regrid['am'].sst.values - zerok,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_mesh1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.15,
    )
cbar.ax.set_xlabel('Sea surface temperature (SST) [$°C$]', linespacing=1.5,)
fig.savefig(output_png)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot lig/pi regridded am sst

with open('scratch/cmip6/lig/lig_sst_regrid_alltime.pkl', 'rb') as f:
    lig_sst_regrid_alltime = pickle.load(f)
with open('scratch/cmip6/lig/pi_sst_regrid_alltime.pkl', 'rb') as f:
    pi_sst_regrid_alltime = pickle.load(f)

output_png = 'figures/0_test/trial.png'
cbar_label = 'Annual mean SST [$°C$]'

pltlevel, pltticks, pltnorm, pltcmp = plt_mesh_pars(
    cm_min=-2, cm_max=26, cm_interval1=2, cm_interval2=2, cmap='RdBu',)

nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.12, 'wspace': 0.02},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(
            northextent=-30, ax_org = axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1

for irow in range(nrow):
    for jcol in range(ncol):
        model = models[jcol + ncol * irow]
        print(model)
        # model = 'GISS-E2-1-G'
        
        plt_mesh = axs[irow, jcol].pcolormesh(
            lig_sst_regrid_alltime[model]['am'].lon,
            lig_sst_regrid_alltime[model]['am'].lat,
            lig_sst_regrid_alltime[model]['am'].values,
            # pi_sst_regrid_alltime[model]['am'].lon,
            # pi_sst_regrid_alltime[model]['am'].lat,
            # pi_sst_regrid_alltime[model]['am'].values,
            norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
        
        plt.text(
            0.5, 1.05, model,
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')

cbar = fig.colorbar(
    plt_mesh, ax=axs, aspect=40,
    orientation="horizontal", shrink=0.75, ticks=pltticks, extend='both',
    anchor=(0.5, -0.3),
    )
cbar.ax.set_xlabel(cbar_label, linespacing=1.5)
cbar.ax.xaxis.set_minor_locator(AutoMinorLocator(1))
fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom, top = 0.97)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am aprt Antarctica

#-------- basic settings

pltctr1 = np.array([0.004, 0.008, 0.05, 0.1, 0.5, ])
pltctr2 = np.array([1, 2, 4, 8, ])

plt_data = wisoaprt_alltime[expid[i]]['am'][0].values * seconds_per_d

output_png = 'figures/6_awi/6.1_echam6/6.1.4_precipitation/6.1.4.0_aprt/6.1.4.0.2_spatiotemporal_dist/6.1.4.0.2 ' + expid[i] + ' am aprt Antarctica.png'

#-------- plot

fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 6.5]) / 2.54, lw=0.1,
    fm_bottom=0.1, fm_top=0.99)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt2 = ax.contour(
    lon, lat,
    plt_data,
    levels=pltctr1, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='dotted',
)
ax.clabel(plt2, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=pltctr1, inline_spacing=10, fontsize=7,)

plt3 = ax.contour(
    lon, lat,
    plt_data,
    levels=pltctr2, colors = 'b', transform=ccrs.PlateCarree(),
    linewidths=0.5, linestyles='solid',
)
ax.clabel(plt3, inline=1, colors='b', fmt=remove_trailing_zero,
          levels=pltctr2, inline_spacing=10, fontsize=7,)

plt.text(
    0.5, -0.08, 'Annual mean precipitation [$mm \; day^{-1}$]',
    transform=ax.transAxes, ha='center', va='center', rotation='horizontal')

fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot psl and u/v in ECHAM


uv_plev = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.uv_plev.pkl', 'rb') as f:
    uv_plev[expid[i]] = pickle.load(f)

psl_zh = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.psl_zh.pkl', 'rb') as f:
    psl_zh[expid[i]] = pickle.load(f)

moisture_flux = {}
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.moisture_flux.pkl', 'rb') as f:
    moisture_flux[expid[i]] = pickle.load(f)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.3_moisture_transport/' + '6.1.4.3 ' + expid[i] + ' am psl and 850hPa wind Antarctica.png'

plt_pres = psl_zh[expid[i]]['psl']['am'] / 100
pres_interval = 5
pres_intervals = np.arange(
    np.floor(np.min(plt_pres) / pres_interval - 1) * pres_interval,
    np.ceil(np.max(plt_pres) / pres_interval + 1) * pres_interval,
    pres_interval)

pltlevel = np.arange(-6, 6 + 1e-4, 0.5)
pltticks = np.arange(-6, 6 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PiYG', len(pltlevel)-1).reversed()


fig, ax = hemisphere_plot(
    northextent=-45,
    figsize=np.array([5.8, 8.8]) / 2.54,
    fm_bottom=0.13,
    )

plt_ctr = ax.contour(
    plt_pres.lon, plt_pres.lat, plt_pres,
    colors='b', levels=pres_intervals, linewidths=0.2,
    transform=ccrs.PlateCarree(), clip_on=True)
ax_clabel = ax.clabel(
    plt_ctr, inline=1, colors='b', fmt='%d',
    levels=pres_intervals, inline_spacing=10, fontsize=6)
h1, _ = plt_ctr.legend_elements()
ax_legend = ax.legend(
    [h1[0]], ['Mean sea level pressure [$hPa$]'],
    loc='lower center', frameon=False,
    bbox_to_anchor=(0.5, -0.14),
    handlelength=1, columnspacing=1)

# plot H/L symbols
plot_maxmin_points(
    plt_pres.lon, plt_pres.lat, plt_pres,
    ax, 'max', 150, symbol='H', color='b',
    transform=ccrs.PlateCarree(),)
plot_maxmin_points(
    plt_pres.lon, plt_pres.lat, plt_pres,
    ax, 'min', 150, symbol='L', color='r',
    transform=ccrs.PlateCarree(),)

# plot wind arrows
iarrow = 2
plt_quiver = ax.quiver(
    plt_pres.lon[::iarrow], plt_pres.lat[::iarrow],
    uv_plev[expid[i]]['u']['am'].sel(plev=85000).values[::iarrow, ::iarrow],
    uv_plev[expid[i]]['v']['am'].sel(plev=85000).values[::iarrow, ::iarrow],
    color='gray', units='height', scale=600,
    width=0.002, headwidth=3, headlength=5, alpha=1,
    transform=ccrs.PlateCarree(),)

ax.quiverkey(plt_quiver, X=0.15, Y=-0.14, U=10,
             label='10 [$m \; s^{-1}$]    850 $hPa$ wind',
             labelpos='E', labelsep=0.05,)

plt_mesh = ax.pcolormesh(
    moisture_flux[expid[i]]['meridional']['am'].lon,
    moisture_flux[expid[i]]['meridional']['am'].lat,
    moisture_flux[expid[i]]['meridional']['am'].sel(plev=85000) * 10**3,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),
    zorder = -2)

cbar1 = fig.colorbar(
    plt_mesh, ax=ax,
    fraction=0.1,
    orientation="horizontal",shrink=1,aspect=40,extend='both',
    anchor=(0.5, 0.9), ticks=pltticks)
cbar1.ax.set_xlabel('Meridional moisture flux at 850 $hPa$\n[$10^{-3} \; kg\;kg^{-1} \; m\;s^{-1}$]', linespacing=1.5)

fig.savefig(output_png)



'''
stats.describe(abs(moisture_flux[expid[i]]['meridional']['am'].sel(plev=85000, lat=slice(-60, -90)) * 10**3),
               axis=None, nan_policy='omit')

(np.isfinite(uv_plev[expid[i]]['u']['am'].sel(plev=85000)) == np.isfinite(uv_plev[expid[i]]['v']['am'].sel(plev=85000))).all()
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate mon_sea_ann psl in ERA5

psl_era5_79_14 = xr.open_dataset('scratch/cmip6/hist/psl/psl_ERA5_mon_sl_197901_201412.nc')

psl_era5_79_14_alltime = mon_sea_ann(var_monthly=psl_era5_79_14.msl)

with open('scratch/cmip6/hist/psl/psl_era5_79_14_alltime.pkl', 'wb') as f:
    pickle.dump(psl_era5_79_14_alltime, f)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot rel. pre_weighted_lon am

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/6.1.3.1 ' + expid[i] + ' pre_weighted_lon am Antarctica.png'

pltlevel = np.arange(-180, 180 + 1e-4, 15)
pltticks = np.arange(-180, 180 + 1e-4, 45)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('twilight_shifted', len(pltlevel)-1).reversed()

fig, ax = hemisphere_plot(northextent=-50, figsize=np.array([5.8, 7]) / 2.54)

cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, ax)

plt1 = ax.pcolormesh(
    lon,
    lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['am'], lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt1, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='neither',
    pad=0.02, fraction=0.15,
    )
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_xlabel('Relative source longitude [$°$]', linespacing=2)
fig.savefig(output_png, dpi=1200)


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot AIS boundaries, ice core sites, surface height


fig, ax = hemisphere_plot(
    northextent=-60, figsize=np.array([5.8, 5.8]) / 2.54,
    fm_bottom=0.01, lw=0.1)

plt_wais = ais_imbie2.loc[ais_imbie2.Regions == 'West'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='red', facecolor='none', linewidths=0.15, zorder=2)
plt_eais = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='blue', facecolor='none', linewidths=0.15, zorder=2)
plt_ap = ais_imbie2.loc[ais_imbie2.Regions == 'Peninsula'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='m', facecolor='none', linewidths=0.15, zorder=2)

ax.scatter(
    x = major_ice_core_site.lon, y = major_ice_core_site.lat,
    s=3, c='none', linewidths=0.5, marker='o',
    transform=ctp.crs.PlateCarree(), edgecolors = 'black',
    )

for irow in range(major_ice_core_site.shape[0]):
    # irow = 0
    ax.text(major_ice_core_site.lon[irow], major_ice_core_site.lat[irow]+1,
            major_ice_core_site.Site[irow], transform=ccrs.PlateCarree(),
            fontsize = 6, color='black')

fig.savefig('figures/1_study_area/trial1.png')





'''
# pltlevel = np.arange(0, 32.01, 0.1)
# pltticks = np.arange(0, 32.01, 4)
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
# pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

# plt_cmp = ax.pcolormesh(
#     x,
#     y,
#     z,
#     norm=pltnorm, cmap=pltcmp, transform=ccrs.PlateCarree(),)

# cbar = fig.colorbar(
#     cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
#     orientation="horizontal", shrink=0.9, ticks=pltticks, extend='max',
#     pad=0.02, fraction=0.2,
#     )
# cbar.ax.set_xlabel('1st line\n2nd line', linespacing=2)
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot AIS boundaries, ice core sites

# Load data
with open('bas_palaeoclim_qino/others/ais_masks.pickle', 'rb') as handle:
    ais_masks = pickle.load(handle)

one_degree_grids_cdo_area = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc')
ais_imbie2 = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp')

fig, ax = hemisphere_plot(
    northextent=-60,
    add_grid_labels=False, plot_scalebar=False, grid_color='black',
    fm_left=0.06, fm_right=0.94, fm_bottom=0.04, fm_top=0.98, add_atlas=False,)

# plt_polygon = ais_imbie2.plot(
#     ax=ax, transform=ccrs.epsg(3031), cmap="viridis")

plt_wais = ais_imbie2.loc[ais_imbie2.Regions == 'West'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='red', facecolor='none', linewidths=0.5, zorder=2)
plt_eais = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='blue', facecolor='none', linewidths=0.5, zorder=2)
plt_ap = ais_imbie2.loc[ais_imbie2.Regions == 'Peninsula'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='m', facecolor='none', linewidths=0.5, zorder=2)

ax.pcolormesh(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['eais_mask01'],
    cmap=ListedColormap(
        np.vstack(([0, 0, 0, 0], mcolors.to_rgba_array('lightblue')))),
    transform=ccrs.PlateCarree())
ax.pcolormesh(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['wais_mask01'],
    cmap=ListedColormap(
        np.vstack(([0, 0, 0, 0], mcolors.to_rgba_array('lightpink')))),
    transform=ccrs.PlateCarree())
ax.pcolormesh(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['ap_mask01'],
    cmap=ListedColormap(
        np.vstack(([0, 0, 0, 0], mcolors.to_rgba_array('lightgray')))),
    transform=ccrs.PlateCarree())


coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)

fig.savefig('figures/1_study_area/01.00.03 Antarctic Ice Sheets.png')


#### methods to extract EAIS mask
'''

#2 points in polygon path obtained in geometry.exterior.coords
eais_mask, eais_mask01 = points_in_polygon(
    lon, lat, Path([(9.74499797821047, -90)] + list(
        ais_imbie2.to_crs(4326).geometry[2].exterior.coords) + \
        [(9.74499797821047, -90)]))

#3 extract path from plotted contours
east_ctr = ais_imbie2.loc[ais_imbie2.Regions == 'East'].plot(
    ax=ax, transform=ccrs.epsg(3031),
    edgecolor='red', facecolor=None,
)
east_ctr.collections[0].get_paths()
'''


'''
ax.pcolormesh(
    lon, lat, one_degree_grids_cdo_area.cell_area,
    cmap='viridis', rasterized=True, transform=ccrs.PlateCarree(),
)

ax.contour(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['eais_mask01'], colors='blue', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=1, linestyles='solid',
    interpolation='none')
ax.contour(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['wais_mask01'], colors='red', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=1, linestyles='solid')
ax.contour(
    one_degree_grids_cdo_area.lon,
    one_degree_grids_cdo_area.lat,
    ais_masks['ap_mask01'], colors='m', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=1, linestyles='solid')


# one_degree_grids_cdo_area.cell_area.values[ais_masks['eais_mask']].sum() / 10**6
# ais_imbie2.geometry[2].area / 10**6
# one_degree_grids_cdo_area.cell_area.values[ais_masks['wais_mask']].sum() / 10**6
# ais_imbie2.geometry[1].area / 10**6
# one_degree_grids_cdo_area.cell_area.values[ais_masks['ap_mask']].sum() / 10**6
# ais_imbie2.geometry[3].area / 10**6
# 9761642.045593847, 9620521.647740476, 2110804.571811703, 2038875.6430063567, 232127.50065176177, 232678.32105463636


# plot ANT_Rignot_Basins_IMBIE2
ant_basins_imbie2 = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.shp'
)
plt_polygon = ant_basins_imbie2.plot(
    ax=ax, transform=ccrs.epsg(3031), cmap="viridis")

# plot MEaSUREs Antarctic Boundaries
basins_IMBIE_ais = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409164/Basins_IMBIE_Antarctica_v02.shp'
)
basins_IMBIE_ais_dis = basins_IMBIE_ais.dissolve('Regions')

plt_polygon1 = basins_IMBIE_ais.plot(
    ax=ax, transform=ccrs.epsg(3031), cmap="viridis",
    # column="Regions",
    )

plt_polygon2 = basins_IMBIE_ais_dis.plot(
    ax=ax, transform=ccrs.epsg(3031), cmap="viridis",
    )


coastline_antarctica = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409161/Coastline_Antarctica_v02.shp'
)
plt_polygon = coastline_antarctica.plot(ax=ax, transform=ccrs.epsg(3031))

iceshelf_antarctica = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409162/IceShelf_Antarctica_v02.shp'
)
plt_polygon = iceshelf_antarctica.plot(ax=ax, transform=ccrs.epsg(3031))

groundingline_antarctica = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409163/GroundingLine_Antarctica_v02.shp'
)
plt_polygon = groundingline_antarctica.plot(ax=ax, transform=ccrs.epsg(3031))

basins_antarctica = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409165/Basins_Antarctica_v02.shp'
)
plt_polygon = basins_antarctica.plot(ax=ax, transform=ccrs.epsg(3031))

iceboundaries_antarctica = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/NSIDC/MEaSUREs Antarctic Boundaries/128409166/IceBoundaries_Antarctica_v02.shp'
)
plt_polygon = iceboundaries_antarctica.plot(ax=ax, transform=ccrs.epsg(3031))


# Plot Bedmap2
bedmap_tif = xr.open_rasterio(
    'bas_palaeoclim_qino/others/bedmap2_tiff/bedmap2_surface.tif')
surface_height_bedmap = bedmap_tif.values.copy().astype(np.float64)
# surface_height_bedmap[surface_height_bedmap == 32767] = np.nan
surface_height_bedmap[surface_height_bedmap == 32767] = 0
bedmap_transform = ccrs.epsg(3031)
pltlevel_sh = np.arange(0, 4000.1, 1)
pltticks_sh = np.arange(0, 4000.1, 1000)
plt_cmp = ax.pcolormesh(
    bedmap_tif.x, bedmap_tif.y,
    surface_height_bedmap[0, :, :],
    cmap=cm.get_cmap('Blues', len(pltlevel_sh)),
    norm=BoundaryNorm(pltlevel_sh, ncolors=len(pltlevel_sh), clip=False),
    rasterized=True, transform=bedmap_transform,)

plt_ctr = ax.contour(
    bedmap_tif.x, bedmap_tif.y, surface_height_bedmap[0, :, :], levels=0,
    colors='red', rasterized=True, transform=bedmap_transform,
    linewidths=0.25)

# Save Bedmap2 contours
from matplotlib.path import Path
import pickle
plt_ctr.collections[0].get_paths()
with open(
    'bas_palaeoclim_qino/others/bedmap2_tiff/bedmap2_surface_contour.pkl',
    'wb') as f:
    pickle.dump(plt_ctr.collections[0].get_paths(), f)

# Plot BAS ADD file
import geopandas as gpd
hr_coastline_polygon_add = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/SCAR_ADD/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4.shp'
)
plt_polygon = hr_coastline_polygon_add.plot(ax=ax, transform=ccrs.epsg(3031))

with open(
    'bas_palaeoclim_qino/others/bedmap2_tiff/bedmap2_surface_contour.pkl',
    'rb') as f:
    bedmap2_surface_contour = pickle.load(f)
polygon = [
    (300, 0), (390, 320), (260, 580),
    (380, 839), (839, 839), (839, 0),
]
poly_path = Path(polygon)


import shapefile
reader = shapefile.Reader(
    'bas_palaeoclim_qino/observations/products/SCAR_ADD/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4.shp'
)
for shape in list(reader.iterShapes()):
    npoints = len(shape.points)  # total points
    nparts = len(shape.parts)  # total parts

    if nparts == 1:
        x_lon = np.zeros((len(shape.points), 1))
        y_lat = np.zeros((len(shape.points), 1))
        for ip in range(len(shape.points)):
            x_lon[ip] = shape.points[ip][0]
            y_lat[ip] = shape.points[ip][1]
        plt.plot(x_lon, y_lat, 'red', linewidth=0.25)

    else:   # loop over parts of each shape, plot separately
        for ip in range(nparts):
            i0 = shape.parts[ip]
            if ip < nparts-1:
                i1 = shape.parts[ip+1]-1
            else:
                i1 = npoints
            seg = shape.points[i0:i1+1]
            x_lon = np.zeros((len(seg), 1))
            y_lat = np.zeros((len(seg), 1))
            for ip in range(len(seg)):
                x_lon[ip] = seg[ip][0]
                y_lat[ip] = seg[ip][1]
            plt.plot(x_lon, y_lat, 'red', linewidth=0.25)

# identify problem in SCAR ADD

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath

projections = ccrs.SouthPolarStereo()
transform = ccrs.PlateCarree()

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 9.3]) / 2.54,
    subplot_kw={'projection': projections}, dpi=600)
ax.set_extent((-180, 180, -90, -60), crs=transform)

hr_coastline_polygon_add = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/SCAR_ADD/add_coastline_high_res_polygon_v7_4/add_coastline_high_res_polygon_v7_4.shp'
)
plt_polygon = hr_coastline_polygon_add.plot(ax=ax,)

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)

# set circular axes boundaries
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

plt.setp(ax.spines.values(), linewidth=0.2)

fig.savefig('figures/0_test/trial.png')


# Zwally_Antarctic_Ice_Sheets
zwally_ais = pd.read_csv(
    'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Zwally_Basins/Zwally_Antarctic_Ice_Sheets.txt',
    sep='\s+', header=None, names=['lat', 'lon', 'ice_sheet_id'], skiprows=9,
    # dtype={'ice_sheet_id': str},
    )

# zwally_ais_gdf = gpd.GeoDataFrame(
#     zwally_ais, geometry=gpd.points_from_xy(zwally_ais.lon, zwally_ais.lat))

geom_ap = Polygon(zip(
    zwally_ais.lon.loc[zwally_ais.ice_sheet_id == 28],
    zwally_ais.lat.loc[zwally_ais.ice_sheet_id == 28]))
polygon_ap = gpd.GeoDataFrame(geometry=[geom_ap])

geom_wais = Polygon(zip(
    zwally_ais.lon.loc[zwally_ais.ice_sheet_id == 29],
    zwally_ais.lat.loc[zwally_ais.ice_sheet_id == 29]))
polygon_wais = gpd.GeoDataFrame(geometry=[geom_wais])

geom_eais = Polygon(zip(
    zwally_ais.lon.loc[zwally_ais.ice_sheet_id == 30],
    zwally_ais.lat.loc[zwally_ais.ice_sheet_id == 30]))
polygon_eais = gpd.GeoDataFrame(geometry=[geom_eais])

polygon_ap.plot(ax=ax, color='red', zorder=3, transform=ccrs.PlateCarree())
polygon_wais.plot(ax=ax, color='blue', zorder=3, transform=ccrs.PlateCarree())
polygon_eais.plot(ax=ax, color='grey', zorder=3, transform=ccrs.PlateCarree())


'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot Antarctica bed height in Bedmap2

bedmap_bed = rh.fetch_bedmap2(datasets=['bed'])
# stats.describe(
#     bedmap_surface.surface.values, axis = None, nan_policy = 'omit')

bedmap_tif = xr.open_rasterio('data_source/bedmap2_tiff/bedmap2_bed.tif')

# from affine import Affine
# bedmap_transform = Affine(*bedmap_tif.attrs["transform"])
bedmap_transform = ccrs.epsg(3031)

projections = ccrs.SouthPolarStereo()
transform = ccrs.PlateCarree()

ticklabel = ticks_labels(-180, 179, -90, -65, 30, 10)
labelsize = 10

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 9.3]) / 2.54,
    subplot_kw={'projection': projections}, dpi=600)

ax.set_extent((-180, 180, -90, -60), crs=transform)

figure_margin = {
    'left': 0.12, 'right': 0.88, 'bottom': 0.08, 'top': 0.96}

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)
gl = ax.gridlines(
    crs=transform, linewidth=0.15, zorder=2, draw_labels=True,
    color='gray', alpha=0.5, linestyle='--',
    xlocs=ticklabel[0], ylocs=ticklabel[2], rotate_labels=False,
)
gl.ylabel_style = {'size': 0, 'color': 'white'}

fig.subplots_adjust(
    left=figure_margin['left'], right=figure_margin['right'],
    bottom=figure_margin['bottom'], top=figure_margin['top'])

# set circular axes boundaries
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

demlevel = np.arange(-2800, 2800.1, 20)
demticks = np.arange(-2800, 2800.1, 700)

cmp_top = cm.get_cmap('Blues_r', int(np.floor(len(demlevel) / 2)))
cmp_bottom = cm.get_cmap('Reds', int(np.floor(len(demlevel) / 2)))
cmp_colors = np.vstack(
    (cmp_top(np.linspace(0, 1, int(np.floor(len(demlevel) / 2)))),
     [1, 1, 1, 1],
     cmp_bottom(np.linspace(0, 1, int(np.floor(len(demlevel) / 2))))))
cmp_cmap = ListedColormap(cmp_colors, name='RedsBlues_r')


plt_dem = ax.pcolormesh(
    bedmap_tif.x.values, bedmap_tif.y.values, bedmap_bed.bed.values,
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cmp_cmap, rasterized=True, transform=bedmap_transform,)
cbar = fig.colorbar(
    plt_dem, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.08,
    shrink=1.2, aspect=25, ticks=demticks, extend='both',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Bed height above sea level [m] in Bedmap2")

plt.setp(ax.spines.values(), linewidth=0.2)

fig.savefig('figures/01_study_area/01.00.02 Bed height in Bedmap2.png')


'''
plt_theta = ax.pcolormesh(
    lon, lat, theta100, cmap=rvor_cmp, rasterized=True, transform=transform,
    norm=BoundaryNorm(theta_level, ncolors=rvor_cmp.N, clip=False), )

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot Antarctica surface height in Bedmap2

bedmap_surface = rh.fetch_bedmap2(datasets=['surface'])
# stats.describe(
#     bedmap_surface.surface.values, axis = None, nan_policy = 'omit')

# from affine import Affine
# bedmap_transform = Affine(*bedmap_tif.attrs["transform"])
bedmap_transform = ccrs.epsg(3031)

projections = ccrs.SouthPolarStereo()
transform = ccrs.PlateCarree()

ticklabel = ticks_labels(-180, 179, -90, -65, 30, 10)
labelsize = 10

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 9.3]) / 2.54,
    subplot_kw={'projection': projections}, dpi=600)

ax.set_extent((-180, 180, -90, -60), crs = transform)

figure_margin = {
    'left': 0.12, 'right': 0.88, 'bottom': 0.08, 'top': 0.96}

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)
gl = ax.gridlines(
    crs=transform, linewidth=0.15, zorder=2, draw_labels=True,
    color='gray', alpha=0.5, linestyle='--',
    xlocs=ticklabel[0], ylocs=ticklabel[2], rotate_labels = False,
    )
gl.ylabel_style = {'size': 0, 'color': 'white'}

fig.subplots_adjust(
    left=figure_margin['left'], right=figure_margin['right'],
    bottom=figure_margin['bottom'], top=figure_margin['top'])

# set circular axes boundaries
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

demlevel = np.arange(0, 4000.1, 20)
demticks = np.arange(0, 4000.1, 1000)

plt_dem = ax.pcolormesh(
    bedmap_tif.x.values, bedmap_tif.y.values,
    bedmap_surface.surface.values,
    norm=BoundaryNorm(demlevel, ncolors=len(demlevel), clip=False),
    cmap=cm.get_cmap('Blues', len(demlevel)), rasterized=True,
    transform=bedmap_transform,)
cbar = fig.colorbar(
    plt_dem, ax=ax, orientation="horizontal",  pad=0.08, fraction=0.07,
    shrink=1, aspect=25, ticks=demticks, extend='max',
    anchor=(0.5, 1), panchor=(0.5, 0))
cbar.ax.set_xlabel("Surface height [m] in Bedmap2")

plt.setp(ax.spines.values(), linewidth=0.2)

fig.savefig('figures/01_study_area/Surface height in Bedmap2.png')


'''
# https://scitools.org.uk/cartopy/docs/latest/gallery/lines_and_polygons/always_circular_stereo.html#sphx-glr-gallery-lines-and-polygons-always-circular-stereo-py

# https://www.fatiando.org/rockhound/latest/gallery/bedmap2.html#sphx-glr-gallery-bedmap2-py


ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.background_img(name='natural_earth', resolution='high',
                  extent=[-180, 180, -90, -60])

# https://stackoverflow.com/questions/45302485/matplotlib-focus-on-specific-lon-lat-using-spstere-projection
# http://neichin.github.io/personalweb/writing/Cartopy-shapefile/
# https://www.fatiando.org/rockhound/latest/gallery/bedmap2.html


from cartopy.mpl.ticker import LongitudeFormatter
projections = ccrs.SouthPolarStereo()
transform = ccrs.PlateCarree()

ticklabel = ticks_labels(-180, 179, -90, -65, 30, 10)
labelsize = 10

fig, ax = plt.subplots(
    1, 1, figsize=np.array([8.8, 9.3]) / 2.54,
    subplot_kw={'projection': projections}, dpi=600)

ax.set_extent((-180, 180, -90, -60), crs = transform)

figure_margin = {
    'left': 0.12, 'right': 0.88, 'bottom': 0.08, 'top': 0.96}

coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)
gl = ax.gridlines(
    crs=transform, linewidth=0.15, zorder=2, draw_labels=True,
    color='gray', alpha=0.5, linestyle='--',
    xlocs=ticklabel[0], ylocs=ticklabel[2],
    rotate_labels = False,
    # xpadding=0, ypadding=0,
    xformatter=LongitudeFormatter(degree_symbol='° '),
    )
gl.ylabel_style = {'size': 0, 'color': 'white'}

fig.subplots_adjust(
    left=figure_margin['left'], right=figure_margin['right'],
    bottom=figure_margin['bottom'], top=figure_margin['top'])

# set circular axes boundaries
theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax.set_boundary(circle, transform=ax.transAxes)

plt.setp(ax.spines.values(), linewidth=0.2)

fig.savefig('figures/0_test/trial.png')


'''



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm aprt Antarctica


#-------- basic set

lon = wisoaprt_alltime[expid[i]]['am'].lon
lat = wisoaprt_alltime[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.1_total_precipitation/' + '6.1.4.1 ' + expid[i] + ' aprt mm Antarctica.png'
# cbar_label1 = 'Precipitation [$mm \; day^{-1}$]'
cbar_label2 = 'Differences in precipitation [$\%$]'

# pltlevel = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
# pltticks = np.array([0, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 6, 8, 10,])
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
# pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1)


pltlevel2 = np.arange(-50, 50 + 1e-4, 10)
pltticks2 = np.arange(-50, 50 + 1e-4, 10)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()



nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])

for jcol in range(ncol):
    for irow in range(nrow):
        plt_mesh1 = axs[irow, jcol].pcolormesh(
            lon, lat, (wisoaprt_alltime[expid[i]]['mm'].sel(month=month_dec_num[jcol*3+irow])[0] / wisoaprt_alltime[expid[i]]['am'][0] - 1) * 100,
            norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
        
        plt.text(
            0.5, 1.05, month_dec[jcol*3+irow],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        
        print(str(month_dec_num[jcol*3+irow]) + ' ' + month_dec[jcol*3+irow])


cbar2 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(0.5, -0.5), ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.8, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm sic Antarctica


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.4_climate_fields/6.1.4.4 pi_alex sic mm Antarctica.png'
cbar_label2 = 'Differences in sea ice concentration [$\%$]'

pltlevel2 = np.arange(-50, 50 + 1e-4, 10)
pltticks2 = np.arange(-50, 50 + 1e-4, 10)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1)


nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-45, ax_org = axs[irow, jcol])

for jcol in range(ncol):
    for irow in range(nrow):
        # irow=0; jcol=0
        plt_mesh1 = axs[irow, jcol].pcolormesh(
            lon, lat,
            seaice['pi_alex_alltime']['mm'].sel(time=(seaice['pi_alex_alltime']['mm'].time.dt.month == month_dec_num[jcol*3+irow])).squeeze() - seaice['pi_alex_alltime']['am'],
            norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
        
        plt.text(
            0.5, 1.05, month_dec[jcol*3+irow],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        
        print(str(month_dec_num[jcol*3+irow]) + ' ' + month_dec[jcol*3+irow])


cbar2 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(0.5, -0.5), ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.8, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region animate daily 2*2 djf+jja pre and daily pre-weighted longitude


#-------------------------------- import total pre

# i = 0
# expid[i]

# tot_pre_alltime = {}

# with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.tot_pre_alltime.pkl', 'rb') as f:
#     tot_pre_alltime[expid[i]] = pickle.load(f)


#-------------------------------- import pre_weighted_lon and ocean pre

j = 0
expid[j]

pre_weighted_lon = {}
with open(exp_odir + expid[j] + '/analysis/echam/' + expid[j] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon[expid[j]] = pickle.load(f)

ocean_pre_alltime = {}
with open(exp_odir + expid[j] + '/analysis/echam/' + expid[j] + '.ocean_pre_alltime.pkl', 'rb') as f:
    ocean_pre_alltime[expid[j]] = pickle.load(f)


#-------------------------------- basic settings

itimestart_djf = np.where(ocean_pre_alltime[expid[j]]['daily'].time == np.datetime64('2025-12-01T23:52:30'))[0][0]
itimestart_jja = np.where(ocean_pre_alltime[expid[j]]['daily'].time == np.datetime64('2026-06-01T23:52:30'))[0][0]


pltlevel = np.concatenate(
    (np.arange(0, 0.5, 0.05), np.arange(0.5, 5 + 1e-4, 0.5)))
pltticks = np.concatenate(
    (np.arange(0, 0.5, 0.1), np.arange(0.5, 5 + 1e-4, 1)))
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=True)
# pltcmp = cm.get_cmap('RdBu', len(pltlevel))
pltcmp = cm.get_cmap('PuOr', len(pltlevel))

pltlevel2 = np.arange(0, 360 + 1e-4, 20)
pltticks2 = np.arange(0, 360 + 1e-4, 60)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()


#-------------------------------- plot

nrow = 2
ncol = 2

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])

djf_pre = ocean_pre_alltime[expid[j]]['daily'][itimestart_djf + 0,].copy() * 3600 * 24
jja_pre = ocean_pre_alltime[expid[j]]['daily'][itimestart_jja + 0,].copy() * 3600 * 24
# djf_pre.values[djf_pre.values < 2e-8 * 3600 * 24] = np.nan
# jja_pre.values[jja_pre.values < 2e-8 * 3600 * 24] = np.nan

plt1 = axs[0, 0].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    djf_pre.sel(lat=slice(-60, -90)), 
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    jja_pre.sel(lat=slice(-60, -90)), 
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt3 = axs[0, 1].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    pre_weighted_lon[expid[j]]['daily'][itimestart_djf + 0,].sel(lat=slice(-60, -90)),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), )
axs[1, 1].pcolormesh(
    ocean_pre_alltime[expid[j]]['daily'].lon,
    ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
    pre_weighted_lon[expid[j]]['daily'][itimestart_jja + 0,].sel(lat=slice(-60, -90)),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), )

plt.text(
    0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[itimestart_djf + 0,].values)[:10], transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[itimestart_jja + 0,].values)[:10], transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'DJF',
    transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'JJA',
    transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.55,aspect=40,extend='max',
    anchor=(-0.3, 0.05), ticks=pltticks)
cbar1.ax.set_xlabel('Precipitation sourced from\nopen ocean [$mm \; day^{-1}$]', linespacing=1.5)

cbar2 = fig.colorbar(
    plt3, ax=axs,
    orientation="horizontal",shrink=0.55,aspect=40,extend='neither',
    anchor=(1.2,-3.2),ticks=pltticks2)
cbar2.ax.set_xlabel('Precipitation-weighted open-oceanic\nsource longitude [$°$]', linespacing=1.5)

fig.subplots_adjust(left=0.04, right = 0.99, bottom = 0.14, top = 0.98)
fig.savefig('figures/test1.png')


#-------------------------------- animate with animation.FuncAnimation

nrow = 2
ncol = 2

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)


for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])

plt.text(
    -0.05, 0.5, 'DJF',
    transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'JJA',
    transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.55,aspect=40,extend='max',
    anchor=(-0.3, 0.05), ticks=pltticks)
cbar1.ax.set_xlabel('Precipitation sourced from\nopen ocean [$mm \; day^{-1}$]', linespacing=1.5)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.55,aspect=40,extend='neither',
    anchor=(1.2,-3.2),ticks=pltticks2)
cbar2.ax.set_xlabel('Precipitation-weighted open-oceanic\nsource longitude [$°$]', linespacing=1.5)
fig.subplots_adjust(left=0.04, right = 0.99, bottom = 0.14, top = 0.98)

plt_objs = []

def update_frames(itime):
    global plt_objs
    for plt_obj in plt_objs:
        plt_obj.remove()
    plt_objs = []
    #---- daily precipitation
    
    djf_pre = ocean_pre_alltime[expid[j]]['daily'][
        itimestart_djf + itime,].copy() * 3600 * 24
    jja_pre = ocean_pre_alltime[expid[j]]['daily'][
        itimestart_jja + itime,].copy() * 3600 * 24
    # djf_pre.values[djf_pre.values < 2e-8 * 3600 * 24] = np.nan
    # jja_pre.values[jja_pre.values < 2e-8 * 3600 * 24] = np.nan
    
    #---- plot daily precipitation
    plt1 = axs[0, 0].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        djf_pre.sel(lat=slice(-60, -90)), 
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt2 = axs[1, 0].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        jja_pre.sel(lat=slice(-60, -90)), 
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #---- plot daily pre_weighted_lon
    plt3 = axs[0, 1].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        pre_weighted_lon[expid[j]]['daily'][itimestart_djf + itime,].sel(lat=slice(-60, -90)),
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), )
    
    plt4 = axs[1, 1].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat.sel(lat=slice(-60, -90)),
        pre_weighted_lon[expid[j]]['daily'][itimestart_jja + itime,].sel(lat=slice(-60, -90)),
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), )
    
    plt5 = axs[0, 0].text(
        0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[
            itimestart_djf + itime,].values)[:10],
        transform=axs[0, 0].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt6 = axs[1, 0].text(
        0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[
            itimestart_jja + itime,].values)[:10],
        transform=axs[1, 0].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt_objs = [plt1, plt2, plt3, plt4, plt5, plt6]
    # plt_objs = [plt6]
    
    return(plt_objs)

ani = animation.FuncAnimation(
    fig, update_frames,
    frames=90, interval=500, blit=False)
ani.save(
    'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0_Antarctic DJF_JJA daily precipitation and pre_weighted_lon ' + expid[j] + '.mp4',
    # 'figures/test.mp4',
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)




'''
ocean_pre_alltime[expid[j]]['daily'].time[itimestart_djf].values
ocean_pre_alltime[expid[j]]['daily'].time[itimestart_jja].values

#-------------------------------- animate with animation.ArtistAnimation

nrow = 2
ncol = 2

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-60, ax_org = axs[irow, jcol])

plt.text(
    -0.05, 0.5, 'DJF',
    transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'JJA',
    transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical')


ims = []

for itime in range(2):
    # itime = 0
    
    #---- daily precipitation
    
    djf_pre = ocean_pre_alltime[expid[j]]['daily'][
        itimestart_djf + itime,].copy() * 3600 * 24
    jja_pre = ocean_pre_alltime[expid[j]]['daily'][
        itimestart_jja + itime,].copy() * 3600 * 24
    
    #---- plot daily precipitation
    plt1 = axs[0, 0].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat,
        djf_pre, rasterized = True,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    plt2 = axs[1, 0].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat,
        jja_pre, rasterized = True,
        norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
    
    #---- plot daily pre_weighted_lon
    plt3 = axs[0, 1].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat,
        pre_weighted_lon[expid[j]]['daily'][itimestart_djf + itime,],
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), rasterized = True,)
    
    plt4 = axs[1, 1].pcolormesh(
        ocean_pre_alltime[expid[j]]['daily'].lon,
        ocean_pre_alltime[expid[j]]['daily'].lat,
        pre_weighted_lon[expid[j]]['daily'][itimestart_jja + itime,],
        norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(), rasterized = True,)
    
    plt5 = plt.text(
        0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[
            itimestart_djf + itime,].values)[:10],
        transform=axs[0, 0].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    plt6 = plt.text(
        0.5, 1.05, str(ocean_pre_alltime[expid[j]]['daily'].time[
            itimestart_jja + itime,].values)[:10],
        transform=axs[1, 0].transAxes,
        ha='center', va='center', rotation='horizontal')
    
    ims.append([plt1, plt2, plt3, plt4, plt5, plt6, ])
    print(str(itime))


cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.55,aspect=40,extend='max',
    anchor=(-0.3, 0.05), ticks=pltticks)
cbar1.ax.set_xlabel('Precipitation sourced from\nopen ocean [$mm \; day^{-1}$]', linespacing=1.5)

cbar2 = fig.colorbar(
    plt3, ax=axs,
    orientation="horizontal",shrink=0.55,aspect=40,extend='neither',
    anchor=(1.2,-3.2),ticks=pltticks2)
cbar2.ax.set_xlabel('Precipitation-weighted open-oceanic\nsource longitude [$°$]', linespacing=1.5)
fig.subplots_adjust(left=0.04, right = 0.99, bottom = 0.14, top = 0.98)

ani = animation.ArtistAnimation(fig, ims, interval=250, blit=True)
ani.save(
    'figures/6_awi/6.1_echam6/6.1.4_extreme_precipitation/6.1.4.0_precipitation_distribution/6.1.4.0_Antarctic DJF_JJA daily precipitation and pre_weighted_lon ' + expid[j] + '_1.mp4',
    # 'figures/test.mp4',
    progress_callback=lambda iframe, n: print(f'Saving frame {iframe} of {n}'),)

(djf_pre.values < 2e-8).sum()
(djf_pre.values < 2e-8 * 3600 * 24).sum()
(ocean_pre_alltime[expid[j]]['daily'].values < 2e-8).sum()
(ocean_pre_alltime[expid[j]]['daily'].values < 2e-8 * 3600 * 24).sum()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA source sst


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.2_sst/' + '6.1.3.2 ' + expid[i] + ' pre_weighted_sst am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source SST [$°C$]'


pltlevel = np.arange(0, 32 + 1e-4, 2)
pltticks = np.arange(0, 32 + 1e-4, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=(len(pltlevel)-1), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-10, 10 + 1e-4, 1)
pltticks2 = np.arange(-10, 10 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()


nrow = 1
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                           add_grid_labels=False)
    plt.text(
        0, 1.05, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='left', va='center', rotation='horizontal')
    ipanel += 1

#-------- Am, DJF, JJA values
plt_mesh1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt_mesh2 = axs[3].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='DJF') - pre_weighted_sst[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

ttest_fdr_res = ttest_fdr_control(
    pre_weighted_sst[expid[i]]['sea'][3::4,],
    pre_weighted_sst[expid[i]]['sea'][1::4,],)
axs[3].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.8), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt_mesh2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-5.5),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA source sst Antarctica


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.2_sst/' + '6.1.3.2 ' + expid[i] + ' pre_weighted_sst am_DJF_JJA Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source SST [$°C$]'

pltlevel = np.arange(8, 20 + 1e-4, 1)
pltticks = np.arange(8, 20 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=(len(pltlevel)-1), clip=True)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-6, 6 + 1e-4, 1)
pltticks2 = np.arange(-6, 6 + 1e-4, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=(len(pltlevel2) - 1), clip=True)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)-1).reversed()


nrow = 1
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-50, ax_org = axs[jcol])
    cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, axs[jcol])
    plt.text(
        0, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1

#-------- Am, DJF, JJA values
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

axs[1].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

axs[2].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)


#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat, pre_weighted_sst[expid[i]]['sm'].sel(season='DJF') - pre_weighted_sst[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

ttest_fdr_res = ttest_fdr_control(
    pre_weighted_sst[expid[i]]['sea'][3::4,],
    pre_weighted_sst[expid[i]]['sea'][1::4,],)
axs[3].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.7),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)


'''
# ctr_level = np.array([1, 2, 3, 4, 5, ])

# plt_ctr1 = axs[0].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_sst[expid[i]]['ann'].std(
#         dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[0].clabel(plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

# plt_ctr2 = axs[1].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_sst[expid[i]]['sea'].sel(
#         time=(pre_weighted_sst[expid[i]]['sea'].time.dt.month == 2)
#         ).std(dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1].clabel(plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7)

# plt_ctr3 = axs[2].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_sst[expid[i]]['sea'].sel(
#         time=(pre_weighted_sst[expid[i]]['sea'].time.dt.month == 8)
#         ).std(dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[2].clabel(plt_ctr3, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA source lon

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/' + '6.1.3.1 ' + expid[i] + ' pre_weighted_lon am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted open-oceanic relative source longitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source longitude [$°$]'

pltlevel = np.arange(-180, 180 + 1e-4, 15)
pltticks = np.arange(-180, 180 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-50, 50 + 1e-4, 10)
pltticks2 = np.arange(-50, 50 + 1e-4, 10)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()


nrow = 1
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                           add_grid_labels=False)
    plt.text(
        0, 1.05, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='left', va='center', rotation='horizontal')
    ipanel += 1

#-------- Am, DJF, JJA values
plt1 = axs[0].pcolormesh(
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['am'], lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'), lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['sm'].sel(season='JJA'), lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat,
    calc_lon_diff(
        pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'),
        pre_weighted_lon[expid[i]]['sm'].sel(season='JJA')),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

wwtest_res = circ.watson_williams(
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 2)).values * np.pi / 180,
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 8)).values * np.pi / 180,
    axis=0,
    )[0] < 0.05
axs[3].scatter(
    x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0.8), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-5.5),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)



'''
pltlevel = np.arange(0, 360 + 1e-4, 15)
pltticks = np.arange(0, 360 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA source lon Antarctic


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/' + '6.1.3.1 ' + expid[i] + ' pre_weighted_lon am_DJF_JJA Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic relative source longitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source longitude [$°$]'

pltlevel = np.arange(-180, 180 + 1e-4, 15)
pltticks = np.arange(-180, 180 + 1e-4, 30)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('BrBG', len(pltlevel)-1).reversed()


pltlevel2 = np.arange(-30, 30 + 1e-4, 5)
pltticks2 = np.arange(-30, 30 + 1e-4, 5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()


nrow = 1
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-50, ax_org = axs[jcol])
    cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, axs[jcol])
    plt.text(
        0, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1

#-------- Am, DJF, JJA values
plt1 = axs[0].pcolormesh(
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['am'], lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'), lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat,
    calc_lon_diff(pre_weighted_lon[expid[i]]['sm'].sel(season='JJA'), lon_2d),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat,
    calc_lon_diff(
        pre_weighted_lon[expid[i]]['sm'].sel(season='DJF'),
        pre_weighted_lon[expid[i]]['sm'].sel(season='JJA')),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

wwtest_res = circ.watson_williams(
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 2)).values * np.pi / 180,
    pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 8)).values * np.pi / 180,
    axis=0,
    )[0] < 0.05
axs[3].scatter(
    x=lon_2d[wwtest_res], y=lat_2d[wwtest_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='neither',
    anchor=(-0.2, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.7),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot mm source lat Antarctica


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat mm Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source latitude [$°$]'

pltlevel = np.arange(-50, -30 + 1e-4, 2)
pltticks = np.arange(-50, -30 + 1e-4, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()


ctr_level = np.array([1, 2, 3, 4, 5, ])

nrow = 3
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.1, 'wspace': 0.1},)

ipanel=0
for irow in range(nrow):
    for jcol in range(ncol):
        axs[irow, jcol] = hemisphere_plot(northextent=-50, ax_org = axs[irow, jcol])
        cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, axs[irow, jcol])
        plt.text(
            0, 0.95, panel_labels[ipanel],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        ipanel += 1


for jcol in range(ncol):
    for irow in range(nrow):
        plt_mesh1 = axs[irow, jcol].pcolormesh(
            lon, lat, pre_weighted_lat[expid[i]]['mm'].sel(
                month=month_dec_num[jcol*3+irow]),
            norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
        # plt_ctr1 = axs[irow, jcol].contour(
        #     lon, lat.sel(lat=slice(-45, -90)),
        #     pre_weighted_lat[expid[i]]['mon'].sel(time=(pre_weighted_lat[expid[
        #         i]]['mon'].time.dt.month == month_dec_num[jcol*3+irow])).std(
        #     dim='time', skipna=True).sel(lat=slice(-45, -90)),
        #     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
        #     linewidths=0.5, linestyles='solid',
        # )
        # axs[irow, jcol].clabel(
        #     plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
        #     levels=ctr_level, inline_spacing=10, fontsize=7,)
        
        plt.text(
            0.5, 1.05, month_dec[jcol*3+irow],
            transform=axs[irow, jcol].transAxes,
            ha='center', va='center', rotation='horizontal')
        
        print(str(month_dec_num[jcol*3+irow]) + ' ' + month_dec[jcol*3+irow])


cbar1 = fig.colorbar(
    plt_mesh1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(0.5, -0.5), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom*0.8, top = 0.98)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA source lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat am_DJF_JJA.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source latitude [$°$]'


pltlevel = np.arange(-60, 60 + 1e-4, 10)
pltticks = np.arange(-60, 60 + 1e-4, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()

pltlevel2 = np.arange(-20, 20 + 1e-4, 2)
pltticks2 = np.arange(-20, 20 + 1e-4, 4)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()


nrow = 1
ncol = 4
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                           add_grid_labels=False)
    plt.text(
        0, 1.05, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='left', va='center', rotation='horizontal')
    ipanel += 1

#-------- Am, DJF, JJA values
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF') - pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

ttest_fdr_res = ttest_fdr_control(
    pre_weighted_lat[expid[i]]['sea'][3::4,],
    pre_weighted_lat[expid[i]]['sea'][1::4,],)
axs[3].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.8), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-5.5),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)



# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot am/DJF/JJA/DJF-JJA source lat Antarctica


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat am_DJF_JJA Antarctica.png'
cbar_label1 = 'Precipitation-weighted open-oceanic source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted open-oceanic source latitude [$°$]'

pltlevel = np.arange(-50, -30 + 1e-4, 2)
pltticks = np.arange(-50, -30 + 1e-4, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)-1).reversed()


pltlevel2 = np.arange(-10, 10 + 1e-4, 2)
pltticks2 = np.arange(-10, 10 + 1e-4, 2)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

ctr_level = np.array([1, 2, 3, 4, 5, ])

nrow = 1
ncol = 4
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

ipanel=0
for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-50, ax_org = axs[jcol])
    cplot_ice_cores(major_ice_core_site.lon, major_ice_core_site.lat, axs[jcol])
    plt.text(
        0, 0.95, panel_labels[ipanel],
        transform=axs[jcol].transAxes,
        ha='center', va='center', rotation='horizontal')
    ipanel += 1

#-------- Am, DJF, JJA values
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['am'],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr1 = axs[0].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_lat[expid[i]]['ann'].std(
#         dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[0].clabel(plt_ctr1, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

axs[1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr2 = axs[1].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_lat[expid[i]]['sea'].sel(
#         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
#         ).std(dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1].clabel(plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7)

axs[2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr3 = axs[2].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_lat[expid[i]]['sea'].sel(
#         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
#         ).std(dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[2].clabel(plt_ctr3, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)


#-------- differences
plt2 = axs[3].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF') - pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

ttest_fdr_res = ttest_fdr_control(
    pre_weighted_lat[expid[i]]['sea'][3::4,],
    pre_weighted_lat[expid[i]]['sea'][1::4,],)
axs[3].scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.5, c='k', marker='.', edgecolors='none',
    transform=ccrs.PlateCarree(),
    )

plt.text(
    0.5, 1.05, 'Annual mean', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF - JJA', transform=axs[3].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    plt2, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.7),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate annual circle of aprt frac over AIS

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_spave.pkl', 'rb') as f:
    aprt_geo7_spave = pickle.load(f)

geo_regions = [
    'NHland', 'SHland', 'Antarctica',
    'NHocean', 'NHseaice', 'SHocean', 'SHseaice']
wisotypes = {'NHland': 16, 'SHland': 17, 'Antarctica': 18,
             'NHocean': 19, 'NHseaice': 20, 'SHocean': 21, 'SHseaice': 22}

aprt_frc_AIS = {}


for imask in aprt_geo7_spave.keys():
    # imask = 'EAIS'
    print(imask)
    
    aprt_mm_AIS = aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values
    
    aprt_frc_AIS[imask] = {}
    
    aprt_frc_AIS[imask]['Open ocean'] = pd.DataFrame(data={
        'Month': month,
        'frc_AIS': (aprt_geo7_spave[imask]['mm'].sel(
            wisotype=[
                wisotypes['NHocean'], wisotypes['NHseaice'],
                wisotypes['SHocean'],
                ]).sum(dim='wisotype').values / aprt_mm_AIS) * 100
        })
    
    aprt_frc_AIS[imask]['SH sea ice'] = pd.DataFrame(data={
        'Month': month,
        'frc_AIS': (aprt_geo7_spave[imask]['mm'].sel(
            wisotype=[
                wisotypes['NHocean'], wisotypes['NHseaice'],
                wisotypes['SHocean'], wisotypes['SHseaice'],
                ]).sum(dim='wisotype').values / aprt_mm_AIS) * 100
        })
    
    aprt_frc_AIS[imask]['Land excl. Antarctica'] = pd.DataFrame(data={
        'Month': month,
        'frc_AIS': (aprt_geo7_spave[imask]['mm'].sel(
            wisotype=[
                wisotypes['NHocean'], wisotypes['NHseaice'],
                wisotypes['SHocean'], wisotypes['SHseaice'],
                wisotypes['NHland'], wisotypes['SHland'],
                ]).sum(dim='wisotype').values / aprt_mm_AIS) * 100
        })
    
    aprt_frc_AIS[imask]['Antarctica'] = pd.DataFrame(data={
        'Month': month,
        'frc_AIS': (aprt_geo7_spave[imask]['mm'].sel(
            wisotype=[
                wisotypes['NHocean'], wisotypes['NHseaice'],
                wisotypes['SHocean'], wisotypes['SHseaice'],
                wisotypes['NHland'], wisotypes['SHland'],
                wisotypes['Antarctica'],
                ]).sum(dim='wisotype').values / aprt_mm_AIS) * 100
        })

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc_AIS.pkl', 'wb') as f:
    pickle.dump(aprt_frc_AIS, f)


'''
#-------------------------------- check

#-------- import data
with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_geo7_spave.pkl', 'rb') as f:
    aprt_geo7_spave = pickle.load(f)

geo_regions = [
    'NHland', 'SHland', 'Antarctica',
    'NHocean', 'NHseaice', 'SHocean', 'SHseaice']
wisotypes = {'NHland': 16, 'SHland': 17, 'Antarctica': 18,
             'NHocean': 19, 'NHseaice': 20, 'SHocean': 21, 'SHseaice': 22}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.aprt_frc_AIS.pkl', 'rb') as f:
    aprt_frc_AIS = pickle.load(f)

imask = 'EAIS'
iregion = 'Open ocean'
res1 = aprt_frc_AIS[imask][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['mm'].sel(
    wisotype=slice(19, 21)).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values) * 100
(res1 == res2).all()


iregion = 'SH sea ice'
res1 = aprt_frc_AIS[imask][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['mm'].sel(
    wisotype=slice(19, 22)).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values) * 100
(res1 == res2).all()

iregion = 'Land excl. Antarctica'
res1 = aprt_frc_AIS[imask][iregion].frc_AIS.values

res2 = (aprt_geo7_spave[imask]['mm'].sel(
    wisotype=[16, 17, 19, 20, 21, 22]).sum(dim='wisotype').values / \
        aprt_geo7_spave[imask]['mm'].sum(dim='wisotype').values) * 100
(res1 == res2).all()
np.max(abs(res1 - res2))

iregion = 'Antarctica'
aprt_frc_AIS[imask][iregion].frc_AIS.values



aprt_frc_AIS[imask]['SH sea ice'].frc_AIS - aprt_frc_AIS[imask]['Open ocean'].frc_AIS
aprt_frc_AIS[imask]['Land excl. Antarctica'].frc_AIS - aprt_frc_AIS[imask]['SH sea ice'].frc_AIS
aprt_frc_AIS[imask]['Antarctica'].frc_AIS - aprt_frc_AIS[imask]['Land excl. Antarctica'].frc_AIS

aprt_geo7_spave['AIS']['mm'].sel(wisotype=20).values / aprt_mm_AIS.values

'''
# endregion
# -----------------------------------------------------------------------------


# region sm plot
# axs[1, 0].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF'),
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr2 = axs[1, 0].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_lat[expid[i]]['sea'].sel(
#         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
#         ).std(dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1, 0].clabel(plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

# axs[1, 1].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='MAM'),
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr3 = axs[1, 1].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_lat[expid[i]]['sea'].sel(
#         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 5)
#         ).std(dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1, 1].clabel(plt_ctr3, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

# axs[1, 2].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr4 = axs[1, 2].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_lat[expid[i]]['sea'].sel(
#         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
#         ).std(dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1, 2].clabel(plt_ctr4, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

# axs[1, 3].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='SON'),
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr5 = axs[1, 3].contour(
#     lon, lat.sel(lat=slice(-50, -90)),
#     pre_weighted_lat[expid[i]]['sea'].sel(
#         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 11)
#         ).std(dim='time', skipna=True).sel(lat=slice(-50, -90)),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1, 3].clabel(plt_ctr5, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)


#-------- sm - am
# plt_mesh2 = axs[2, 0].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF') - pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
# axs[2, 1].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='MAM') - pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
# axs[2, 2].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='JJA') - pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
# axs[2, 3].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='SON') - pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)


# plt.text(
#     0.5, 1.05, 'DJF', transform=axs[1, 0].transAxes,
#     ha='center', va='center', rotation='horizontal')
# plt.text(
#     0.5, 1.05, 'MAM', transform=axs[1, 1].transAxes,
#     ha='center', va='center', rotation='horizontal')
# plt.text(
#     0.5, 1.05, 'JJA', transform=axs[1, 2].transAxes,
#     ha='center', va='center', rotation='horizontal')
# plt.text(
#     0.5, 1.05, 'SON', transform=axs[1, 3].transAxes,
#     ha='center', va='center', rotation='horizontal')

# plt.text(
#     0.5, 1.05, 'DJF - Annual mean', transform=axs[2, 0].transAxes,
#     ha='center', va='center', rotation='horizontal')
# plt.text(
#     0.5, 1.05, 'MAM - Annual mean', transform=axs[2, 1].transAxes,
#     ha='center', va='center', rotation='horizontal')
# plt.text(
#     0.5, 1.05, 'JJA - Annual mean', transform=axs[2, 2].transAxes,
#     ha='center', va='center', rotation='horizontal')
# plt.text(
#     0.5, 1.05, 'SON - Annual mean', transform=axs[2, 3].transAxes,
#     ha='center', va='center', rotation='horizontal')




# #-------- sm
# axs[1, 0].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF'),
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr2 = axs[1, 0].contour(
#     lon, lat,
#     pre_weighted_lat[expid[i]]['sea'].sel(
#         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
#         ).std(dim='time', skipna=True),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1, 0].clabel(plt_ctr2, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

# axs[1, 1].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='MAM'),
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr3 = axs[1, 1].contour(
#     lon, lat,
#     pre_weighted_lat[expid[i]]['sea'].sel(
#         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 5)
#         ).std(dim='time', skipna=True),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1, 1].clabel(plt_ctr3, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

# axs[1, 2].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='JJA'),
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr4 = axs[1, 2].contour(
#     lon, lat,
#     pre_weighted_lat[expid[i]]['sea'].sel(
#         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
#         ).std(dim='time', skipna=True),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1, 2].clabel(plt_ctr4, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)

# axs[1, 3].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='SON'),
#     norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
# plt_ctr5 = axs[1, 3].contour(
#     lon, lat,
#     pre_weighted_lat[expid[i]]['sea'].sel(
#         time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 11)
#         ).std(dim='time', skipna=True),
#     levels=ctr_level, colors = 'b', transform=ccrs.PlateCarree(),
#     linewidths=0.5, linestyles='solid',
# )
# axs[1, 3].clabel(plt_ctr5, inline=1, colors='b', fmt=remove_trailing_zero,
#           levels=ctr_level, inline_spacing=10, fontsize=7,)


# #-------- sm - am
# plt_mesh2 = axs[2, 0].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='DJF') - pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
# axs[2, 1].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='MAM') - pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
# axs[2, 2].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='JJA') - pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
# axs[2, 3].pcolormesh(
#     lon, lat, pre_weighted_lat[expid[i]]['sm'].sel(season='SON') - pre_weighted_lat[expid[i]]['am'],
#     norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

# plt.text(
#     0.5, 1.05, 'DJF', transform=axs[1, 0].transAxes,
#     ha='center', va='center', rotation='horizontal')

# plt.text(
#     0.5, 1.05, 'MAM', transform=axs[1, 1].transAxes,
#     ha='center', va='center', rotation='horizontal')

# plt.text(
#     0.5, 1.05, 'JJA', transform=axs[1, 2].transAxes,
#     ha='center', va='center', rotation='horizontal')

# plt.text(
#     0.5, 1.05, 'SON', transform=axs[1, 3].transAxes,
#     ha='center', va='center', rotation='horizontal')

# plt.text(
#     0.5, 1.05, 'DJF - Annual mean', transform=axs[2, 0].transAxes,
#     ha='center', va='center', rotation='horizontal')

# plt.text(
#     0.5, 1.05, 'MAM - Annual mean', transform=axs[2, 1].transAxes,
#     ha='center', va='center', rotation='horizontal')

# plt.text(
#     0.5, 1.05, 'JJA - Annual mean', transform=axs[2, 2].transAxes,
#     ha='center', va='center', rotation='horizontal')

# plt.text(
#     0.5, 1.05, 'SON - Annual mean', transform=axs[2, 3].transAxes,
#     ha='center', va='center', rotation='horizontal')
# endregion

# -----------------------------------------------------------------------------
# region significancy test


djf_mean = pre_weighted_lat[expid[i]]['sm'].sel(season='DJF')
jja_mean = pre_weighted_lat[expid[i]]['sm'].sel(season='JJA')

djf_data = pre_weighted_lat[expid[i]]['sea'].sel(
    time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
    )
jja_data = pre_weighted_lat[expid[i]]['sea'].sel(
    time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
    )
ann_data = pre_weighted_lat[expid[i]]['ann']

#-------- check normality
# check_normality_3d(jja_data.values)

#-------- check variance
# check_equal_variance_3d(djf_data.values, jja_data.values)

#---- student t test

ttest_fdr_res = ttest_fdr_control(djf_data.values, jja_data.values,)


#-------- plot

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat
lon_2d, lat_2d = np.meshgrid(lon, lat,)

output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat DJF_JJA significancy test.png'
cbar_label2 = 'Differences in precipitation-weighted\nopen-oceanic source latitude [$°$]'

pltlevel2 = np.arange(-20, 20 + 1e-4, 2)
pltticks2 = np.arange(-20, 20 + 1e-4, 4)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2)-1, clip=True)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)-1).reversed()

fig, ax = globe_plot(
    add_grid_labels=False,
    fm_left=0.01, fm_right=0.99, fm_bottom=0.09, fm_top=0.99,)

plt_mesh = ax.pcolormesh(
    lon, lat,
    djf_mean - jja_mean,
    norm=pltnorm2, cmap=pltcmp2, transform=ccrs.PlateCarree(),)

ax.scatter(
    x=lon_2d[ttest_fdr_res], y=lat_2d[ttest_fdr_res],
    s=0.1, c='k', marker='.', edgecolors='none'
    )

cbar = fig.colorbar(
    plt_mesh, ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks2, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label2, linespacing=1.5)
fig.savefig(output_png)



'''
#-------- check normality
array = ann_data.values

whether_normal = np.full(array.shape[1:], True)

for ilat in range(whether_normal.shape[0]):
    for ilon in range(whether_normal.shape[1]):
        # ilat = 48; ilon = 96
        test_data = array[:, ilat, ilon][np.isfinite(array[:, ilat, ilon])]
        
        if (len(test_data) < 3):
            whether_normal[ilat, ilon] = False
        else:
            whether_normal[ilat, ilon] = stats.shapiro(test_data).pvalue > 0.05

whether_normal.sum() / len(whether_normal.flatten())


check_normality_3d(array)

#-------- check FDR control
# method 1

fdr_bh = multitest.fdrcorrection(
    ttest_djf_jja.reshape(-1),
    alpha=0.05,
    method='i',
)
bh_test1 = fdr_bh[0].reshape(ttest_djf_jja.shape)
(bh_test1 == bh_test4).all()

bh_test1.sum()
(ttest_djf_jja < 0.05).sum()


# method 2

bh_fdr = 0.05

sortind = np.argsort(ttest_djf_jja.reshape(-1))
pvals_sorted = np.take(ttest_djf_jja.reshape(-1), sortind)
rank = np.arange(1, len(pvals_sorted)+1)
bh_critic = rank / len(pvals_sorted) * bh_fdr

where_smaller = np.where(pvals_sorted < bh_critic)

bh_test2 = ttest_djf_jja <= pvals_sorted[where_smaller[0][-1]]

(bh_test1 == bh_test2).all()

# method 3
import mne
fdr_bh3 = mne.stats.fdr_correction(
    ttest_djf_jja.reshape(-1), alpha=0.05, method='indep')
bh_test3 = fdr_bh3[0].reshape(ttest_djf_jja.shape)
(bh_test1 == bh_test3).all()

#-------- check variance

array1 = djf_data.values
array2 = jja_data.values

variance_equal = np.full(array1.shape[1:], True)
for ilat in range(variance_equal.shape[0]):
    for ilon in range(variance_equal.shape[1]):
        # ilat = 48; ilon = 96
        test_data1 = array1[:, ilat, ilon][np.isfinite(array1[:, ilat, ilon])]
        test_data2 = array2[:, ilat, ilon][np.isfinite(array2[:, ilat, ilon])]
        variance_equal[ilat, ilon] = stats.fligner(test_data1, test_data2).pvalue > 0.05

variance_equal.sum() / len(variance_equal.flatten())

check_equal_variance_3d(array1, array2)

#-------- check student t test

ttest_djf_jja = stats.ttest_ind(
    djf_data, jja_data,
    nan_policy='omit',
    alternative='two-sided',
    ).pvalue.data
ttest_djf_jja[np.isnan(ttest_djf_jja)] = 1

#---- FDR control

bh_test4 = fdr_control_bh(ttest_djf_jja)

bh_test5 = ttest_fdr_control(djf_data, jja_data,)
ttest_res2 = ttest_fdr_control(djf_data, jja_data, fdr_control=False)
(bh_test4 == bh_test5).all()
(ttest_djf_jja == ttest_res2).all()

#-------- test for normality
ilat = 48
ilon = 96
test_data = djf_data[:, ilat, ilon]
test_data = jja_data[:, ilat, ilon]
test_data = pre_weighted_lat[expid[i]]['sea'][:, ilat, ilon]
test_data = pre_weighted_lat[expid[i]]['ann'][:, ilat, ilon]
stats.shapiro(test_data.values[np.isfinite(test_data.values)],)

# other checks
np.isnan(jja_data.values).all(axis=0).sum()
np.isnan(djf_data.values).all(axis=0).sum()
np.isnan(djf_mean.values).sum()

'''

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot ann/DJF/JJA standard deviation of source lon


#-------- basic set

lon = pre_weighted_lon[expid[i]]['am'].lon
lat = pre_weighted_lon[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.1_lon/' + '6.1.3.1 ' + expid[i] + ' pre_weighted_lon ann_DJF_JJA std.png'
cbar_label1 = 'Standard deviation of precipitation-weighted open-oceanic source longitude [$°$]'

pltlevel = np.arange(0, 10 + 1e-4, 1)
pltticks = np.arange(0, 10 + 1e-4, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

nrow = 1
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                           add_grid_labels=False)

#-------- Annual, DJF, JJA std
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_lon[expid[i]]['ann'].std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 2)
        ).std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs, ticks=pltticks,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(0.5, 0.8),
    )
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)



'''
np.isnan(pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True)).sum()
np.isnan(pre_weighted_lon[expid[i]]['sea'].sel(
        time=(pre_weighted_lon[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=False)).sum()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot ann/DJF/JJA std of source lat


#-------- basic set

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat ann_DJF_JJA std.png'
cbar_label1 = 'Standard deviation of precipitation-weighted open-oceanic source latitude [$°$]'

pltlevel = np.arange(0, 10 + 1e-4, 1)
pltticks = np.arange(0, 10 + 1e-4, 2)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)

nrow = 1
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                           add_grid_labels=False)

#-------- Annual, DJF, JJA std
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['ann'].std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
        ).std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs, ticks=pltticks,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(0.5, 0.8),
    )
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)



'''
np.isnan(pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True)).sum()
np.isnan(pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=False)).sum()

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot ann/DJF/JJA std of source lat Antarctic


#-------- basic set

lon = pre_weighted_lat[expid[i]]['am'].lon
lat = pre_weighted_lat[expid[i]]['am'].lat


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.3_source_var/6.1.3.0_lat/' + '6.1.3.0 ' + expid[i] + ' pre_weighted_lat ann_DJF_JJA std Antarctica.png'
cbar_label1 = 'Standard deviation of precipitation-weighted open-oceanic source latitude [$°$]'

pltlevel = np.arange(0, 5 + 1e-4, 0.5)
pltticks = np.arange(0, 5 + 1e-4, 1)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel)-1, clip=True)
pltcmp = cm.get_cmap('Blues', len(pltlevel)-1)


nrow = 1
ncol = 3
fm_bottom = 2 / (5.8*nrow + 2)

fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([5.8*ncol, 5.8*nrow + 2]) / 2.54,
    subplot_kw={'projection': ccrs.SouthPolarStereo()},
    gridspec_kw={'hspace': 0.05, 'wspace': 0.05},)

for jcol in range(ncol):
    axs[jcol] = hemisphere_plot(northextent=-60, ax_org = axs[jcol])

#-------- Annual, DJF, JJA std
plt1 = axs[0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['ann'].std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 2)
        ).std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[i]]['sea'].sel(
        time=(pre_weighted_lat[expid[i]]['sea'].time.dt.month == 8)
        ).std(dim='time', skipna=True),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Annual', transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'DJF', transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'JJA', transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    plt1, ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='max',
    anchor=(0.5, 0.4), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.8, top = 0.94)
fig.savefig(output_png)

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region function to generate mask for three AIS


def create_ais_mask(lon_lat_file = None, ais_file = None):
    '''
    ---- Input
    lon_lat_file: a file contains the desired lon/lat
    ais_file: a shapefile contains the AIS.
    
    ---- Output
    
    '''
    
    import numpy as np
    import xarray as xr
    import geopandas as gpd
    from geopandas.tools import sjoin
    
    if (lon_lat_file is None):
        lon_lat_file = 'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc'
    
    ncfile = xr.open_dataset(lon_lat_file)
    lon, lat = np.meshgrid(ncfile.lon, ncfile.lat,)
    
    if (ais_file is None):
        ais_file = 'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp'
    
    shpfile = gpd.read_file(ais_file)
    
    lon_lat = gpd.GeoDataFrame(
        crs="EPSG:4326", geometry=gpd.points_from_xy(
            lon.reshape(-1, 1), lat.reshape(-1, 1))).to_crs(3031)
    
    pointInPolys = sjoin(lon_lat, shpfile, how='left')
    regions = pointInPolys.Regions.to_numpy().reshape(
        lon.shape[0], lon.shape[1])
    
    eais_mask = (regions == 'East')
    eais_mask01 = np.zeros(eais_mask.shape)
    eais_mask01[eais_mask] = 1
    
    wais_mask = (regions == 'West')
    wais_mask01 = np.zeros(wais_mask.shape)
    wais_mask01[wais_mask] = 1
    
    ap_mask = (regions == 'Peninsula')
    ap_mask01 = np.zeros(ap_mask.shape)
    ap_mask01[ap_mask] = 1
    
    ais_mask = (eais_mask | wais_mask | ap_mask)
    ais_mask01 = np.zeros(ais_mask.shape)
    ais_mask01[ais_mask] = 1
    
    eais_area = ncfile.cell_area.values[eais_mask].sum()
    wais_area = ncfile.cell_area.values[wais_mask].sum()
    ap_area = ncfile.cell_area.values[ap_mask].sum()
    
    return (lon, lat, eais_mask, eais_mask01, wais_mask, wais_mask01, ap_mask,
            ap_mask01, ais_mask, ais_mask01, eais_area, wais_area, ap_area)


'''
# Production run to create area and masks files
import pickle
(lon, lat, eais_mask, eais_mask01, wais_mask, wais_mask01, ap_mask,
 ap_mask01, ais_mask, ais_mask01, eais_area, wais_area, ap_area
 ) = create_ais_mask()

ais_masks = {'lon': lon, 'lat': lat, 'eais_mask': eais_mask,
             'eais_mask01': eais_mask01, 'wais_mask': wais_mask,
             'wais_mask01': wais_mask01, 'ap_mask': ap_mask,
             'ap_mask01': ap_mask01, 'ais_mask': ais_mask,
             'ais_mask01': ais_mask01,}
ais_area = {'eais': eais_area, 'wais': wais_area, 'ap': ap_area,
            'ais': eais_area + wais_area + ap_area,}

with open('bas_palaeoclim_qino/others/ais_masks.pickle', 'wb') as handle:
    pickle.dump(ais_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('bas_palaeoclim_qino/others/ais_area.pickle', 'wb') as handle:
    pickle.dump(ais_area, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''


'''
# check
from a_basic_analysis.b_module.mapplot import (
    framework_plot1,hemisphere_plot,)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pickle

with open('others/ais_masks.pickle', 'rb') as handle:
    ais_masks = pickle.load(handle)
with open('others/ais_area.pickle', 'rb') as handle:
    ais_area = pickle.load(handle)
ais_area
# 9761642.045593847, 9620521.647740476, 2110804.571811703, 2038875.6430063567, 232127.50065176177, 232678.32105463636

fig, ax = hemisphere_plot(
    northextent=-60,
    add_grid_labels=False, plot_scalebar=False, grid_color='black',
    fm_left=0.06, fm_right=0.94, fm_bottom=0.04, fm_top=0.98, add_atlas=False,)

# fig, ax = framework_plot1("global")
ax.contour(
    ais_masks['lon'], ais_masks['lat'], ais_masks['eais_mask01'],
    colors='blue', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
ax.contour(
    ais_masks['lon'], ais_masks['lat'], ais_masks['wais_mask01'],
    colors='red', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
ax.contour(
    ais_masks['lon'], ais_masks['lat'], ais_masks['ap_mask01'],
    colors='yellow', levels=np.array([0.5]),
    transform=ccrs.PlateCarree(), linewidths=0.5, linestyles='solid')
coastline = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '10m', edgecolor='black',
    facecolor='none', lw=0.25)
ax.add_feature(coastline, zorder=2)

fig.savefig('figures/0_test/trial.png')
'''


'''
# derivation
one_degree_grids_cdo_area = xr.open_dataset(
    'bas_palaeoclim_qino/others/one_degree_grids_cdo_area.nc')
lon, lat = np.meshgrid(
    one_degree_grids_cdo_area.lon, one_degree_grids_cdo_area.lat, )

ais_imbie2 = gpd.read_file(
    'bas_palaeoclim_qino/observations/products/IMBIE_2016_drainage_basins/Rignot_Basins/ANT_IceSheets_IMBIE2/ANT_IceSheets_IMBIE2_v1.6.shp'
)

#1 geopandas.tools.sjoin point and polygon geometry
lon_lat = gpd.GeoDataFrame(
    crs="EPSG:4326", geometry=gpd.points_from_xy(
        lon.reshape(-1, 1), lat.reshape(-1, 1))).to_crs(3031)

from geopandas.tools import sjoin
pointInPolys = sjoin(lon_lat, ais_imbie2, how='left')
# pointInPolys = pointInPolys.groupby([pointInPolys.index], as_index=False).nth(0)
regions = pointInPolys.Regions.to_numpy().reshape(lon.shape[0], lon.shape[1])
# pointInPolys.Regions.value_counts(dropna=False)

eais_mask = (regions == 'East')
eais_mask01 = np.zeros(eais_mask.shape)
eais_mask01[eais_mask] = 1

wais_mask = (regions == 'West')
wais_mask01 = np.zeros(wais_mask.shape)
wais_mask01[wais_mask] = 1

ap_mask = (regions == 'Peninsula')
ap_mask01 = np.zeros(ap_mask.shape)
ap_mask01[ap_mask] = 1

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region functions to create a diverging color map

def rb_colormap(pltlevel, right_c = 'Blues_r', left_c = 'Reds'):
    '''
    ----Input
    pltlevel: levels used for the color bar, must be even;
    
    ----output
    cmp_cmap:
    '''
    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    
    cmp_top = cm.get_cmap(right_c, int(np.floor(len(pltlevel) / 2)))
    cmp_bottom = cm.get_cmap(left_c, int(np.floor(len(pltlevel) / 2)))
    
    cmp_colors = np.vstack(
        (cmp_top(np.linspace(0, 1, int(np.floor(len(pltlevel) / 2)))),
        #  [1, 1, 1, 1],
        #  [1, 1, 1, 1],
        #  [1, 1, 1, 1],
         cmp_bottom(np.linspace(0, 1, int(np.floor(len(pltlevel) / 2))))))
    cmp_cmap = ListedColormap(cmp_colors)
    
    return(cmp_cmap)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region calculate normalized sincoslon


#-------- import data

pre_weighted_sinlon = {}
pre_weighted_sinlon_ann = {}
pre_weighted_sinlon_sea = {}
pre_weighted_sinlon_am = {}
pre_weighted_coslon = {}
pre_weighted_coslon_ann = {}
pre_weighted_coslon_sea = {}
pre_weighted_coslon_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_sinlon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon.nc')
    pre_weighted_sinlon_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon_ann.nc')
    pre_weighted_sinlon_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon_sea.nc')
    pre_weighted_sinlon_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_sinlon_am.nc')
    pre_weighted_coslon[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon.nc')
    pre_weighted_coslon_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon_ann.nc')
    pre_weighted_coslon_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon_sea.nc')
    pre_weighted_coslon_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_coslon_am.nc')


pre_weighted_norm_sincoslon = {}
pre_weighted_norm_sincoslon_ann = {}
pre_weighted_norm_sincoslon_sea = {}
pre_weighted_norm_sincoslon_am = {}


i = 0
expid[i]
#-------- monthly

pre_weighted_norm_sincoslon[expid[i]] = xr.concat([
    pre_weighted_sinlon[expid[i]].pre_weighted_sinlon,
    pre_weighted_coslon[expid[i]].pre_weighted_coslon,], dim='sincos')
stats.describe(np.linalg.norm(pre_weighted_norm_sincoslon[expid[i]], axis=0, keepdims=True), axis=None, nan_policy='omit')

pre_weighted_norm_sincoslon[expid[i]].values[:] = pre_weighted_norm_sincoslon[expid[i]].values[:] / np.linalg.norm(pre_weighted_norm_sincoslon[expid[i]], axis=0, keepdims=True)
stats.describe(np.linalg.norm(pre_weighted_norm_sincoslon[expid[i]], axis=0, keepdims=True), axis=None, nan_policy='omit')

pre_weighted_norm_sincoslon[expid[i]] = pre_weighted_norm_sincoslon[expid[i]].rename('pre_weighted_norm_sincoslon')

pre_weighted_norm_sincoslon[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_norm_sincoslon.nc'
)


#-------- Annual

pre_weighted_norm_sincoslon_ann[expid[i]] = xr.concat([
    pre_weighted_sinlon_ann[expid[i]].pre_weighted_sinlon_ann,
    pre_weighted_coslon_ann[expid[i]].pre_weighted_coslon_ann,], dim='sincos')
stats.describe(np.linalg.norm(pre_weighted_norm_sincoslon_ann[expid[i]], axis=0, keepdims=True), axis=None, nan_policy='omit')

pre_weighted_norm_sincoslon_ann[expid[i]].values[:] = pre_weighted_norm_sincoslon_ann[expid[i]].values[:] / np.linalg.norm(pre_weighted_norm_sincoslon_ann[expid[i]], axis=0, keepdims=True)
stats.describe(np.linalg.norm(pre_weighted_norm_sincoslon_ann[expid[i]], axis=0, keepdims=True), axis=None, nan_policy='omit')

pre_weighted_norm_sincoslon_ann[expid[i]] = pre_weighted_norm_sincoslon_ann[expid[i]].rename('pre_weighted_norm_sincoslon_ann')

pre_weighted_norm_sincoslon_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_norm_sincoslon_ann.nc'
)


#-------- seasonal

pre_weighted_norm_sincoslon_sea[expid[i]] = xr.concat([
    pre_weighted_sinlon_sea[expid[i]].pre_weighted_sinlon_sea,
    pre_weighted_coslon_sea[expid[i]].pre_weighted_coslon_sea,], dim='sincos')
stats.describe(np.linalg.norm(pre_weighted_norm_sincoslon_sea[expid[i]], axis=0, keepdims=True), axis=None, nan_policy='omit')

pre_weighted_norm_sincoslon_sea[expid[i]].values[:] = pre_weighted_norm_sincoslon_sea[expid[i]].values[:] / np.linalg.norm(pre_weighted_norm_sincoslon_sea[expid[i]], axis=0, keepdims=True)
stats.describe(np.linalg.norm(pre_weighted_norm_sincoslon_sea[expid[i]], axis=0, keepdims=True), axis=None, nan_policy='omit')

pre_weighted_norm_sincoslon_sea[expid[i]] = pre_weighted_norm_sincoslon_sea[expid[i]].rename('pre_weighted_norm_sincoslon_sea')

pre_weighted_norm_sincoslon_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_norm_sincoslon_sea.nc'
)


#-------- Annual mean

pre_weighted_norm_sincoslon_am[expid[i]] = xr.concat([
    pre_weighted_sinlon_am[expid[i]].pre_weighted_sinlon_am,
    pre_weighted_coslon_am[expid[i]].pre_weighted_coslon_am,], dim='sincos')
stats.describe(np.linalg.norm(pre_weighted_norm_sincoslon_am[expid[i]], axis=0, keepdims=True), axis=None, nan_policy='omit')

pre_weighted_norm_sincoslon_am[expid[i]].values[:] = pre_weighted_norm_sincoslon_am[expid[i]].values[:] / np.linalg.norm(pre_weighted_norm_sincoslon_am[expid[i]], axis=0, keepdims=True)
stats.describe(np.linalg.norm(pre_weighted_norm_sincoslon_am[expid[i]], axis=0, keepdims=True), axis=None, nan_policy='omit')

pre_weighted_norm_sincoslon_am[expid[i]] = pre_weighted_norm_sincoslon_am[expid[i]].rename('pre_weighted_norm_sincoslon_am')

pre_weighted_norm_sincoslon_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_norm_sincoslon_am.nc'
)


'''
#-------- check two ways of calculating source lon
i = 0
expid[i]
pre_weighted_norm_sincoslon_am = {}
pre_weighted_lon_am = {}
pre_weighted_norm_sincoslon_am[expid[i]] = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_norm_sincoslon_am.nc')
pre_weighted_lon_am[expid[i]] = xr.open_dataset(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon_am.nc')


test = np.arctan2(
    pre_weighted_norm_sincoslon_am[expid[i]].pre_weighted_norm_sincoslon_am[0],
    pre_weighted_norm_sincoslon_am[expid[i]].pre_weighted_norm_sincoslon_am[1],
    )  * 180 / np.pi

test.values[test.values < 0] = test.values[test.values < 0] + 360

(pre_weighted_lon_am[expid[i]].pre_weighted_lon_am.values == test.values).all()

test1 = pre_weighted_lon_am[expid[i]].pre_weighted_lon_am.values - test.values
wheremax = np.where(abs(test1) == np.max(abs(test1)))
np.max(abs(test1))
test1[wheremax]
pre_weighted_lon_am[expid[i]].pre_weighted_lon_am.values[wheremax]
test.values[wheremax]

#-------- check the function np.linalg.norm
(np.sqrt((pre_weighted_norm_sincoslon[expid[i]] ** 2).sum(axis=0)).values == np.linalg.norm(pre_weighted_norm_sincoslon[expid[i]], axis=0)).all()

    #-------- calculate differences in normalized sin-/coslon
    
    # diff_pre_weighted_sincoslon_sea[expid[i+1]] = pre_weighted_norm_sincoslon_sea[expid[i+1]].pre_weighted_norm_sincoslon_sea - pre_weighted_norm_sincoslon_sea[expid[0]].pre_weighted_norm_sincoslon_sea
    # diff_pre_weighted_sincoslon_am[expid[i+1]] = pre_weighted_norm_sincoslon_am[expid[i+1]].pre_weighted_norm_sincoslon_am - pre_weighted_norm_sincoslon_am[expid[0]].pre_weighted_norm_sincoslon_am
    
    #-------- calculate the corresponding differences in lon
    
    # diff_pre_weighted_lon_sea[expid[i+1]] = np.arctan2(
    #     diff_pre_weighted_sincoslon_sea[expid[i+1]][0],
    #     diff_pre_weighted_sincoslon_sea[expid[i+1]][1]) * 180 / np.pi
    # diff_pre_weighted_lon_sea[expid[i+1]] = diff_pre_weighted_lon_sea[expid[i+1]].rename('diff_pre_weighted_lon_sea')
    # diff_pre_weighted_lon_sea[expid[i+1]].to_netcdf(
    #     exp_odir + expid[i+1] + '/analysis/echam/' + expid[i+1] + '.diff_pre_weighted_lon_sea.nc'
    # )
    
    # diff_pre_weighted_lon_am[expid[i+1]] = np.arctan2(
    #     diff_pre_weighted_sincoslon_am[expid[i+1]][0],
    #     diff_pre_weighted_sincoslon_am[expid[i+1]][1]) * 180 / np.pi
    # diff_pre_weighted_lon_am[expid[i+1]] = diff_pre_weighted_lon_am[expid[i+1]].rename('diff_pre_weighted_lon_am')
    # diff_pre_weighted_lon_am[expid[i+1]].to_netcdf(
    #     exp_odir + expid[i+1] + '/analysis/echam/' + expid[i+1] + '.diff_pre_weighted_lon_am.nc'
    # )

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate source lat - 58 lat bins

i=0
expid[i]

lat = exp_org_o[expid[i]]['wiso'].lat

ocean_pre = {}
lat_binned_pre = {}
pre_weighted_lat = {}
ocean_pre_ann = {}
lat_binned_pre_ann = {}
pre_weighted_lat_ann = {}
ocean_pre_sea = {}
lat_binned_pre_sea = {}
pre_weighted_lat_sea = {}
ocean_pre_am = {}
lat_binned_pre_am = {}
pre_weighted_lat_am = {}


lat_binned_pre[expid[i]] = xr.concat([
    (exp_org_o['pi_echam6_1y_213_3.60']['wiso'].wisoaprl[:, 4:, :, :] + \
        exp_org_o['pi_echam6_1y_213_3.60']['wiso'].wisoaprc[:, 4:, :, :]),
    (exp_org_o['pi_echam6_1y_212_3.60']['wiso'].wisoaprl[:, 4:, :, :] + \
        exp_org_o['pi_echam6_1y_212_3.60']['wiso'].wisoaprc[:, 4:, :, :]),
    ], dim="wisotype")

ocean_pre[expid[i]] = lat_binned_pre[expid[i]].sum(axis=1)


#---------------- monthly values

pre_weighted_lat[expid[i]] = \
    (lat_binned_pre[expid[i]] * lat.values[None, :, None, None]
     ).sum(axis=1) / ocean_pre[expid[i]]

pre_weighted_lat[expid[i]].values[
    np.where(ocean_pre[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat[expid[i]] = \
    pre_weighted_lat[expid[i]].rename('pre_weighted_lat')

pre_weighted_lat[expid[i]].to_netcdf(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc')



#---------------- annual values
ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
lat_binned_pre_ann[expid[i]] = lat_binned_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_lat_ann[expid[i]] = (lat_binned_pre_ann[expid[i]] * lat.values[None, :, None, None]).sum(axis=1) / ocean_pre_ann[expid[i]]
pre_weighted_lat_ann[expid[i]].values[np.where(ocean_pre_ann[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_ann[expid[i]] = pre_weighted_lat_ann[expid[i]].rename('pre_weighted_lat_ann')

pre_weighted_lat_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc'
)


#---------------- seasonal values

ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)
lat_binned_pre_sea[expid[i]] = lat_binned_pre[expid[i]][12:, :, :, :].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_lat_sea[expid[i]] = (lat_binned_pre_sea[expid[i]] * lat.values[None, :, None, None]).sum(axis=1) / ocean_pre_sea[expid[i]]
pre_weighted_lat_sea[expid[i]].values[np.where(ocean_pre_sea[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_sea[expid[i]] = pre_weighted_lat_sea[expid[i]].rename('pre_weighted_lat_sea')
pre_weighted_lat_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc'
)


#---------------- annual mean values

ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)
lat_binned_pre_am[expid[i]] = lat_binned_pre[expid[i]][12:, :, :, :].mean(dim="time", skipna=True)

pre_weighted_lat_am[expid[i]] = (lat_binned_pre_am[expid[i]] * lat.values[:, None, None]).sum(axis=0) / ocean_pre_am[expid[i]]
pre_weighted_lat_am[expid[i]].values[np.where(ocean_pre_am[expid[i]] < 1e-9)] = np.nan
pre_weighted_lat_am[expid[i]] = pre_weighted_lat_am[expid[i]].rename('pre_weighted_lat_am')
pre_weighted_lat_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc'
)

'''

# 'pi_echam6_1y_213_3.60'
lat[:48]
lat[48:]

'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot Jun/Sep/Dec source lat

#-------- import data

pre_weighted_lat = {}
# pre_weighted_lat_ann = {}
# pre_weighted_lat_sea = {}
# pre_weighted_lat_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lat[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc')
    # pre_weighted_lat_ann[expid[i]] = xr.open_dataset(
    #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc')
    # pre_weighted_lat_sea[expid[i]] = xr.open_dataset(
    #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc')
    # pre_weighted_lat_am[expid[i]] = xr.open_dataset(
    #     exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc')


#-------- basic set

lon = pre_weighted_lat[expid[0]].lon
lat = pre_weighted_lat[expid[1]].lat
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.1_lat/' + '6.1.0.1.1_pre_weighted_lat_compare_different_bins.png'
cbar_label1 = 'Precipitation-weighted source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted source latitude [$°$]'

pltlevel = np.arange(-80, 80.1, 10)
pltticks = np.arange(-80, 80.1, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)).reversed()

pltlevel2 = np.arange(-5, 5.01, 0.5)
pltticks2 = np.arange(-5, 5.01, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)).reversed()

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


#-------- scaled values J-S-D
axs[0, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[5, ],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[8, ],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[11, ],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)


#-------- Differences to 10 degree bin values J-S-D
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[5, ] - \
        pre_weighted_lat[expid[1]].pre_weighted_lat[5, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[8, ] - \
        pre_weighted_lat[expid[1]].pre_weighted_lat[8, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[11, ] - \
        pre_weighted_lat[expid[1]].pre_weighted_lat[11, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)


#-------- Differences to 10 degree bin values J-S-D
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[5, ] - \
        pre_weighted_lat[expid[2]].pre_weighted_lat[5, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[8, ] - \
        pre_weighted_lat[expid[2]].pre_weighted_lat[8, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_lat[expid[0]].pre_weighted_lat[11, ] - \
        pre_weighted_lat[expid[2]].pre_weighted_lat[11, ],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Jun', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Sep', transform=axs[0, 1].transAxes,
    ha='center', va='center', rotation='horizontal')
plt.text(
    0.5, 1.05, 'Dec', transform=axs[0, 2].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    -0.05, 0.5, 'Scaling tag map with latitude', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.08, 0.5, 'Differences with partitioning tag\nmap with $10°$ latitude bins', transform=axs[1, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)
plt.text(
    -0.08, 0.5, 'Differences with partitioning\ntag map with each latitude', transform=axs[2, 0].transAxes,
    ha='center', va='center', rotation='vertical', linespacing=1.5)

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




'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot lat initial version

i = 0
minsst = -90
maxsst = 90
# ocean_pre1 = (exp_org_o[expid[i]]['echam'].aprl[-1, :, :] + exp_org_o[expid[i]]['echam'].aprc[-1, :, :]) - (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 3, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 3, :, :])
ocean_pre1 = (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 4:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 4:, :, :]).sum(axis=0)

tsw1 = (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 4, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 4, :, :]) / ocean_pre1 * (maxsst - minsst) + minsst
# stats.describe(ocean_pre1, axis=None)
# np.where(ocean_pre1 < 1e-15)
# ocean_pre1[43, 186]
tsw1.values[np.where(ocean_pre1 < 1e-9)] = np.nan
stats.describe(tsw1, axis=None, nan_policy='omit') # 271.81658443 - 301.80649666
tsw1.to_netcdf('/work/ollie/qigao001/0_backup/lat1.nc')


pltlevel = np.arange(-90, 90.01, 0.1)
pltticks = np.arange(-90, 90.01, 10)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tsw1.lon,
    tsw1.lat,
    tsw1,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Precipitation-weighted source lat [$°$]\n1 year simulation, last month, [0, 400]',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    '0_backup/trial.png')


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot am/DJF/JJA source lat

#-------- import data

pre_weighted_lat = {}
pre_weighted_lat_ann = {}
pre_weighted_lat_sea = {}
pre_weighted_lat_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_lat[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat.nc')
    pre_weighted_lat_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_ann.nc')
    pre_weighted_lat_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_sea.nc')
    pre_weighted_lat_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lat_am.nc')


#-------- basic set
i = 0
j = 1
lon = pre_weighted_lat[expid[i]].lon
lat = pre_weighted_lat[expid[i]].lat
print('#-------- ' + expid[i] + ' & '+ expid[j])
mpl.rc('font', family='Times New Roman', size=10)


#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.1_lat/' + '6.1.0.1.0_' + expid[i] + '_and_' + expid[j] + '_pre_weighted_lat_compare.png'
cbar_label1 = 'Precipitation-weighted source latitude [$°$]'
cbar_label2 = 'Differences in precipitation-weighted source latitude [$°$]'


pltlevel = np.arange(-80, 80.1, 10)
pltticks = np.arange(-80, 80.1, 20)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('PuOr', len(pltlevel)).reversed()

pltlevel2 = np.arange(-5, 5.01, 0.5)
pltticks2 = np.arange(-5, 5.01, 1)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('PiYG', len(pltlevel2)).reversed()


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
    lon, lat, pre_weighted_lat_am[expid[i]].pre_weighted_lat_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_lat_am[expid[j]].pre_weighted_lat_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_lat_am[expid[i]].pre_weighted_lat_am - \
        pre_weighted_lat_am[expid[j]].pre_weighted_lat_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- DJF values
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[j]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='DJF') - pre_weighted_lat_sea[expid[j]].pre_weighted_lat_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- JJA values
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[j]].pre_weighted_lat_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_lat_sea[expid[i]].pre_weighted_lat_sea.sel(season='JJA') - pre_weighted_lat_sea[expid[j]].pre_weighted_lat_sea.sel(season='JJA'),
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
    -0.05, 0.5, 'Scaling tag map with latitude', transform=axs[0, 0].transAxes,
    ha='center', va='center', rotation='vertical')
plt.text(
    -0.05, 0.5, 'Partitioning tag map with latitude', transform=axs[1, 0].transAxes,
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

'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region calculate source SST - 58 SST bins

i = 0
expid[i]

sstbins = np.concatenate((np.array([-100]), np.arange(0, 28.1, 0.5), np.array([100])))
# sstbins_mid = np.arange(-0.25, 28.251, 0.5)
sstbins_mid = np.concatenate((np.array([271.38 - zerok]), np.arange(0.25, 28.251, 0.5)))

ocean_pre = {}
sst_binned_pre = {}
pre_weighted_tsw = {}
ocean_pre_ann = {}
sst_binned_pre_ann = {}
pre_weighted_tsw_ann = {}
ocean_pre_sea = {}
sst_binned_pre_sea = {}
pre_weighted_tsw_sea = {}
ocean_pre_am = {}
sst_binned_pre_am = {}
pre_weighted_tsw_am = {}


sst_binned_pre[expid[i]] = (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 4:, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[:, 4:, :, :])
ocean_pre[expid[i]] = sst_binned_pre[expid[i]].sum(axis=1)

#---------------- monthly values

pre_weighted_tsw[expid[i]] = ( sst_binned_pre[expid[i]] * sstbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre[expid[i]]
pre_weighted_tsw[expid[i]].values[ocean_pre[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw[expid[i]] = pre_weighted_tsw[expid[i]].rename('pre_weighted_tsw')
pre_weighted_tsw[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc'
)

#---------------- annual values
sst_binned_pre_ann[expid[i]] = sst_binned_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)
ocean_pre_ann[expid[i]] = ocean_pre[expid[i]].groupby('time.year').sum(dim="time", skipna=True)

pre_weighted_tsw_ann[expid[i]] = ( sst_binned_pre_ann[expid[i]] * sstbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre_ann[expid[i]]
pre_weighted_tsw_ann[expid[i]].values[ocean_pre_ann[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_ann[expid[i]] = pre_weighted_tsw_ann[expid[i]].rename('pre_weighted_tsw_ann')
pre_weighted_tsw_ann[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc'
)

#---------------- seasonal values
# spin up: one year

sst_binned_pre_sea[expid[i]] = sst_binned_pre[expid[i]][12:, :, :, :].groupby('time.season').sum(dim="time", skipna=True)
ocean_pre_sea[expid[i]] = ocean_pre[expid[i]][12:, :, :].groupby('time.season').sum(dim="time", skipna=True)

pre_weighted_tsw_sea[expid[i]] = ( sst_binned_pre_sea[expid[i]] * sstbins_mid[None, :, None, None]).sum(axis=1) / ocean_pre_sea[expid[i]]
pre_weighted_tsw_sea[expid[i]].values[ocean_pre_sea[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_sea[expid[i]] = pre_weighted_tsw_sea[expid[i]].rename('pre_weighted_tsw_sea')
pre_weighted_tsw_sea[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc'
)

#---------------- annual mean values

# spin up: one year

sst_binned_pre_am[expid[i]] = sst_binned_pre[expid[i]][12:, :, :, :].mean(dim="time", skipna=True)
ocean_pre_am[expid[i]] = ocean_pre[expid[i]][12:, :, :].mean(dim="time", skipna=True)

pre_weighted_tsw_am[expid[i]] = ( sst_binned_pre_am[expid[i]] * sstbins_mid[:, None, None]).sum(axis=0) / ocean_pre_am[expid[i]]
pre_weighted_tsw_am[expid[i]].values[ocean_pre_am[expid[i]].values < 1e-9] = np.nan
pre_weighted_tsw_am[expid[i]] = pre_weighted_tsw_am[expid[i]].rename('pre_weighted_tsw_am')
pre_weighted_tsw_am[expid[i]].to_netcdf(
    exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc'
)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot 36th month source SST, two scaling factors


#-------- import data

pre_weighted_tsw = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_tsw[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc')


#-------- basic set

lon = pre_weighted_tsw[expid[0]].lon
lat = pre_weighted_tsw[expid[0]].lat

print('#-------- Control: ' + expid[0] + ' & ' + expid[1])
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration

output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2_' + expid[0] + '_and_' + expid[1] + '_pre_weighted_tsw_compare.png'
cbar_label1 = 'Precipitation-weighted source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted source SST [$°C$]'

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

pltlevel2 = np.arange(-2, 2.01, 0.01)
pltticks2 = np.arange(-2, 2.01, 0.5)
pltnorm2 = BoundaryNorm(pltlevel2, ncolors=len(pltlevel2), clip=False)
pltcmp2 = cm.get_cmap('BrBG', len(pltlevel2)).reversed()

nrow = 1
ncol = 3
fm_bottom = 2.5 / (4.6*nrow + 2.5)


fig, axs = plt.subplots(
    nrow, ncol, figsize=np.array([8.8*ncol, 4.6*nrow + 2.5]) / 2.54,
    subplot_kw={'projection': ccrs.PlateCarree()},
    gridspec_kw={'hspace': 0.15, 'wspace': 0.02},)

for jcol in range(ncol):
    axs[jcol] = globe_plot(ax_org = axs[jcol],
                                 add_grid_labels=False)


#-------- 12th month values
axs[0].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[35, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[1]].pre_weighted_tsw[35, :, :],
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)

#-------- differences
axs[2].pcolormesh(
    lon, lat, pre_weighted_tsw[expid[0]].pre_weighted_tsw[35, :, :] - pre_weighted_tsw[expid[1]].pre_weighted_tsw[35, :, :],
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [260, 310]',
    transform=axs[0].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'Scaling tag map with SST. Scaling factors: [0, 400]',
    transform=axs[1].transAxes,
    ha='center', va='center', rotation='horizontal')

plt.text(
    0.5, 1.05, 'Differences',
    transform=axs[2].transAxes,
    ha='center', va='center', rotation='horizontal')

cbar1 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(-0.2, 0.8), ticks=pltticks)
cbar1.ax.set_xlabel(cbar_label1, linespacing=2)

cbar2 = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm2, cmap=pltcmp2), ax=axs,
    orientation="horizontal",shrink=0.5,aspect=40,extend='both',
    anchor=(1.1,-3.6),ticks=pltticks2)
cbar2.ax.set_xlabel(cbar_label2, linespacing=2)

fig.subplots_adjust(left=0.01, right = 0.99, bottom = fm_bottom * 0.75, top = 0.92)
fig.savefig(output_png)


'''
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot am/DJF/JJA source SST

#-------- import data

pre_weighted_tsw = {}
pre_weighted_tsw_ann = {}
pre_weighted_tsw_sea = {}
pre_weighted_tsw_am = {}

for i in range(len(expid)):
    print('#-------- ' + expid[i])
    pre_weighted_tsw[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw.nc')
    pre_weighted_tsw_ann[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_ann.nc')
    pre_weighted_tsw_sea[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_sea.nc')
    pre_weighted_tsw_am[expid[i]] = xr.open_dataset(
        exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_tsw_am.nc')

# stats.describe(pre_weighted_tsw[expid[i]].pre_weighted_tsw, axis=None, nan_policy='omit')
# stats.describe(pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea, axis=None, nan_policy='omit')

#-------- basic set
i = 0
j = 1
lon = pre_weighted_tsw[expid[i]].lon
lat = pre_weighted_tsw[expid[i]].lat
print('#-------- ' + expid[i] + ' & '+ expid[j])
mpl.rc('font', family='Times New Roman', size=10)

#-------- plot configuration
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.2_' + expid[i] + '_and_' + expid[j] + '_pre_weighted_tsw_compare.png'
cbar_label1 = 'Precipitation-weighted source SST [$°C$]'
cbar_label2 = 'Differences in precipitation-weighted source SST [$°C$]'

pltlevel = np.arange(0, 32.01, 2)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()
# pltcmp = rb_colormap(pltlevel, right_c = 'Blues_r', left_c = 'Reds')

pltlevel2 = np.arange(-1, 1.01, 0.125)
pltticks2 = np.arange(-1, 1.01, 0.25)
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
    lon, lat, pre_weighted_tsw_am[expid[i]].pre_weighted_tsw_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[j]].pre_weighted_tsw_am,
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 0].pcolormesh(
    lon, lat, pre_weighted_tsw_am[expid[i]].pre_weighted_tsw_am - \
        pre_weighted_tsw_am[expid[j]].pre_weighted_tsw_am,
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- DJF values
axs[0, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[j]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 1].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='DJF') - pre_weighted_tsw_sea[expid[j]].pre_weighted_tsw_sea.sel(season='DJF'),
    norm=pltnorm2, cmap=pltcmp2,transform=ccrs.PlateCarree(),)

#-------- JJA values
axs[0, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[1, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[j]].pre_weighted_tsw_sea.sel(season='JJA'),
    norm=pltnorm, cmap=pltcmp,transform=ccrs.PlateCarree(),)
axs[2, 2].pcolormesh(
    lon, lat, pre_weighted_tsw_sea[expid[i]].pre_weighted_tsw_sea.sel(season='JJA') - pre_weighted_tsw_sea[expid[j]].pre_weighted_tsw_sea.sel(season='JJA'),
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# region plot source SST - SST bins



run_length = '12'
run_units = 'month'
time_step = 11 # for plot


plt_x = pre_weighted_tsw_bin[expid[i]].lon
plt_y = pre_weighted_tsw_bin[expid[i]].lat
plt_z = pre_weighted_tsw_bin[expid[i]][time_step, :, :]


#-------------------------------- global plot

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

cbar_label = 'Precipitation-weighted source SST [$°C$] in ' + expid[i] + '\n' + 'Run length: ' + run_length + ' ' + run_units + '; Time period: ' + str(time_step + 1) + '; Bins: [-2, 30, 2]'
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.1_' + expid[i] + '_pre_weighted_tsw_bin_' + run_length + '_' + run_units + '_' + str(time_step+1) + '.png'


fig, ax = globe_plot()
plt_cmp = ax.pcolormesh(plt_x, plt_y, plt_z, transform=ccrs.PlateCarree(),
                        norm=pltnorm, cmap=pltcmp,)
cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.2,)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label, linespacing=2)
fig.savefig(output_png)


#-------------------------------- Antarctic plot

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

cbar_label = 'Precipitation-weighted source SST [$°C$]\n' + expid[i]
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.1_' + expid[i] + '_pre_weighted_tsw_bin_' + run_length + '_' + run_units + '_' + str(time_step+1) + '_Antarctica.png'

fig, ax = hemisphere_plot(northextent=-45)
plt_cmp = ax.pcolormesh(plt_x, plt_y, plt_z, transform=ccrs.PlateCarree(),
                        norm=pltnorm, cmap=pltcmp,)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel(cbar_label, linespacing=2)
fig.savefig(output_png)





# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region plot source SST - scaling tagmap with SST

run_length = '12'
run_units = 'month'
time_step = 11 # for plot



plt_x = pre_weighted_tsw[expid[i]].lon
plt_y = pre_weighted_tsw[expid[i]].lat
plt_z = pre_weighted_tsw[expid[i]][time_step, :, :] - 273.15


#-------------------------------- global plot

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

cbar_label = 'Precipitation-weighted source SST [$°C$] in ' + expid[i] + '\n' + 'Run length: ' + run_length + ' ' + run_units + '; Time period: ' + str(time_step + 1) + '; Scaling factors: ' + '[' + str(minsst[expid[i]]) + ', ' + str(maxsst[expid[i]]) + ']'
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.0_' + expid[i] + '_pre_weighted_tsw_' + run_length + '_' + run_units + '_' + str(time_step+1) + '_' + str(minsst[expid[i]]) + '_' + str(maxsst[expid[i]]) + '_sum_pre.png'


fig, ax = globe_plot()
plt_cmp = ax.pcolormesh(plt_x, plt_y, plt_z, transform=ccrs.PlateCarree(),
                        norm=pltnorm, cmap=pltcmp,)
cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.7, ticks=pltticks, extend='both',
    pad=0.1, fraction=0.2,)
cbar.ax.tick_params(length=2, width=0.4)
cbar.ax.set_xlabel(cbar_label, linespacing=2)
fig.savefig(output_png)


#-------------------------------- Antarctic plot

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

cbar_label = 'Precipitation-weighted source SST [$°C$]\n' + expid[i]
output_png = 'figures/6_awi/6.1_echam6/6.1.0_tagging_test/6.1.0.0_sst/' + '6.1.0.0.0_' + expid[i] + '_pre_weighted_tsw_' + run_length + '_' + run_units + '_' + str(time_step+1) + '_' + str(minsst[expid[i]]) + '_' + str(maxsst[expid[i]]) + '_sum_pre_Antarctica.png'

fig, ax = hemisphere_plot(northextent=-45)
plt_cmp = ax.pcolormesh(plt_x, plt_y, plt_z, transform=ccrs.PlateCarree(),
                        norm=pltnorm, cmap=pltcmp,)

cbar = fig.colorbar(
    cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
    orientation="horizontal", shrink=0.9, ticks=pltticks, extend='both',
    pad=0.02, fraction=0.2,
    )
cbar.ax.set_xlabel(cbar_label, linespacing=2)
fig.savefig(output_png)





'''
# ocean_pre[expid[i]] = (
#     exp_org_o[expid[i]]['echam'].aprl[:, :, :] + \
#         exp_org_o[expid[i]]['echam'].aprc[:, :, :]) - \
#             (exp_org_o[expid[i]]['wiso'].wisoaprl[:, 3, :, :] + \
#                 exp_org_o[expid[i]]['wiso'].wisoaprc[:, 3, :, :])

# stats.describe(ocean_pre[expid[i]], axis=None)
# stats.describe(pre_weighted_tsw[expid[i]], axis=None, nan_policy='omit')
# pre_weighted_tsw[expid[i]].to_netcdf('0_backup/test.nc')

i = 0
minsst = 260
maxsst = 310

ocean_pre1 = (exp_org_o[expid[i]]['echam'].aprl[-1, :, :] + exp_org_o[expid[i]]['echam'].aprc[-1, :, :]) - (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 3, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 3, :, :])
tsw1 = (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 4, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 4, :, :]) / ocean_pre1 * (maxsst - minsst) + minsst
tsw1.values[np.where(ocean_pre1 < 1e-9)] = np.nan
stats.describe(tsw1, axis=None, nan_policy='omit') # 271.81658443 - 301.80649666


i=1
minsst = 0
maxsst = 400
ocean_pre2 = (exp_org_o[expid[i]]['echam'].aprl[-1, :, :] + exp_org_o[expid[i]]['echam'].aprc[-1, :, :]) - (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 3, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 3, :, :])
tsw2 = (exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 4, :, :] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 4, :, :]) / ocean_pre2 * (maxsst - minsst) + minsst
tsw2.values[np.where(ocean_pre2 < 1e-9)] = np.nan
stats.describe(tsw2, axis=None, nan_policy='omit') # 229.00815916 - 304.1026493
tsw2.to_netcdf('0_backup/tsw2.nc')
test = tsw1 - tsw2
stats.describe(test, axis=None, nan_policy='omit')
test.to_netcdf('0_backup/test.nc')

np.where(test>50)
tsw1[94, 79] # 281.60002634
tsw2[94, 79] # 229.00815916
ocean_pre2[94, 79] # 6.69987505e-08
(exp_org_o[expid[i]]['wiso'].wisoaprl[-1, 4, 94, 79] + exp_org_o[expid[i]]['wiso'].wisoaprc[-1, 4, 94, 79]) # 3.83581513e-08

pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tsw1.lon,
    tsw1.lat,
    tsw1 - 273.15,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Precipitation-weighted source SST [$°C$]\n1 year simulation, last month, [260. 310]',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    '0_backup/trial.png')



pltlevel = np.arange(0, 32.01, 0.1)
pltticks = np.arange(0, 32.01, 4)
pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

fig, ax = framework_plot1("global", figsize=np.array([8.8*2, 11]) / 2.54)

plt_cmp = ax.pcolormesh(
    tsw2.lon,
    tsw2.lat,
    tsw2 - 273.15,
    norm=pltnorm, cmap=pltcmp, rasterized=True, transform=ccrs.PlateCarree(),)

cbar = fig.colorbar(
    plt_cmp, ax=ax, orientation="horizontal", pad=0.06,
    fraction=0.09, shrink=0.6, aspect=40, anchor=(0.5, -0.6),
    ticks=pltticks, extend="both",)

cbar.ax.set_xlabel(
    'Precipitation-weighted source SST [$°C$]\n1 year simulation, last month, [0, 400]',
    linespacing=1.5)

fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.995)
fig.savefig(
    '0_backup/trial2.png')
'''
# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------

