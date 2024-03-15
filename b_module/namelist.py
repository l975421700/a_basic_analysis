
import numpy as np
import cartopy as ctp

# -----------------------------------------------------------------------------
# region Seasons

month = np.array(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
monthini = np.array(
    ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
month_num = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
month_dec = np.array(
    ['Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May',
     'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov',])
month_dec_num = np.array([12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

seasons = np.array(['DJF', 'MAM', 'JJA', 'SON'])
seasons_last_num = np.array([2, 5, 8, 11])

hours = ['00', '01', '02', '03', '04', '05',
         '06', '07', '08', '09', '10', '11',
         '12', '13', '14', '15', '16', '17',
         '18', '19', '20', '21', '22', '23',
         ]

months = ['01', '02', '03', '04', '05', '06',
          '07', '08', '09', '10', '11', '12']

month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

seconds_per_d = 86400

# endregion
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# region computation

quantiles = np.array(
    ([0, 5, 10, 25, 50, 75, 90, 95, 100],
     ['0%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '100%',
      'ptp100_0', 'ptp95_5', 'ptp90_10', 'ptp75_25',
      'mean', 'std']), dtype=object)

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region Physical constants

g = 9.80665  # gravity [m/s^2]
m = 0.0289644  # Molar mass of dry air [kg/mol]
r0 = 8.314462618  # Universal gas constant [J/(mol·K)]
cp = 1004.68506  # specific heat of air at constant pressure [J/(kg·K)]
r = 287.0  # gas constant of air [J/kgK]
r_v = 461.0  # gas cons tant of vapor [J/kgK]
p0sl = 100000.0  # pressure at see level
t0sl = 288.1499938964844   # temperature at see level

zerok = 273.15

# endregion
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# region plot

panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)',
                '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)',
                '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(u)',
                '(v)', '(w)', '(x)', '(y)', '(z)',
                ]


ten_sites_names = [
    'EDC', 'DOME F', 'Vostok', 'EDML', 'WDC',
    'Rothera', 'Halley', 'Neumayer', 'Law Dome', "Dumont d'Urville"]

marker_recs = {'EC': 'o', 'JH': 's', 'DC': 'v', 'MC': '^'}

plot_labels = {
    'dO18': '$\delta^{18}O$ [$‰$]',
    'd18O': '$\delta^{18}O$ [$‰$]',
    'dD': '$\delta D$ [$‰$]',
    'd_excess': '$d_{xs}$ [$‰$]',
    'd_xs': '$d_{xs}$ [$‰$]',
    'd_ln': '$d_{ln}$ [$‰$]',
    'lat': 'Source latitude [$°\;S$]',
    'lon': 'Source longitude [$°$]',
    'sst': 'Source SST [$°C$]',
    'RHsst': 'Source RHsst [$\%$]',
    'rh2m': 'Source rh2m [$\%$]',
    'wind10': 'Source vel10 [$m \; s^{-1}$]',
    'distance': 'Source-sink distance [$× 10^2 km$]',
    'temp2': '2 m temperature [$°C$]',
    'pressure': 'Pressure [$hPa$]',
    'wind_speed': 'Wind speed [$m \; s^{-1}$]',
    'wisoaprt': 'Precipitation [$mm \; day^{-1}$]',
    'temperature': 'Surface temperature [$°C$]',
    'accumulation': 'Accumulation [$cm \; year^{-1}$]',
    't_air': 'Air temperature [$°C$]',
    't_3m': '2 m temperature [$°C$]',
    'q': 'Specific humidity [$g/kg$]',
    'pre': 'Precipitation [$mm \; day^{-1}$]',
    'local_rh2m': 'rh2m [$\%$]',
}

plot_labels_no_unit = {
    'wisoaprt': 'Precipitation',
    'dO18': '$\delta^{18}O$',
    'd18O': '$\delta^{18}O$',
    'dD': '$\delta D$',
    'd_excess': '$d_{xs}$',
    'd_xs': '$d_{xs}$',
    'd_ln': '$d_{ln}$',
    'lat': 'Source latitude',
    'lon': 'Source longitude',
    'sst': 'Source SST',
    'RHsst': 'Source RHsst',
    'rh2m': 'Source rh2m',
    'wind10': 'Source vel10',
    'distance': 'Source-sink distance',
    'temp2': 'temp2',
    't_air': 'Air temperature',
    'q': 'Specific humidity',
}

plot_labels_only_unit = {
    'dO18': '$‰$',
    'd18O': '$‰$',
    'dD': '$‰$',
    'd_excess': '$‰$',
    'd_xs': '$‰$',
    'd_ln': '$‰$',
    'lat': '$°\;S$',
    'lon': '$°$',
    'sst': '$°C$',
    'RHsst': '$\%$',
    'rh2m': '$\%$',
    'wind10': ' $m \; s^{-1}$',
    'distance': ' $× 10^2 km$',
    'temp2': '$°C$',
    'pressure': ' $hPa$',
    'wind_speed': ' $m \; s^{-1}$',
    'wisoaprt': ' $mm \; day^{-1}$',
    'temperature': '$°C$',
    'accumulation': ' $cm \; year^{-1}$',
    't_air': '$°C$',
    't_3m': '$°C$',
    'q': ' $g/kg$',
    'pre': ' $mm \; day^{-1}$',
    'local_rh2m': '$\%$',
}

expid_labels = {
    'pi_600_5.0': '$PI_{control}$',
    'pi_601_5.1': '$PI_{smooth}$',
    'pi_602_5.2': '$PI_{rough}$',
    'pi_605_5.5': '$PI_{low\_ss}$',
    'pi_606_5.6': '$PI_{lowT}$',
    'pi_609_5.7': '$PI_{highT}$',
    
    'pi_603_5.3': '$PI_{no\_ss}$',

    'nudged_713_6.0_2yr': '$Nudge_{control}$',
    'nudged_712_6.0_k52_2yr': '$Nudge_{k52}$',
    'nudged_714_6.0_k52_88_2yr': '$Nudge_{k52\_88}$',
    'nudged_715_6.0_k43_2yr': '$Nudge_{k43}$',
    'nudged_716_6.0_I01_2yr': '$Nudge_{I01}$',
    'nudged_717_6.0_I03_2yr': '$Nudge_{I03}$',
    'nudged_718_6.0_S3_2yr': '$Nudge_{S3}$',
    'nudged_719_6.0_S6_2yr': '$Nudge_{S6}$',
    
    'nudged_705_6.0': '$Nudge_{control}$',
    'nudged_703_6.0_k52': '$Nudge_{k52}$',
    'nudged_706_6.0_k52_88': '$Nudge_{k52\_88}$',
    'nudged_707_6.0_k43': '$Nudge_{k43}$',
    'nudged_708_6.0_I01': '$Nudge_{I01}$',
    'nudged_709_6.0_I03': '$Nudge_{I03}$',
    'nudged_710_6.0_S3': '$Nudge_{S3}$',
    'nudged_711_6.0_S6': '$Nudge_{S6}$',

}

expid_colours = {
    'nudged_705_6.0':       'tab:blue',
    'nudged_703_6.0_k52':   'tab:orange',
    'nudged_707_6.0_k43':   'tab:green',
    'nudged_708_6.0_I01':   'tab:red',
    'nudged_709_6.0_I03':   'tab:purple',
    'nudged_710_6.0_S3':    'tab:brown',
    'nudged_711_6.0_S6':    'grey',
    
    # 'nudged_706_6.0_k52_88': 'tab:orange',
    'pi_600_5.0': 'black',
    'pi_601_5.1': 'tab:blue',
    'pi_602_5.2': 'tab:orange',
    'pi_605_5.5': 'tab:green',
    'pi_606_5.6': 'tab:red',
    'pi_609_5.7': 'tab:purple',
    'pi_603_5.3': 'tab:brown',
    'pi_610_5.8': 'grey',

}

time_labels = {
    'daily'         : 'daily',
    'mon'           : 'monthly',
    'mon no mm'     : 'monthly (w/o monthly mean)',
    'ann'           : 'annual',
    'ann no am'     : 'annual (w/o annual mean)',
}

# endregion
# -----------------------------------------------------------------------------


