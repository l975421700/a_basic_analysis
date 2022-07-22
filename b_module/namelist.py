
import numpy as np
import cartopy as ctp

# Seasons
# -----------------------------------------------------------------------------

month = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

seasons = np.array(['DJF', 'MAM', 'JJA', 'SON'])

hours = ['00', '01', '02', '03', '04', '05',
         '06', '07', '08', '09', '10', '11',
         '12', '13', '14', '15', '16', '17',
         '18', '19', '20', '21', '22', '23',
         ]

months = ['01', '02', '03', '04', '05', '06',
          '07', '08', '09', '10', '11', '12']

month_days = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

# computation
# -----------------------------------------------------------------------------

quantiles = np.array(
    ([0, 5, 10, 25, 50, 75, 90, 95, 100],
     ['0%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '100%',
      'ptp100_0', 'ptp95_5', 'ptp90_10', 'ptp75_25',
      'mean', 'std']), dtype=object)


# folder
# -----------------------------------------------------------------------------


# Physical constants
# -----------------------------------------------------------------------------

g = 9.80665  # gravity [m/s^2]
m = 0.0289644  # Molar mass of dry air [kg/mol]
r0 = 8.314462618  # Universal gas constant [J/(mol·K)]
cp = 1004.68506  # specific heat of air at constant pressure [J/(kg·K)]
r = 287.0  # gas constant of air [J/kgK]
r_v = 461.0  # gas cons tant of vapor [J/kgK]
p0sl = 100000.0  # pressure at see level
t0sl = 288.1499938964844   # temperature at see level

zerok = 273.15

# plot
# -----------------------------------------------------------------------------



