

# -----------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')
import pickle
import xarray as xr
from scipy import stats
import numpy as np
import pycircstat as circ
from scipy.stats import vonmises
from a_basic_analysis.b_module.mapplot import (
    globe_plot,
    hemisphere_plot,
)

# -----------------------------------------------------------------------------

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_416_4.9',
    ]
i = 0

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon = pickle.load(f)


# -----------------------------------------------------------------------------

#-------- Parametric Watson-Williams multi-sample test for equal means
# assumes underlying von-Mises distributrions.
# All groups are assumed to have a common concentration parameter k.

wwtest_all = circ.watson_williams(
    pre_weighted_lon['sea'][3::4, ].values * np.pi / 180,
    pre_weighted_lon['sea'][1::4, ].values * np.pi / 180,
    axis=0,
)[0] < 0.05
wwtest_all.sum() / (len(wwtest_all.flatten()))


# with open('scratch/test.npy', 'wb') as f:
#     np.save(f, wwtest_all)

#-------- Uniformity test: uniform?

rayleightest_all = circ.rayleigh(pre_weighted_lon['ann'].values * np.pi / 180, axis=0)[0] < 0.05
rayleightest_all.sum() / (len(rayleightest_all.flatten()))

# np.where(rayleightest_all == False)


raospacingtest_all = circ.raospacing(pre_weighted_lon['ann'].values * np.pi / 180, axis=0)[0] < 0.05
raospacingtest_all.sum() / (len(raospacingtest_all.flatten()))




import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600
mpl.rc('font', family='Times New Roman', size=9)
mpl.rcParams['axes.linewidth'] = 0.2
import cartopy.crs as ccrs

# pltlevel = np.arange(0, 32.01, 0.1)
# pltticks = np.arange(0, 32.01, 4)
# pltnorm = BoundaryNorm(pltlevel, ncolors=len(pltlevel), clip=False)
# pltcmp = cm.get_cmap('RdBu', len(pltlevel)).reversed()

fig, ax = globe_plot()

plt_cmp = ax.pcolormesh(
    pre_weighted_lon['ann'].lon,
    pre_weighted_lon['ann'].lat,
    # rayleightest_all,
    # raospacingtest_all,
    wwtest_all,
    transform=ccrs.PlateCarree(),)

# cbar = fig.colorbar(
#     cm.ScalarMappable(norm=pltnorm, cmap=pltcmp), ax=ax, aspect=30,
#     orientation="horizontal", shrink=0.7, ticks=pltticks, extend='max',
#     pad=0.1, fraction=0.2,
#     )
# cbar.ax.tick_params(length=2, width=0.4)
# cbar.ax.set_xlabel('1st line\n2nd line', linespacing=2)
fig.savefig('figures/trial.png')












#-------- 'Normality' von Mises distribution?
ilat = 45
ilon = 90
data = pre_weighted_lon['ann'][:, ilat, ilon].values * np.pi / 180
# data = vonmises.rvs(kappa=3, size=1000)
# data = np.linspace(0, 2 * np.pi, 1000)

kappa, loc, scale = vonmises.fit(data, fscale=1,)

stats.cramervonmises(data, vonmises.cdf, args=(kappa, loc, scale),).pvalue > 0.05


# expensive to run
def check_vonmises_3d(array, return_frc = True):
    '''
    #---- Input
    array: 3-D numpy.ndarray. von Mises distribution is checked along the 1st dimension
    
    #---- Output
    
    '''
    
    import numpy as np
    from scipy import stats
    
    whether_vonmises = np.full(array.shape[1:], True)
    
    for ilat in range(whether_vonmises.shape[0]):
        for ilon in range(whether_vonmises.shape[1]):
            # ilat = 48; ilon = 96
            test_data = array[:, ilat, ilon][np.isfinite(array[:, ilat, ilon])]

            if (len(test_data) < 3):
                whether_vonmises[ilat, ilon] = False
            else:
                kappa, loc, scale = vonmises.fit(test_data, fscale=1,)
                whether_vonmises[ilat, ilon] = stats.cramervonmises(
                    test_data,
                    vonmises.cdf,
                    args=(kappa, loc, scale),).pvalue > 0.05
    
    if return_frc:
        vonmises_frc = whether_vonmises.sum() / len(whether_vonmises.flatten())
        return(vonmises_frc)
    else:
        return(whether_vonmises)

whether_vonmises = check_vonmises_3d(
    pre_weighted_lon['ann'].values * np.pi / 180, return_frc=False)

# array = pre_weighted_lon['ann'].values * np.pi / 180





'''
#-------- check Watson-Williams test
ilat = 45
ilon = 90
wwtest = circ.watson_williams(
    pre_weighted_lon['sea'][3::4, ilat, ilon].values * np.pi / 180,
    pre_weighted_lon['sea'][1::4, ilat, ilon].values * np.pi / 180,
)
wwtest_all[ilat, ilon]
wwtest[0] < 0.05


#-------- basic statistics calculation
ilat = 48
ilon = 96

# mean: 195.26321425905067
stats.circmean(
    pre_weighted_lon['ann'][:, ilat, ilon].values,
    low=0, high=360, nan_policy='omit',
    )

# variance: 5.046245541107108e-05
stats.circvar(
    pre_weighted_lon['ann'][:, ilat, ilon].values,
    low=0, high=360, nan_policy='omit',
)

# kappa: 9908.60655698
kappa = circ.distributions.kappa(
    pre_weighted_lon['ann'][:, ilat, ilon].values * np.pi / 180)

# vonmises.mean(kappa, loc, scale) * 180 / np.pi # 195.26321426865007
# vonmises.stats(kappa, loc, scale, moments='mvsk')
# vonmises.var(kappa, loc, scale) # 0.00010092745853577095


#-------- Use R in python

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

base = importr('base')
utils = importr('utils')
circular = importr('circular')

utils.chooseCRANmirror(ind=1)
packnames = ('circular', 'ggplot2', 'hexbin')

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

#-------- check cumulative distribution function
cdf = vonmises.cdf(data, kappa, loc, scale,)
circ.raospacing(cdf * 2 * np.pi)[0] > 0.05
circ.rayleigh(cdf * 2 * np.pi)[0]

#-------- check raospacing test

ilat = 45
ilon = 90
data = pre_weighted_lon['ann'][:, ilat, ilon].values * np.pi / 180

circ.raospacing(data)[0] < 0.05
raospacingtest_all[ilat, ilon]

#-------- check rayleigh

ilat = 45
ilon = 90
data = pre_weighted_lon['ann'][:, ilat, ilon].values * np.pi / 180
# data = vonmises.rvs(kappa=3, size=1000)
# data = np.linspace(0, 2 * np.pi, 1000)

circ.rayleigh(data)[0] < 0.05
rayleightest_all[ilat, ilon]

'''



