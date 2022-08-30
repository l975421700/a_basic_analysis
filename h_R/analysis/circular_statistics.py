


import warnings
warnings.filterwarnings('ignore')
import pickle
import xarray as xr
from scipy import stats
import numpy as np
import pycircstat as circ
from scipy.stats import vonmises

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_416_4.9',
    ]
i = 0
# pre_weighted_lon = {}

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon = pickle.load(f)



ilat = 48
ilon = 96
# mean: 195.26321425905067
stats.circmean(
    pre_weighted_lon['ann'][:, ilat, ilon].values,
    low=0, high=360, nan_policy='omit',
    )

# circular variance: 5.046245541107108e-05
stats.circvar(
    pre_weighted_lon['ann'][:, ilat, ilon].values,
    low=0, high=360, nan_policy='omit',
)

data = pre_weighted_lon['ann'][:, ilat, ilon].values * np.pi / 180
data = vonmises.rvs(kappa=3, size=1000)
data = np.linspace(0, 2 * np.pi, 1000)
kappa, loc, scale = vonmises.fit(data, fscale=1,)
# stats.cramervonmises(data, vonmises.cdf, args=(kappa, loc, scale),).pvalue

# circ.rayleigh(data)[0]
# circ.raospacing(data)[0]

cdf = vonmises.cdf(data, kappa, loc, scale,)
circ.raospacing(cdf * 2 * np.pi)[0] > 0.05
circ.rayleigh(cdf * 2 * np.pi)[0]

'''
# kappa: 9908.60655698
kappa = circ.distributions.kappa(
    pre_weighted_lon['ann'][:, ilat, ilon].values * np.pi / 180)

vonmises.mean(kappa, loc, scale) * 180 / np.pi # 195.26321426865007
vonmises.stats(kappa, loc, scale, moments='mvsk')
vonmises.var(kappa, loc, scale) # 0.00010092745853577095

'''

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

base = importr('base')
utils = importr('utils')
circular = importr('circular')


utils.chooseCRANmirror(ind=1)
packnames = ('circular', 'ggplot2', 'hexbin')

from rpy2.robjects.vectors import StrVector

names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))



