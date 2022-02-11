

import pyfesom2 as pf
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import xarray as xr
from a_basic_analysis.b_module.mapplot import (
    hemisphere_plot,
)
import cartopy.crs as ccrs


sst_fesom_pi_final_qg = xr.open_dataset('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/sst.fesom.200001_202912.nc')
data_to_plot = sst_fesom_pi_final_qg.sst[0, ].copy()


#### trial 3.2
mesh = pf.load_mesh('startdump/core2/')
box=[-180, 180, -90, 90]
box_mesh = [box[0] - 1, box[1] + 1, box[2] - 1, box[3] + 1]
left, right, down, up = box_mesh
selection = ((mesh.x2 >= left) & (mesh.x2 <= right) & (mesh.y2 >= down)
    & (mesh.y2 <= up))
elem_selection = selection[mesh.elem]
no_nan_triangles = np.all(elem_selection, axis=1)
elem_no_nan = mesh.elem[no_nan_triangles]
d = mesh.x2[elem_no_nan].max(axis=1) - mesh.x2[elem_no_nan].min(axis=1)
no_cyclic_elem2 = np.argwhere(d < 100).ravel()
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

fig.savefig('figures/0_test/trial.png')





#### trial 3.1
mesh = pf.load_mesh('startdump/core2/')
sst_fesom_pi_final_qg = xr.open_dataset('output/awiesm-2.1-wiso/pi_final_qg/analysis/fesom/sst.fesom.200001_202912.nc')

box=[-180, 180, 60, 90]
box_mesh = [box[0] - 1, box[1] + 1, box[2] - 1, box[3] + 1]
left, right, down, up = box_mesh
selection = ((mesh.x2 >= left) & (mesh.x2 <= right) & (mesh.y2 >= down)
    & (mesh.y2 <= up))
elem_selection = selection[mesh.elem]
no_nan_triangles = np.all(elem_selection, axis=1)
elem_no_nan = mesh.elem[no_nan_triangles]
d = mesh.x2[elem_no_nan].max(axis=1) - mesh.x2[elem_no_nan].min(axis=1)
no_cyclic_elem2 = np.argwhere(d < 100).ravel()


data_to_plot = sst_fesom_pi_final_qg.sst[0, ].copy()
# data_to_plot[data_to_plot == 0] = np.nan
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

fig.savefig('figures/0_test/trial.png')


'''
# mesh.x2
# mesh.y2
# mesh.elem.shape
'''

