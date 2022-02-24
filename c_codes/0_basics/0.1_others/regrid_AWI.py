#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys
import glob
import subprocess


# In[3]:


files=files=glob.glob('/home/users/rahuls/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_*.nc')


# cdo genycon,global_2 siconc_SImon_AWI-ESM-1-1-LR_lig127k_r1i1p1f1_gn_300101-301012.nc regrid2deg_weights_AWI-ESM-1-1-LR,nc

# In[6]:


weightfile='/home/users/rahuls/PMIP_LIG/ESGF_download/CMIP6/model-output/seaIce/siconc/regrid2deg_weights_AWI-ESM-1-1-LR.nc'


# In[10]:


for infile in files:
    outfile=infile[:-3]+'_regrid.nc'
    optstring='remap,global_2,'+weightfile
    print(outfile)
    subprocess.call(['cdo',optstring,infile,outfile])


# In[ ]:




