

# -----------------------------------------------------------------------------
# region miniconda management

# bash Miniconda3-latest-Linux-x86_64.sh
# conda create --name deepice
# conda env list
# conda env remove -n deepice

# source ${HOME}/miniconda3/bin/activate deepice

conda install -c conda-forge python=3.10.6 -y
conda install -c conda-forge mamba -y
mamba install -c conda-forge xesmf -y
mamba install -c conda-forge matplotlib -y
mamba install -c conda-forge dask -y
mamba install -c conda-forge xarray -y
mamba install -c conda-forge seaborn -y
mamba install -c conda-forge cartopy -y
mamba install -c conda-forge netcdf4 -y
mamba install -c conda-forge metpy -y
mamba install -c conda-forge cdsapi -y
mamba install -c conda-forge haversine -y
mamba install -c conda-forge h5netcdf -y
mamba install -c conda-forge pyhdf -y
mamba install -c conda-forge cdo -y
mamba install -c conda-forge windrose -y
mamba install -c conda-forge cfgrib -y
mamba install -c conda-forge geopy -y
mamba install -c conda-forge jupyter -y
mamba install -c conda-forge ffmpeg -y
mamba install -c conda-forge pywavelets -y
mamba install -c conda-forge pytables -y
mamba install -c conda-forge mscorefonts -y
mamba install -c conda-forge line_profiler -y
mamba install -c conda-forge notebook -y
mamba install -c conda-forge rasterio -y
mamba install -c conda-forge satpy -y
mamba install -c conda-forge rioxarray -y
mamba install -c conda-forge pyfesom2 -y
mamba install -c conda-forge geopandas -y
mamba install -c conda-forge pymannkendall -y
mamba install -c conda-forge fortls -y
mamba install -c conda-forge ncl -y
mamba install -c conda-forge mne -y
mamba install -c conda-forge radian -y
mamba install -c conda-forge libgcc -y
mamba install -c conda-forge nose -y
pip install pycircstat
pip install rpy2
mamba install -c conda-forge jupyterlab -y
mamba install -c conda-forge proplot -y
mamba install -c conda-forge cmip6_preprocessing -y
mamba install -c conda-forge pint-xarray -y


'''
#-------------------------------- TEST REGION

source ${HOME}/miniconda3/bin/activate test

#-------------------------------- Others

# clean conda installed pkgs
# conda clean -a

# pip install igra
# pip install siphon

# regridding
# https://github.com/JiaweiZhuang/xESMF/issues/47
# conda search -c conda-forge -f esmpy
# conda install -c conda-forge esmpy
# conda install -c conda-forge xesmf
# pytest -v --pyargs xesmf


# mamba install -c conda-forge nco
# mamba install -c conda-forge pynco
# mamba install -c conda-forge eofs
# mamba install -c conda-forge bottleneck
# mamba install -c conda-forge gh
# mamba install -c conda-forge mpich esmpy
# mamba install -c conda-forge nc-time-axis
# mamba install -c conda-forge rockhound
# mamba install -c conda-forge esmvaltool
# mamba install -c conda-forge pytest -y
# mamba install -c conda-forge ipykernel -y
# mamba install -c conda-forge pylint -y
# mamba install -c conda-forge autopep8 -y
# mamba install -c conda-forge ecmwf-api-client -y
# mamba install -c conda-forge gcc_impl_linux-64 -y
# mamba install -c conda-forge pycwt -y
# mamba install -c conda-forge proplot -y
# mamba install -c conda-forge jupyter_contrib_nbextensions -y
# mamba install -c conda-forge numpy=1.21 -y
# mamba install -c conda-forge python-cdo -y
# mamba install -c conda-forge scikit-learn -y
# mamba install -c conda-forge pybind11 -y


'''
# endregion
# -----------------------------------------------------------------------------


