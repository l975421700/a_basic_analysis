

# =============================================================================
# region miniconda management on jasmin 2021-10-17 (Conda 4.10.3 Python 3.9.5)

# bash Miniconda3-latest-Linux-x86_64.sh

# conda create --name deepice
# conda env list
# conda env remove -n deepice
# conda list

source /home/users/qino/miniconda3/bin/activate
conda activate deepice

conda install -c conda-forge mamba
mamba install -c conda-forge matplotlib
mamba install -c conda-forge dask
mamba install -c conda-forge xarray
mamba install -c conda-forge seaborn
mamba install -c conda-forge cartopy
mamba install -c conda-forge netcdf4
mamba install -c conda-forge metpy
mamba install -c conda-forge nco
mamba install -c conda-forge pynco
mamba install -c conda-forge cdsapi
mamba install -c conda-forge pytest
mamba install -c conda-forge ipykernel
mamba install -c conda-forge pylint
mamba install -c conda-forge autopep8
mamba install -c conda-forge haversine
mamba install -c conda-forge h5netcdf
mamba install -c conda-forge pyhdf
mamba install -c conda-forge eofs
mamba install -c conda-forge scikit-learn
mamba install -c conda-forge cdo
mamba install -c conda-forge python-cdo
mamba install -c conda-forge windrose
mamba install -c conda-forge cfgrib
mamba install -c conda-forge ecmwf-api-client
mamba install -c conda-forge geopy
mamba install -c conda-forge jupyter
mamba install -c conda-forge ffmpeg
mamba install -c conda-forge pywavelets
mamba install -c conda-forge pytables
mamba install -c conda-forge gcc_impl_linux-64
mamba install -c conda-forge libgcc
mamba install -c conda-forge pybind11
mamba install -c conda-forge pycwt
mamba install -c conda-forge mscorefonts
mamba install -c conda-forge line_profiler
mamba install -c conda-forge xesmf
mamba install -c conda-forge proplot
mamba install -c conda-forge cmip6_preprocessing
mamba install -c conda-forge bottleneck
mamba install -c conda-forge gh
# gh auth login
mamba install -c ncas -c conda-forge cf-python cf-plot udunits2
mamba install -c conda-forge mpich esmpy
pip install cf-view
mamba install -c conda-forge nc-time-axis
mamba install -c conda-forge jupyter_contrib_nbextensions
mamba install -c conda-forge notebook
mamba install -c conda-forge rasterio
mamba install -c conda-forge rockhound
mamba install -c conda-forge satpy
mamba install -c conda-forge esmvaltool
mamba install -c conda-forge rioxarray
mamba install -c conda-forge pyfesom2


# clean conda installed pkgs
# conda clean -a
'''
conda install basemap joblib seawater click


# jupyter contrib nbextension install --user
# jupyter nbextension enable spellchecker/main
# jupyter nbextension enable codefolding/main

pip install igra
pip install siphon
pip install fortran-language-server

# regridding
# https://github.com/JiaweiZhuang/xESMF/issues/47
# conda search -c conda-forge -f esmpy
# conda install -c conda-forge esmpy
# conda install -c conda-forge xesmf
# pytest -v --pyargs xesmf

mamba install -c conda-forge opencv
mamba install -c conda-forge python-geotiepoints
mamba install -c conda-forge urllib3

'''
# endregion
# =============================================================================

# =============================================================================
# region miniconda management on jasmin 2021-12-16 (Using pynio)

# conda create --name deepice_pynio --channel conda-forge/label/cf201901 pynio pyngl

conda activate deepice_pynio

conda install -c conda-forge mamba
mamba install -c conda-forge xarray
mamba install -c conda-forge dask
mamba install -c conda-forge ipykernel

# endregion
# =============================================================================

