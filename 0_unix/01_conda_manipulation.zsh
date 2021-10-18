

# =============================================================================
# region miniconda management on jasmin 2021-10-17 (Conda 4.10.3 Python 3.9.5)

# bash Miniconda3-latest-Linux-x86_64.sh
# source miniconda3/bin/activate # required on Mac

# conda create --name deepice211017 -y
# conda env list
# conda env remove -n deepice211017
# conda list
# conda search -c conda-forge -f esmpy

conda activate deepice211017

# https://github.com/JiaweiZhuang/xESMF/issues/47
conda install -c conda-forge esmpy=8.1.1 -y
conda install -c conda-forge xesmf=0.6.1 -y
conda install -c conda-forge dask=2021.9.1 -y
conda install -c conda-forge pytest=6.2.5 -y
pytest -v --pyargs xesmf

conda install -c conda-forge mamba=0.17.0 -y

mamba install -c conda-forge matplotlib -y
mamba install -c conda-forge ipykernel -y
mamba install -c conda-forge netcdf4 -y
mamba install -c conda-forge cartopy -y
mamba install -c conda-forge pylint -y
mamba install -c conda-forge autopep8 -y
mamba install -c conda-forge metpy -y
mamba install -c conda-forge haversine -y
mamba install -c conda-forge h5netcdf -y
mamba install -c conda-forge pyhdf -y
mamba install -c conda-forge nco -y
mamba install -c conda-forge pynco -y
mamba install -c conda-forge eofs -y
mamba install -c conda-forge cdsapi -y
mamba install -c conda-forge scikit-learn -y
mamba install -c conda-forge cdo -y
mamba install -c conda-forge python-cdo -y
mamba install -c conda-forge windrose -y
mamba install -c conda-forge cfgrib -y
mamba install -c conda-forge ecmwf-api-client -y
mamba install -c conda-forge geopy -y
mamba install -c conda-forge rasterio -y
mamba install -c conda-forge jupyter -y
mamba install -c conda-forge ffmpeg -y
mamba install -c conda-forge basemap -y
mamba install -c conda-forge pywavelets -y
mamba install -c conda-forge opencv -y
mamba install -c conda-forge pytables -y
mamba install -c conda-forge libgcc-ng -y
mamba install -c conda-forge gcc_impl_linux-64 -y
mamba install -c conda-forge libgcc -y
mamba install -c conda-forge pybind11 -y
mamba install -c conda-forge pycwt -y
mamba install -c conda-forge seaborn -y
mamba install -c conda-forge statsmodels -y
mamba install -c conda-forge python-geotiepoints -y
mamba install -c conda-forge urllib3 -y

pip install line_profiler
pip install satpy
pip install igra
pip install siphon
pip install fortran-language-server

# clean conda installed pkgs
# conda clean -t


# endregion
# =============================================================================

# =============================================================================
# region
# endregion
# =============================================================================

