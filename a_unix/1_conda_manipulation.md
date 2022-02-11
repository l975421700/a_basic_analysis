
# miniconda management on ollie

## Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)


## Create a conda environment.

Activate conda:

    $ source miniconda3/bin/activate

Check which conda and python are activated:

    $ which conda
    $ which python

Create a conda Environment (training) and activate it:

    $ conda create --name training
    $ conda activate training

## Install necessary python packages.
Firstly, [mamba](https://github.com/mamba-org/mamba) is recommended for installing packages as an alternative. When you install mamba, the newest python will also be installed in this conda environment. It is recommended to use 'conda-forge' channel to ensure consistency across packages.

When you issue each of the folloing command, you will be informed which packages will be downloaded/upgraded/downgraded. And you will be asked whether you agree or not. It is possible to add ' -y' to the end of each command to agree the installation all the time. But it is better to be aware of the processes in case some important package is downgraded.

    $ conda install -c conda-forge mamba
    $ mamba install -c conda-forge matplotlib
    $ mamba install -c conda-forge dask
    $ mamba install -c conda-forge xarray
    $ mamba install -c conda-forge seaborn
    $ mamba install -c conda-forge cartopy
    $ mamba install -c conda-forge netcdf4
    $ mamba install -c conda-forge metpy
    $ mamba install -c conda-forge nco
    $ mamba install -c conda-forge pynco
    $ mamba install -c conda-forge cdsapi
    $ mamba install -c conda-forge pytest
    $ mamba install -c conda-forge ipykernel
    $ mamba install -c conda-forge pylint
    $ mamba install -c conda-forge autopep8
    $ mamba install -c conda-forge haversine
    $ mamba install -c conda-forge h5netcdf
    $ mamba install -c conda-forge pyhdf
    $ mamba install -c conda-forge eofs
    $ mamba install -c conda-forge scikit-learn
    $ mamba install -c conda-forge cdo
    $ mamba install -c conda-forge python-cdo
    $ mamba install -c conda-forge windrose
    $ mamba install -c conda-forge cfgrib
    $ mamba install -c conda-forge ecmwf-api-client
    $ mamba install -c conda-forge geopy
    $ mamba install -c conda-forge jupyter
    $ mamba install -c conda-forge ffmpeg
    $ mamba install -c conda-forge pywavelets
    $ mamba install -c conda-forge pytables
    $ mamba install -c conda-forge gcc_impl_linux-64
    $ mamba install -c conda-forge libgcc
    $ mamba install -c conda-forge pybind11
    $ mamba install -c conda-forge pycwt
    $ mamba install -c conda-forge mscorefonts
    $ mamba install -c conda-forge line_profiler
    $ mamba install -c conda-forge xesmf
    <!-- $ mamba install -c conda-forge proplot -->
    $ mamba install -c conda-forge cmip6_preprocessing
    $ mamba install -c conda-forge bottleneck
    $ mamba install -c conda-forge gh
    <!-- $ mamba install -c ncas -c conda-forge cf-python cf-plot udunits2 -->
    $ mamba install -c conda-forge mpich esmpy
    <!-- $ pip install cf-view -->
    $ mamba install -c conda-forge nc-time-axis
    $ mamba install -c conda-forge jupyter_contrib_nbextensions
    $ mamba install -c conda-forge rasterio
    $ mamba install -c conda-forge rockhound
    $ mamba install -c conda-forge satpy
    $ mamba install -c conda-forge esmvaltool
    $ mamba install -c conda-forge rioxarray
    $ mamba install -c conda-forge pyfesom2
    $ mamba install -c conda-forge geopandas
    $ mamba install -c conda-forge pymannkendall
    $ mamba install -c pyviz holoviews bokeh
    $ mamba install -c pyviz geoviews
    $ mamba install -c conda-forge cmocean
    $ mamba install -c pyviz hvplot
    $ mamba install -c conda-forge numpy=1.21
    $ mamba install -c conda-forge pynio



## Other useful command

List available conda environment

    $ conda env list

List installed packages:

    $ conda list

Don't run, unless you wanna remove the conda environment

    $ conda env remove -n training

Update conda

    $ conda update -n base -c defaults conda

Remove unused packages and caches.

    $ conda clean -t -i


## Test for pyfesom2

    $ conda create --name pyfesom2
    $ conda activate pyfesom2
    $ conda install -c conda-forge pyfesom2
    $ conda install -c pyviz holoviews bokeh
    $ conda install -c pyviz geoviews
    $ conda install -c pyviz hvplot
    $ conda install -c conda-forge xarray
    $ conda install -c conda-forge matplotlib
    $ conda install -c conda-forge pandas
    $ conda install -c conda-forge cartopy
    $ conda install -c conda-forge numpy=1.21
    $ conda install -c conda-forge mamba
