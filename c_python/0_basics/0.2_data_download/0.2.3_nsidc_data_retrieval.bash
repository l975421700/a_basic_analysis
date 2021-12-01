


Date:2021.11.30
Data: Sea Ice Index, Version 3
Website: https://nsidc.org/data/G02135/versions/3

Instruction: https://nsidc.org/support/64231694-FTP-Client-Data-Access
# How to access data using the command line, check the folder place
Steps:
ftp sidads.colorado.edu
anonymous
cd /DATASETS/NOAA/G02135/south/monthly/geotiff
binary

# download with wget
wget --ftp-user=anonymous -r -nd ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/south/monthly/geotiff



