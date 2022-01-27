

./app/nemo_cice/rose-app.conf:12:NEMO_START=/work/n02/n02/vittoria/lig127k_startfiles/au022o_22000101_restart.nc

./app/nemo_cice/rose-app.conf:8:NEMO_ICEBERGS_START=/work/n02/n02/vittoria/lig127k_startfiles/au022o_icebergs_22000101_restart.nc


# =============================================================================
# region /home/d05/amclaren/useful_command.txt on Monsoon



CUMF:

module load um_tools

mule-cumf --summary ~/cylc-run/u-cg123/share/data/History_Data/cg123a.da19880902_00 ~/cylc-run/u-cg092/share/data/History_Data/cg092a.da19880902_00

Deep conv only:

mule-cumf --summary ~/cylc-run/u-cg123/share/data/History_Data/cg123a.da19880902_00 ~/cylc-run/u-ci309/share/data/History_Data/ci309a.da19880902_00

DISK QUOTAS:

quota.py -u lustre_home

quota.py -g ukesm lustre_multi


DISK USAGE:

du -ks amclaren        - size of subdirectory 'amclaren'

du -k --max-depth=1    - size of directories one level down

XCONV:

/home/d00/jecole/bin/xconv

Checkout a branch:

fcm checkout fcm:um.x_br/dev/alisonmclauren/vn11.9_water_tracer_conv

# endregion
# =============================================================================

