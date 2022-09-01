

# -----------------------------------------------------------------------------
# region rsync download figures

# download figures
rsync -avzP sci8:/home/users/qino/figures /Users/gao/OneDrive\ -\ University\ of\ Cambridge/research/

rsync -avzP ollie:/work/ollie/qigao001/figures /Users/gao/Library/CloudStorage/OneDrive-UniversityofCambridge/research/

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region backup jasmin to lacie

rsync -avz sci3:/home/users/qino/a00_basic_analysis /Volumes/LaCie/jasmin/qino/
rsync -avz sci3:/home/users/qino/figures /Volumes/LaCie/jasmin/qino/
rsync -avz sci3:/home/users/qino/git_repo /Volumes/LaCie/jasmin/qino/

rsync -avz sci3:/gws/nopw/j04/bas_palaeoclim/qino/* /Volumes/LaCie/jasmin/qino/bas_palaeoclim_qino/

# endregion
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# region partial backup ollie to levante
# -l, --links                 copy symlinks as symlinks
# -F --exclude=PATTERN       exclude files matching PATTERN

rsync -avzLP --exclude '**/run_/*' /work/ollie/qigao001/output/archived_exp levante:/work/ab0246/a270200/output/

rsync -avzLP --exclude '**/run_/*' /work/ollie/qigao001/output levante:/work/ab0246/a270200/

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region partial backup ollie to jasmin

rsync -avzLP --exclude '**/run_/*' /work/ollie/qigao001/output/archived_exp xfer1:/gws/nopw/j04/bas_palaeoclim/qino/output/

rsync -avzLP --exclude '**/run_/*' /work/ollie/qigao001/output xfer1:/gws/nopw/j04/bas_palaeoclim/qino/

# endregion
# -----------------------------------------------------------------------------

