

# -----------------------------------------------------------------------------
# region rsync

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
