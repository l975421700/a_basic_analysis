

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
# region backup ollie to jasmin

rsync -avzLP /work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_m_502_5.0 xfer1:/home/users/qino/output/echam-6.3.05p2-wiso/pi/

rsync -avzLP /work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_d_500_wiso xfer1:/home/users/qino/output/echam-6.3.05p2-wiso/pi/
rsync -avzLP /work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_d_501_5.0 xfer1:/home/users/qino/output/echam-6.3.05p2-wiso/pi/
rsync -avzLP /work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_m_416_4.9 xfer1:/home/users/qino/output/echam-6.3.05p2-wiso/pi/
rsync -avzLP /work/ollie/qigao001/output/echam-6.3.05p2-wiso/pi/pi_m_503_5.0 xfer1:/home/users/qino/output/echam-6.3.05p2-wiso/pi/

rsync -avzLP /work/ollie/qigao001/data_sources xfer1:/home/users/qino/

rsync -avzLP /work/ollie/qigao001/finse_school xfer1:/home/users/qino/

rsync -avzLP /work/ollie/qigao001/model_codes xfer1:/home/users/qino/

rsync -avzLP /work/ollie/qigao001/scratch/* xfer1:/home/users/qino/scratch/

rsync -avzLP /work/ollie/qigao001/startdump/* xfer1:/home/users/qino/startdump/

# endregion
# -----------------------------------------------------------------------------

