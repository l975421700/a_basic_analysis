

# -----------------------------------------------------------------------------
# region on Monsoon

# Suite Discovery and Management: rosie go
rosie go

# Standard Suite Output
# To run Rose Bush on Monsoon run:
firefox http://localhost/rose-bush
# Rose Bush on PUMA
http://puma.nerc.ac.uk/rose-bush

# Compilation output
~/cylc-run/<suitename>/share/fcm_make/fcm-make2.log

# Standard output
~/cylc-run/<suitename>/log/job/1/atmos/NN/job.out
~/cylc-run/<suitename>/log/job/1/atmos/NN/job.err


# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region fcm

fcm branch-create -k <ticket> <branch_name> fcm:um.x-tr@vn11.8

# https://code.metoffice.gov.uk/svn/um/main/branches/dev/qingganggao/vn11.8_tutorial
fcm checkout URL


# endregion
# -----------------------------------------------------------------------------


