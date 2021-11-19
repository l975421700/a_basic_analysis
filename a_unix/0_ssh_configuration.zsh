

# =============================================================================
# region JASMIN

# ssh-add -l
# ssh-add .ssh/id_rsa
# ssh-copy-id -i .ssh/id_rsa.pub qino@server
ssh -A qino@login1.jasmin.ac.uk
ssh qino@sci1.jasmin.ac.uk

# /home/users/qino/miniconda3/envs/deepice211017/bin/python -c "from IPython import start_ipython; start_ipython()" --no-autoindent

ssh -X qino@mass-cli.jasmin.ac.uk

# /gws/nopw/j04/bas_palaeoclim
# ssh-keygen -t rsa -b 2048 -C "qino@bas.ac.uk" -f ~/.ssh/id_rsa

# module load jaspy
# module load jasmin-sci

# endregion
# =============================================================================


# =============================================================================
# region
ssh lander
ssh -XY xcsc

/projects/ukesm/qgao # project directory
/home/d05/qgao # home directory

# endregion
# =============================================================================

