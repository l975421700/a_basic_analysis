

# -----------------------------------------------------------------------------
# region JASMIN

# ssh-add -l
# ssh-add .ssh/id_rsa
# ssh-copy-id -i .ssh/id_rsa.pub qino@server
ssh -A qino@login1.jasmin.ac.uk
ssh qino@sci1.jasmin.ac.uk

# python -c "from IPython import start_ipython; start_ipython()" --no-autoindent

ssh -X qino@mass-cli.jasmin.ac.uk

# /gws/nopw/j04/bas_palaeoclim
# ssh-keygen -t rsa -b 2048 -C "qino@bas.ac.uk" -f ~/.ssh/id_rsa
# ssh-keygen -t ed25519 -b 2048 -C "gaoqg229@gmail.com" -f ~/.ssh/id_ed25519_dkrz

# module load jaspy
# module load jasmin-sci
# which xconv

# endregion
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# region monsoon
ssh lander
ssh -XY xcsc

/projects/ukesm/qgao # project directory
/home/d05/qgao # home directory

# endregion
# -----------------------------------------------------------------------------


