

source ${HOME}/miniconda3/bin/activate deepice


#---------------------------------------------------------------- LIG
# region
#-------- create ensemble mean

ls /home/a/a270064/bb1029/awiesm_exp/lig_final/outdata/fesom/sst*
# 2401 - 3001

cdo -ensmean /home/a/a270064/bb1029/awiesm_exp/lig_final/outdata/fesom/sst.fesom.29??01.01.nc /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.lig_final.nc

cdo -ensmean /home/a/a270064/bb1029/awiesm_exp/lig_final/outdata/fesom/a_ice.fesom.29??01.01.nc /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.lig_final.nc

cdo -f nc copy -timmean -chname,var72,wisosw_d -select,code=72 /home/a/a270064/bb1029/awiesm_exp/lig_final/outdata/echam/lig_final_29????.01_wiso /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.lig_final.nc


#-------- regrid and change attrs
# 2072 - 3100

cdo -setattribute,sst@units=K -addc,273.15 -remapcon,T63grid -setgrid,/work/ab0246/a270200/startdump/fesom2/mesh/mesh_CORE2_finaltopo_mean/sl.grid.CDO /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.lig_final.nc /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.lig_final_t63.nc

cdo -chname,a_ice,sic -mulc,100 -remapcon,T63grid -setgrid,/work/ab0246/a270200/startdump/fesom2/mesh/mesh_CORE2_finaltopo_mean/sl.grid.CDO /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.lig_final.nc /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.lig_final_t63.nc

# endregion


#---------------------------------------------------------------- PI
# region

#-------- create ensemble mean

ls /home/a/a270064/bb1029/awiesm_exp/pi_final/outdata/fesom/sst*

cdo -ensmean /home/a/a270064/bb1029/awiesm_exp/pi_final/outdata/fesom/sst.fesom.29??01.01.nc /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.pi_final.nc

cdo -ensmean /home/a/a270064/bb1029/awiesm_exp/pi_final/outdata/fesom/a_ice.fesom.29??01.01.nc /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.pi_final.nc

cdo -f nc copy -timmean -chname,var72,wisosw_d -select,code=72 /home/a/a270064/bb1029/awiesm_exp/pi_final/outdata/echam/pi_final_29????.01_wiso /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/wisosw_d.echam.2900_2999.pi_final.nc


#-------- regrid and change attrs

cdo -setattribute,sst@units=K -addc,273.15 -remapcon,T63grid -setgrid,/work/ab0246/a270200/startdump/fesom2/mesh/mesh_CORE2_finaltopo_mean/sl.grid.CDO /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.pi_final.nc /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sst.fesom.2900_2999.pi_final_t63.nc

cdo -chname,a_ice,sic -mulc,100 -remapcon,T63grid -setgrid,/work/ab0246/a270200/startdump/fesom2/mesh/mesh_CORE2_finaltopo_mean/sl.grid.CDO /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.pi_final.nc /work/ab0246/a270200/startdump/model_input/lig/bc_lig_pi_final/sic.fesom.2900_2999.pi_final_t63.nc

# endregion


#---------------------------------------------------------------- LGM
# region

#-------- create ensemble mean

ls /home/a/a270064/bb1029/AWIESM2.1-EXP/lgm-50/outdata/fesom/sst*
# 2000-2233

cdo -ensmean /home/a/a270064/bb1029/AWIESM2.1-EXP/lgm-50/outdata/fesom/sst.fesom.22[0-2]?01.01.nc /work/ab0246/a270200/startdump/model_input/lgm/lgm-50/sst.fesom.2200_2229.lgm-50.nc

cdo -ensmean /home/a/a270064/bb1029/AWIESM2.1-EXP/lgm-50/outdata/fesom/a_ice.fesom.22[0-2]?01.01.nc /work/ab0246/a270200/startdump/model_input/lgm/lgm-50/sic.fesom.2200_2229.lgm-50.nc

cdo -f nc copy -timmean -chname,var72,wisosw_d -select,code=72 /home/a/a270064/bb1029/AWIESM2.1-EXP/lgm-50/outdata/echam/MM_lgm-50_22[0-2]???.01_wiso /work/ab0246/a270200/startdump/model_input/lgm/lgm-50/wisosw_d.echam.2200_2229.lgm-50.nc

#-------- regrid and change attrs


cdo -setattribute,sst@units=K -addc,273.15 -remapdis,T63grid -setgrid,/work/ab0246/a270200/startdump/model_input/lgm/lgm-50/mesh_description_lgm.txt /work/ab0246/a270200/startdump/model_input/lgm/lgm-50/sst.fesom.2200_2229.lgm-50.nc /work/ab0246/a270200/startdump/model_input/lgm/lgm-50/sst.fesom.2200_2229.lgm-50_t63.nc

cdo -chname,a_ice,sic -mulc,100 -remapdis,T63grid -setgrid,/work/ab0246/a270200/startdump/model_input/lgm/lgm-50/mesh_description_lgm.txt /work/ab0246/a270200/startdump/model_input/lgm/lgm-50/sic.fesom.2200_2229.lgm-50.nc /work/ab0246/a270200/startdump/model_input/lgm/lgm-50/sic.fesom.2200_2229.lgm-50_t63.nc

# endregion



