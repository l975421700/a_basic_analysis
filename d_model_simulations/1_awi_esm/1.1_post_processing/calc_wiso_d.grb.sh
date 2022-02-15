
#
# **** calc_wiso_d.e5-wiso ****
#
# Script calculates for every time step delta values of several water isotope fields for ECHAM5-wiso data
#
# externals: ${cdo}-Routines by U.Schulzweida (MPI MET, Hamburg)
#
#
# set file paths
cdo="/global/AWIsoft/cdo/1.9.2/bin/cdo"
CODES_WISO="/work/ollie/qigao001/startdump/wiso/calc_wiso_d/CODES.WISO"
CODES_WISO_TXT="/work/ollie/qigao001/startdump/wiso/calc_wiso_d/CODES.WISO.txt"
SMOW_FAC="/work/ollie/qigao001/startdump/wiso/calc_wiso_d/SMOW.FAC.T63.nwiso_3.nc"
#
IN=${1}_echam.nc
IN_WISO=${1}_wiso.nc
OUT_WISO=${1}_wiso_d.nc
#
#
TMPDIR=tmp
#
mkdir ${TMPDIR}
cd ${TMPDIR}
# pwd
#
# echo "JOB CALC_WISO_D IS RUNNING IN "${TMPDIR}
#
#
# echo " "
# echo "*** CALCULATE DELTA VALUES OF FILE: "$IN" ***"
# echo " "
#
# get the following standard ECHAM water fluxes and resevoirs:
# large scale precip., convective precip., snow fall, evaporation, runoff, soil wetness, snow depth
#
echo "Get ECHAM default fields"
# echo " "
#
# convert  precipitation, evaporation and runoff fields from kg/m**2s to mm/month
# (conversion factor: 3600*24*365/12*1000/1000 = 2.628e6)
#
# convert 2m-temperature and surface temperature from K to C)
#
${cdo} -s -f nc -subc,273.15  -selcode,167 $IN temp2
${cdo} -s -f nc -subc,273.15  -selcode,169 $IN tsurf
${cdo} -s -f nc -mulc,2.628e6 -selcode,142 $IN aprl
${cdo} -s -f nc -mulc,2.628e6 -selcode,143 $IN aprc
${cdo} -s -f nc -mulc,2.628e6 -selcode,144 $IN aprs
${cdo} -s -f nc -mulc,2.628e6 -selcode,182 $IN evap
${cdo} -s -f nc               -selcode,140 $IN ws
${cdo} -s -f nc               -selcode,141 $IN sn
#
# calculate total precipitation
${cdo} -s -f nc -chvar,aprl,aprt -chcode,142,260 -add aprl aprc aprt
#
# calculate total precipitation - evaporation
${cdo} -s -f nc -chvar,aprt,pe -chcode,260,265 -add aprt evap pe
#
#
# get related water isotope fields and reservoirs
#
echo "Get ECHAM water isotope fields"
# echo " "
#
${cdo} -s -f nc -mulc,2.628e6 -selcode,53 $IN_WISO wisoaprl
${cdo} -s -f nc -mulc,2.628e6 -selcode,54 $IN_WISO wisoaprc
${cdo} -s -f nc -mulc,2.628e6 -selcode,55 $IN_WISO wisoaprs
${cdo} -s -f nc -mulc,2.628e6 -selcode,59 $IN_WISO wisoevap
${cdo} -s -f nc               -selcode,51 $IN_WISO wisows
${cdo} -s -f nc               -selcode,52 $IN_WISO wisosn
${cdo} -s -f nc               -selcode,75 $IN_WISO snglac
${cdo} -s -f nc               -selcode,76 $IN_WISO wisosnglac
#
# calculate total precipitation
${cdo} -s -f nc -chvar,wisoaprl,wisoaprt -chcode,53,50 -add wisoaprl wisoaprc wisoaprt
#
# calculate total precipitation - evaporation
${cdo} -s -f nc -chvar,wisoaprt,wisope -chcode,50,60 -add wisoaprt wisoevap wisope
#
#
# set ECHAM defaults fields to 0. if flux fields (reservoirs) are less than 0.05 mm/month (0.05 mm)
#
${cdo} -s -f nc -setmisstoc,0. -ifthen -gec,0.05 aprt aprt dummy; mv dummy aprt
${cdo} -s -f nc -setmisstoc,0. -ifthen -gec,0.05 aprl aprl dummy; mv dummy aprl
${cdo} -s -f nc -setmisstoc,0. -ifthen -gec,0.05 aprc aprc dummy; mv dummy aprc
${cdo} -s -f nc -setmisstoc,0. -ifthen -gec,0.05 aprs aprs dummy; mv dummy aprs
${cdo} -s -f nc -setmisstoc,0. -ifthen -gec,0.05 evap evap dummy1; ${cdo} -s -f nc -setmisstoc,0. -ifthen -lec,-0.05 evap evap dummy2; ${cdo} -s -f nc -add dummy1 dummy2 evap; rm dummy1 dummy2
${cdo} -s -f nc -setmisstoc,0. -ifthen -gec,0.05 pe   pe   dummy1; ${cdo} -s -f nc -setmisstoc,0. -ifthen -lec,-0.05 pe   pe   dummy2; ${cdo} -s -f nc -add dummy1 dummy2 pe  ; rm dummy1 dummy2
${cdo} -s -f nc -setmisstoc,0. -ifthen -gec,5.e-5 ws ws dummy; mv dummy ws
${cdo} -s -f nc -setmisstoc,0. -ifthen -gec,5.e-5 sn sn dummy; mv dummy sn
${cdo} -s -f nc -setmisstoc,0. -ifthen -gec,5.e-5 snglac snglac dummy; mv dummy snglac
#
# set water isotope fields to 0. if ECHAM default fields (reservoirs) are zero
# (split into different isotope fields (=levels) to achieve correct assigment of isotope records to aprt records)
#
${cdo} -s -f nc splitlevel wisoaprt dummy.l; for lev in 1 2 3; do ${cdo} -s -f nc -setmisstoc,0. -ifthen -nec,0. aprt dummy.l00000${lev}.nc dummy.${lev}; done; ${cdo} -s -f nc -O merge dummy.? wisoaprt; rm dummy.*
${cdo} -s -f nc splitlevel wisoaprl dummy.l; for lev in 1 2 3; do ${cdo} -s -f nc -setmisstoc,0. -ifthen -nec,0. aprl dummy.l00000${lev}.nc dummy.${lev}; done; ${cdo} -s -f nc -O merge dummy.? wisoaprl; rm dummy.*
${cdo} -s -f nc splitlevel wisoaprc dummy.l; for lev in 1 2 3; do ${cdo} -s -f nc -setmisstoc,0. -ifthen -nec,0. aprc dummy.l00000${lev}.nc dummy.${lev}; done; ${cdo} -s -f nc -O merge dummy.? wisoaprc; rm dummy.*
${cdo} -s -f nc splitlevel wisoaprs dummy.l; for lev in 1 2 3; do ${cdo} -s -f nc -setmisstoc,0. -ifthen -nec,0. aprs dummy.l00000${lev}.nc dummy.${lev}; done; ${cdo} -s -f nc -O merge dummy.? wisoaprs; rm dummy.*
${cdo} -s -f nc splitlevel wisoevap dummy.l; for lev in 1 2 3; do ${cdo} -s -f nc -setmisstoc,0. -ifthen -nec,0. evap dummy.l00000${lev}.nc dummy.${lev}; done; ${cdo} -s -f nc -O merge dummy.? wisoevap; rm dummy.*
${cdo} -s -f nc splitlevel wisope   dummy.l; for lev in 1 2 3; do ${cdo} -s -f nc -setmisstoc,0. -ifthen -nec,0. pe   dummy.l00000${lev}.nc dummy.${lev}; done; ${cdo} -s -f nc -O merge dummy.? wisope;   rm dummy.*
${cdo} -s -f nc splitlevel wisows   dummy.l; for lev in 1 2 3; do ${cdo} -s -f nc -setmisstoc,0. -ifthen -nec,0. ws   dummy.l00000${lev}.nc dummy.${lev}; done; ${cdo} -s -f nc -O merge dummy.? wisows;   rm dummy.*
${cdo} -s -f nc splitlevel wisosn   dummy.l; for lev in 1 2 3; do ${cdo} -s -f nc -setmisstoc,0. -ifthen -nec,0. sn   dummy.l00000${lev}.nc dummy.${lev}; done; ${cdo} -s -f nc -O merge dummy.? wisosn;   rm dummy.*
${cdo} -s -f nc splitlevel wisosnglac dummy.l; for lev in 1 2 3; do ${cdo} -s -f nc -setmisstoc,0. -ifthen -nec,0. snglac dummy.l00000${lev}.nc dummy.${lev}; done; ${cdo} -s -f nc -O merge dummy.? wisosnglac; rm dummy.*
#
# calculate delta values of all water isotope fields (reference standard: SMOW)
# - the SMOW values have to be stored in the file ${SMOW_FAC} (with the correct grid size & order of isotope values!)
#
echo "Calculate delta values of water isotope fields"
# echo " "
#
${cdo} -s -f nc -chvar,wisoaprt,wisoaprt_d     -chcode,50,10 -mulc,1000 -subc,1. -div -div wisoaprt   aprt   ${SMOW_FAC} wisoaprt_d
${cdo} -s -f nc -chvar,wisoaprl,wisoaprl_d     -chcode,53,13 -mulc,1000 -subc,1. -div -div wisoaprl   aprl   ${SMOW_FAC} wisoaprl_d
${cdo} -s -f nc -chvar,wisoaprc,wisoaprc_d     -chcode,54,14 -mulc,1000 -subc,1. -div -div wisoaprc   aprc   ${SMOW_FAC} wisoaprc_d
${cdo} -s -f nc -chvar,wisoaprs,wisoaprs_d     -chcode,55,15 -mulc,1000 -subc,1. -div -div wisoaprs   aprs   ${SMOW_FAC} wisoaprs_d
${cdo} -s -f nc -chvar,wisoevap,wisoevap_d     -chcode,59,19 -mulc,1000 -subc,1. -div -div wisoevap   evap   ${SMOW_FAC} wisoevap_d
${cdo} -s -f nc -chvar,wisope,wisope_d         -chcode,60,20 -mulc,1000 -subc,1. -div -div wisope     pe     ${SMOW_FAC} wisope_d
${cdo} -s -f nc -chvar,wisows,wisows_d         -chcode,51,11 -mulc,1000 -subc,1. -div -div wisows     ws     ${SMOW_FAC} wisows_d
${cdo} -s -f nc -chvar,wisosn,wisosn_d         -chcode,52,12 -mulc,1000 -subc,1. -div -div wisosn     sn     ${SMOW_FAC} wisosn_d
${cdo} -s -f nc -chvar,wisosnglac,wisosnglac_d -chcode,76,33 -mulc,1000 -subc,1. -div -div wisosnglac snglac ${SMOW_FAC} wisosnglac_d
#
#
# merge all files together to output file
#
echo "Merge all files and clean up"
# echo " "
#
${cdo} -s -f nc -t ${CODES_WISO} -merge \
   wisoaprt_d wisoaprl_d wisoaprc_d wisoaprs_d wisoevap_d wisope_d wisows_d wisosn_d wisosnglac_d \
   wisoaprt   wisoaprl   wisoaprc   wisoaprs   wisoevap   wisope   wisows   wisosn   wisosnglac   \
       aprt       aprl       aprc       aprs       evap       pe       ws       sn       snglac   \
   temp2 tsurf dummy
#
${cdo} -s -f nc -setpartabn,${CODES_WISO_TXT} dummy $OUT_WISO
rm dummy
#
# clean up
#
rm wisoaprt_d wisoaprl_d wisoaprc_d wisoaprs_d wisoevap_d wisope_d wisows_d wisosn_d wisosnglac_d
rm wisoaprt   wisoaprl   wisoaprc   wisoaprs   wisoevap   wisope   wisows   wisosn   wisosnglac
rm     aprt       aprl       aprc       aprs       evap       pe       ws       sn       snglac
rm temp2 tsurf
#
# echo "*** CALCULATION DONE... ***"
# echo " "
#
cd ..
rmdir ${TMPDIR}
#
exit

