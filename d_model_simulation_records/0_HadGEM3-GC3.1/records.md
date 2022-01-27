
# General procedures
1, Check suite info
2, check suite conf
3, check initial/boundary conditions, e.g. grep -nr '/data/d01/' .
4, check output/dump frequency


# u-ck900, copy of u-ar930, HadGEM3-GC3.1 N96ORCA1 PI Control for CMIP6, officially ported for Monsoon2


# u-cl202, copy of u-bi805, HadGEM3-GC3.1 N96ORCA1 ssp585 for CMIP6: With redistributed ozone - all stash except COSP
moo mkset -v moose:crum/u-cl202 -p=project-ukesm
<!-- http://cms.ncas.ac.uk/ticket/1742 -->


# u-cl295, copy of u-bg466, HadGEM3-GC3.1 N96ORCA1 HIST run CMIP6 member #1 : 1875+
moo mkset -v moose:crum/u-cl295 -p=project-ukesm


# u-cl317, copy of u-ba937, [Std Suite] HadGEM3-GC3.1 N96ORCA1 LIG127k for CMIP6 - PRODUCTION

<!-- # u-ck646, copy of u-ar766, HadGEM3-GC3.1 N96ORCA1 PI Control for CMIP6

Usage: Initial attempt to transfer MetO-cray suite to Monsoon
Result: Failed because of unknown problems. -->
