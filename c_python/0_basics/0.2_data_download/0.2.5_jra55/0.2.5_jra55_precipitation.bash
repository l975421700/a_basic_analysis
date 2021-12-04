#! /bin/csh -f
#
# c-shell script to download selected files from rda.ucar.edu using Wget
# NOTE: if you want to run under a different shell, make sure you change
#       the 'set' commands according to your shell's syntax
# after you save the file, don't forget to make it executable
#   i.e. - "chmod 755 <name_of_script>"
#
# Experienced Wget Users: add additional command-line flags here
#   Use the -r (--recursive) option with care
#   Do NOT use the -b (--background) option - simultaneous file downloads
#       can cause your data access to be blocked
set opts = "-N"
#
# Replace xxxxxx with your rda.ucar.edu password on the next uncommented line
# IMPORTANT NOTE:  If your password uses a special character that has special
#                  meaning to csh, you should escape it with a backslash
#                  Example:  set passwd = "my\!password"
set passwd = 'gaoQINGgang000'
set num_chars = `echo "$passwd" |awk '{print length($0)}'`
if ($num_chars == 0) then
  echo "You need to set your password before you can continue"
  echo "  see the documentation in the script"
  exit
endif
@ num = 1
set newpass = ""
while ($num <= $num_chars)
  set c = `echo "$passwd" |cut -b{$num}-{$num}`
  if ("$c" == "&") then
    set c = "%26";
  else
    if ("$c" == "?") then
      set c = "%3F"
    else
      if ("$c" == "=") then
        set c = "%3D"
      endif
    endif
  endif
  set newpass = "$newpass$c"
  @ num ++
end
set passwd = "$newpass"
#
set cert_opt = ""
# If you get a certificate verification error (version 1.10 or higher),
# uncomment the following line:
#set cert_opt = "--no-check-certificate"
#
if ("$passwd" == "xxxxxx") then
  echo "You need to set your password before you can continue - see the documentation in the script"
  exit
endif
#
# authenticate - NOTE: You should only execute this command ONE TIME.
# Executing this command for every data file you download may cause
# your download privileges to be suspended.
wget $cert_opt -O auth_status.rda.ucar.edu --save-cookies auth.rda.ucar.edu.$$ --post-data="email=gaoqg229@gmail.com&passwd=$passwd&action=login" https://rda.ucar.edu/cgi-bin/login
#
# download the file(s)
# NOTE:  if you get 403 Forbidden errors when downloading the data files, check
#        the contents of the file 'auth_status.rda.ucar.edu'
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1958/fcst_phy2m.061_tprat.reg_tl319.195801_195812
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1959/fcst_phy2m.061_tprat.reg_tl319.195901_195912
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1960/fcst_phy2m.061_tprat.reg_tl319.196001_196012
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1961/fcst_phy2m.061_tprat.reg_tl319.196101_196112
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1962/fcst_phy2m.061_tprat.reg_tl319.196201_196212
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1963/fcst_phy2m.061_tprat.reg_tl319.196301_196312
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1964/fcst_phy2m.061_tprat.reg_tl319.196401_196412
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1965/fcst_phy2m.061_tprat.reg_tl319.196501_196512
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1966/fcst_phy2m.061_tprat.reg_tl319.196601_196612
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1967/fcst_phy2m.061_tprat.reg_tl319.196701_196712
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1968/fcst_phy2m.061_tprat.reg_tl319.196801_196812
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1969/fcst_phy2m.061_tprat.reg_tl319.196901_196912
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1970/fcst_phy2m.061_tprat.reg_tl319.197001_197012
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1971/fcst_phy2m.061_tprat.reg_tl319.197101_197112
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1972/fcst_phy2m.061_tprat.reg_tl319.197201_197212
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1973/fcst_phy2m.061_tprat.reg_tl319.197301_197312
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1974/fcst_phy2m.061_tprat.reg_tl319.197401_197412
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1975/fcst_phy2m.061_tprat.reg_tl319.197501_197512
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1976/fcst_phy2m.061_tprat.reg_tl319.197601_197612
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1977/fcst_phy2m.061_tprat.reg_tl319.197701_197712
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1978/fcst_phy2m.061_tprat.reg_tl319.197801_197812
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1979/fcst_phy2m.061_tprat.reg_tl319.197901_197912
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1980/fcst_phy2m.061_tprat.reg_tl319.198001_198012
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1981/fcst_phy2m.061_tprat.reg_tl319.198101_198112
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1982/fcst_phy2m.061_tprat.reg_tl319.198201_198212
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1983/fcst_phy2m.061_tprat.reg_tl319.198301_198312
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1984/fcst_phy2m.061_tprat.reg_tl319.198401_198412
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1985/fcst_phy2m.061_tprat.reg_tl319.198501_198512
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1986/fcst_phy2m.061_tprat.reg_tl319.198601_198612
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1987/fcst_phy2m.061_tprat.reg_tl319.198701_198712
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1988/fcst_phy2m.061_tprat.reg_tl319.198801_198812
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1989/fcst_phy2m.061_tprat.reg_tl319.198901_198912
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1990/fcst_phy2m.061_tprat.reg_tl319.199001_199012
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1991/fcst_phy2m.061_tprat.reg_tl319.199101_199112
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1992/fcst_phy2m.061_tprat.reg_tl319.199201_199212
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1993/fcst_phy2m.061_tprat.reg_tl319.199301_199312
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1994/fcst_phy2m.061_tprat.reg_tl319.199401_199412
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1995/fcst_phy2m.061_tprat.reg_tl319.199501_199512
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1996/fcst_phy2m.061_tprat.reg_tl319.199601_199612
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1997/fcst_phy2m.061_tprat.reg_tl319.199701_199712
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1998/fcst_phy2m.061_tprat.reg_tl319.199801_199812
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/1999/fcst_phy2m.061_tprat.reg_tl319.199901_199912
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2000/fcst_phy2m.061_tprat.reg_tl319.200001_200012
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2001/fcst_phy2m.061_tprat.reg_tl319.200101_200112
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2002/fcst_phy2m.061_tprat.reg_tl319.200201_200212
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2003/fcst_phy2m.061_tprat.reg_tl319.200301_200312
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2004/fcst_phy2m.061_tprat.reg_tl319.200401_200412
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2005/fcst_phy2m.061_tprat.reg_tl319.200501_200512
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2006/fcst_phy2m.061_tprat.reg_tl319.200601_200612
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2007/fcst_phy2m.061_tprat.reg_tl319.200701_200712
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2008/fcst_phy2m.061_tprat.reg_tl319.200801_200812
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2009/fcst_phy2m.061_tprat.reg_tl319.200901_200912
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2010/fcst_phy2m.061_tprat.reg_tl319.201001_201012
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2011/fcst_phy2m.061_tprat.reg_tl319.201101_201112
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2012/fcst_phy2m.061_tprat.reg_tl319.201201_201212
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2013/fcst_phy2m.061_tprat.reg_tl319.201301_201312
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2014/fcst_phy2m.061_tprat.reg_tl319.201401_201412
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2015/fcst_phy2m.061_tprat.reg_tl319.201501_201512
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2016/fcst_phy2m.061_tprat.reg_tl319.201601_201612
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2017/fcst_phy2m.061_tprat.reg_tl319.201701_201712
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2018/fcst_phy2m.061_tprat.reg_tl319.201801_201812
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2019/fcst_phy2m.061_tprat.reg_tl319.201901_201912
wget $cert_opt $opts --load-cookies auth.rda.ucar.edu.$$ https://rda.ucar.edu/data/ds628.1/fcst_phy2m/2020/fcst_phy2m.061_tprat.reg_tl319.202001_202012
#
# clean up
rm auth.rda.ucar.edu.$$ auth_status.rda.ucar.edu
