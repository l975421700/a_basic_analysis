
# module load R/4.1.0_conda
# r

suppressPackageStartupMessages(library(circular))
suppressPackageStartupMessages(library(reticulate))



repl_python()

import warnings
warnings.filterwarnings('ignore')
import pickle
import xarray as xr
import numpy as np

exp_odir = 'output/echam-6.3.05p2-wiso/pi/'
expid = [
    'pi_m_416_4.9',
    ]
i = 0

with open(exp_odir + expid[i] + '/analysis/echam/' + expid[i] + '.pre_weighted_lon.pkl', 'rb') as f:
    pre_weighted_lon = pickle.load(f)

with open('scratch/test.npy', 'rb') as f:
    wwtest_all = np.load(f)

exit

# sum(py$wwtest_all) / length(py$wwtest_all)


#-------- Watson’s Large-sample Nonparametric Test
YgVal <- function(cdat, ndat, g) {
    N <- length(cdat)
    ndatcsum <- cumsum(ndat)
    delhat <- 0
    tbar <- 0
    for (k in 1:g) {
        sample <- circular(0)
        if (k==1) {low <- 0} else
        if (k > 1) {low <- ndatcsum[k-1]}
        for (j in 1:ndat[k]) { sample[j] <- cdat[j+low] }
        tm1 <- trigonometric.moment(sample, p=1)
        tm2 <- trigonometric.moment(sample, p=2)
        Rbar1 <- tm1$rho
        Rbar2 <- tm2$rho
        tbar[k] <- tm1$mu
        delhat[k] <- (1-Rbar2)/(2*Rbar1*Rbar1)
    }
    dhatmax <- max(delhat)
    dhatmin <- min(delhat)
    if (dhatmax/dhatmin <= 4) {
        CP <- 0
        SP <- 0
        dhat0 <- 0
        for (k in 1:g) {
            CP <- CP+ndat[k]*cos(tbar[k])
            SP <- SP+ndat[k]*sin(tbar[k])
            dhat0 <- dhat0+ndat[k]*delhat[k] }
        dhat0 <- dhat0/N
        RP <- sqrt(CP*CP+SP*SP)
        Yg <- 2*(N-RP)/dhat0
        return(Yg)
        }
    else if (dhatmax/dhatmin > 4) {
        CM <- 0
        SM <- 0
        Yg <- 0
        for (k in 1:g) {
            CM <- CM+(ndat[k]*cos(tbar[k])/delhat[k])
            SM <- SM+(ndat[k]*sin(tbar[k])/delhat[k])
            Yg <- Yg+(ndat[k]/delhat[k]) }
            RM <- sqrt(CM*CM+SM*SM)
            Yg <- 2*(Yg-RM)
            return(Yg)
        }
}

wwtest_all1 = py$wwtest_all
wwtest_all2 = py$wwtest_all

for (ilat in 1:dim(wwtest_all1)[1]){
    for (ilon in 1:dim(wwtest_all1)[2]){
        # ilat <- 48 + 1
        # ilon <- 96 + 1
        sampler1 <- circular::circular(
            na.omit(py$pre_weighted_lon$sea$values[seq(4, 199, 4), ilat, ilon]) * pi / 180,
            type = "angles",
            units = "radians",
            # template = "geographics",
            modulo = "2pi",
            zero = 0,
            rotation = 'counter',
        )
        sampler2 <- circular::circular(
            na.omit(py$pre_weighted_lon$sea$values[seq(2, 199, 4), ilat, ilon]) * pi / 180,
            # NaN,
            type = "angles",
            units = "radians",
            # template = "geographics",
            modulo = "2pi",
            zero = 0,
            rotation = 'counter',
        )
        if( (length(sampler1) < 10) | (length(sampler2) < 10) ){
            wwtest_all1[ilat, ilon] <- FALSE
            wwtest_all2[ilat, ilon] <- FALSE
        } else{
            cdat <- c(sampler1, sampler2)
            ndat <- c(length(sampler1), length(sampler2))
            g <- 2
            YgObs <- YgVal(cdat, ndat, g)
            wwtest_all1[ilat, ilon] <- pchisq(YgObs, g-1, lower.tail=F) < 0.05
            wwtest_all2[ilat, ilon] <- circular::watson.williams.test(list(sampler1, sampler2))$p.value < 0.05
        }
    }
    print(ilat)
}



ilat <- 45 + 1
ilon <- 90 + 1

sampler1 <- circular::circular(
    py$pre_weighted_lon$sea$values[seq(4, 199, 4), ilat, ilon] * pi / 180,
    type = "angles",
    units = "radians",
    # template = "geographics",
    modulo = "2pi",
    zero = 0,
    rotation = 'counter',
    )
sampler2 <- circular::circular(
    py$pre_weighted_lon$sea$values[seq(2, 199, 4), ilat, ilon] * pi / 180,
    type = "angles",
    units = "radians",
    # template = "geographics",
    modulo = "2pi",
    zero = 0,
    rotation = 'counter',
    )

cdat <- c(sampler1, sampler2)
ndat <- c(length(sampler1), length(sampler2))
g <- 2
YgObs <- YgVal(cdat, ndat, g)
pchisq(YgObs, g-1, lower.tail=F) < 0.05
wwtest_all1[ilat, ilon]

circular::watson.williams.test(list(sampler1, sampler2))$p.value < 0.05
wwtest_all2[ilat, ilon]

py$wwtest_all[ilat, ilon]

# Test in python is stricter
sum(py$wwtest_all) / length(py$wwtest_all) # 0.7431098
sum(wwtest_all1) / length(wwtest_all1) # 0.7908529
sum(wwtest_all2) / length(wwtest_all2) # 0.7843967

sum(py$wwtest_all == wwtest_all1) / length(py$wwtest_all) # 0.9506293
sum(py$wwtest_all == wwtest_all2) / length(py$wwtest_all) # 0.9587131




'''
cdat1 <- circular(fisherB10$set1*2*pi/360)
cdat2 <- circular(fisherB10$set2*2*pi/360)
cdat3 <- circular(fisherB10$set3*2*pi/360)

cdat <- c(cdat1, cdat2, cdat3)
ndat <- c(length(cdat1), length(cdat2), length(cdat3))
g <- 3
YgObs <- YgVal(cdat, ndat, g)
pchisq(YgObs, g-1, lower.tail=F)


cdat1 <- rvonmises(n=500, mu=circular(0), kappa=4)
cdat2 <- rvonmises(n=500, mu=circular(1), kappa=6)
cdat <- c(cdat1, cdat2)
ndat <- c(length(cdat1), length(cdat2))
g <- 2
YgObs <- YgVal(cdat, ndat, g)
pchisq(YgObs, g-1, lower.tail=F)
'''























sample = circular::circular(
    py$pre_weighted_lon$ann$values[, ilat, ilon],
    type = "angles",
    units = "degrees",
    template = "geographics",
    modulo = "2pi",
    )

sampler = circular::circular(
    py$pre_weighted_lon$ann$values[, ilat, ilon] * pi / 180,
    type = "angles",
    units = "radians",
    # template = "geographics",
    modulo = "2pi",
    zero = 0,
    rotation = 'counter',
    )

circular::mean.circular(sample) # 195.2632
circular::var.circular(sample) # 5.046246e-05
circular::rayleigh.test(sample) # uniformity

circular::mean.circular(sampler) * 180 / pi
circular::var.circular(sampler)
circular::rayleigh.test(sampler)


x <- rvonmises(n=50, mu=circular(0), kappa=4)
watson.test(x, dist="vonmises")
# watson.test(sampler, dist="vonmises")


#-------- watson's test

cdat <- circular(fisherB6$set1*2*pi/360)
vMmle <- mle.vonmises(cdat, bias=TRUE)
muhat <- vMmle$mu ; semu <- vMmle$se.mu ; muhat ; semu
kaphat <- vMmle$kappa ; sekap <- vMmle$se.kappa ; kaphat ; sekap

vMGoF <- function(circdat, mu, kappa) {
    tdf <- pvonmises(circdat, circular(mu), kappa, from=circular(0), tol=1e-06)
    cunif <- circular(2*pi*tdf)
    kuires <- kuiper.test(cunif)
    rayres <- rayleigh.test(cunif)
    raores <- rao.spacing.test(cunif)
    watres <- watson.test(cunif)
    return(list(kuires, rayres, raores, watres))
}

vMGoFBoot <- function(origdat, B) {
    n <- length(origdat)
    vMmle <- mle.vonmises(origdat, bias=TRUE)
    muhat0 <- vMmle$mu
    kaphat0 <- vMmle$kappa
    tdf <- pvonmises(origdat, muhat0, kaphat0, from=circular(0), tol = 1e-06)
    cunif <- circular(2*pi*tdf)
    unitest0 <- 0
    nxtrm <- 0
    pval <- 0
    for (k in 1:4) {unitest0[k]=0
    nxtrm[k]=1}
    unitest0[1] <- kuiper.test(cunif)$statistic
    unitest0[2] <- rayleigh.test(cunif)$statistic
    unitest0[3] <- rao.spacing.test(cunif)$statistic
    unitest0[4] <- watson.test(cunif)$statistic
    for (b in 2:(B+1)) {
    bootsamp <- rvonmises(n, muhat0, kaphat0)
    vMmle <- mle.vonmises(bootsamp, bias=TRUE)
    muhat1 <- vMmle$mu
    kaphat1 <- vMmle$kappa
    tdf <- pvonmises(bootsamp, muhat1, kaphat1, from=circular(0), tol = 1e-06)
    cunif <- circular(2*pi*tdf)
    kuiper1 <- kuiper.test(cunif)$statistic
    if (kuiper1 >= unitest0[1]) {nxtrm[1] <- nxtrm[1] + 1}
    rayleigh1 <- rayleigh.test(cunif)$statistic
    if (rayleigh1 >= unitest0[2]) {nxtrm[2] <- nxtrm[2] + 1}
    rao1 <- rao.spacing.test(cunif)$statistic
    if (rao1 >= unitest0[3]) {nxtrm[3] <- nxtrm[3] + 1}
    watson1 <- watson.test(cunif)$statistic
    if (watson1 >= unitest0[4]) {nxtrm[4] <- nxtrm[4] + 1}
    }
    for (k in 1:4) {pval[k] <- nxtrm[k]/(B+1)}
    return(pval)
}

vMGoF(cdat, muhat, kaphat)
B <- 10 ; pval <- vMGoFBoot(cdat, B) ; pval
watson.test(cdat, dist="vonmises")

B <- 10 ; pval <- vMGoFBoot(x, B) ; pval
watson.test(x, dist="vonmises")

# B <- 1 ; pval <- vMGoFBoot(sampler, B) ; pval

vMmle_s <- mle.vonmises(sampler, bias=TRUE)
muhat_s <- vMmle_s$mu
kaphat_s <- vMmle_s$kappa

# vMGoF(sampler, muhat_s, kaphat_s)

circdat = sampler
mu = muhat_s
kappa = kaphat_s
# watson.test(cdat, dist="vonmises")














