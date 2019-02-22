# pshBAR

Broken Adaptive Ridge (BAR) regression for competing risks data.

Introduction
============

pshBAR is an `R` package for performing L_0-based regressions for the popular Fine-Gray model for competing risks data.

Dependencies
============
 * `survival`
 

Getting Started
===============
1. On Windows, make sure [RTools](https://CRAN.R-project.org/bin/windows/Rtools/) is installed.
2. In R, use the following commands to download and install pshBAR:

  ```r
  install.packages("devtools")
  library(devtools)
  install_github("erickawaguchi/pshBAR")
  ```

3. To perform L_0-penalized regression, use the following commands in R:
  ```r
  library(pshBAR)
  #Assume cause of interest of fstatus = 1.
  fit <- pshBAR(ftime, fstatus, X, failcode = 1, cencode = 0, lambda = log(ncovs), xi = 1)
  fit$coef #Extract coefficients
  ```
  
Examples
========
 ```r
set.seed(10)
ftime <- rexp(200)
fstatus <- sample(0:2, 200, replace = TRUE)
cov <- matrix(runif(1000), nrow = 200)
dimnames(cov)[[2]] <- c('x1','x2','x3','x4','x5')
fit <- pshBAR(ftime, fstatus, cov, lambda = log(dim(cos)[2]), xi = 1)
fit$coef
 ```
 
Development
===========
pshBAR is being developed in R Studio. If there are any questions or comments please email me at erickawaguchi[at]ucla.edu.

