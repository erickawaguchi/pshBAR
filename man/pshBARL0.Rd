% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pshBARL0.R
\name{pshBARL0}
\alias{pshBARL0}
\title{Efficient L0-BAR for the Fine-Gray Model via Coordinate-Wise Algorithm}
\usage{
pshBARL0(ftime, fstatus, X, failcode = 1, cencode = 0, lambda = 0,
  xi = 0, eps = 1e-06, max.iter = 1000)
}
\arguments{
\item{ftime}{A vector of event/censoring times.}

\item{fstatus}{A vector with unique code for each event type and a separate code for censored observations.}

\item{X}{A matrix of fixed covariates (nobs x ncovs)}

\item{failcode}{Integer: code of \code{fstatus} that event type of interest (default is 1)}

\item{cencode}{Integer: code of \code{fstatus} that denotes censored observations (default is 0)}

\item{lambda}{Numeric: BAR tuning parameter value}

\item{xi}{Numeric: tuning parameter for initial ridge regression}

\item{eps}{Numeric: algorithm stops when the relative change in any coefficient is less than \code{eps} (default is \code{1E-6})}

\item{max.iter}{Numeric: maximum iterations to achieve convergence (default is 1000)}
}
\value{
Returns a list of class \code{pshBAR}.
}
\description{
Fits broken adaptive ridge regression for competing risks regression using the coordinate wise BAR algorithm.
}
\details{
The \code{pshBAR} function penalizes the log-partial likelihood of the proportional subdistribution hazards model
from Fine and Gray (1999) with the Broken Adaptive Ridge (BAR) penalty. A cyclic coordinate descent algorithm is used for implementation.
For stability, the covariate matrix \code{X} is standardized prior to implementation.

Special cases: Fixing \code{xi} and \code{lambda} to 0 results in the standard competing risk regression using \code{crr}.
Fixing \code{lambda} to 0 and specifying \code{xi} will result in a ridge regression solution.
}
\examples{
set.seed(10)
ftime <- rexp(200)
fstatus <- sample(0:2, 200, replace = TRUE)
cov <- matrix(runif(1000), nrow = 200)
dimnames(cov)[[2]] <- c('x1','x2','x3','x4','x5')
fit <- pshBARL0(ftime, fstatus, cov, lambda = log(5), / 2, xi = log(5))
fit$coef
}
\references{
Breheny, P. and Huang, J. (2011) Coordinate descent algorithms for nonconvex penalized regression, with applications to biological feature selection. \emph{Ann. Appl. Statist.}, 5: 232-253.
Fine J. and Gray R. (1999) A proportional hazards model for the subdistribution of a competing risk.  \emph{JASA} 94:496-509.
}
