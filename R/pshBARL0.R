#' Efficient L0-BAR for the Fine-Gray Model via Coordinate-Wise Algorithm
#'
#' @description Fits broken adaptive ridge regression for competing risks regression using the coordinate wise BAR algorithm.
#'
#' @param ftime A vector of event/censoring times.
#' @param fstatus A vector with unique code for each event type and a separate code for censored observations.
#' @param X A matrix of fixed covariates (nobs x ncovs)
#' @param failcode Integer: code of \code{fstatus} that event type of interest (default is 1)
#' @param cencode Integer: code of \code{fstatus} that denotes censored observations (default is 0)
#' @param lambda Numeric: BAR tuning parameter value
#' @param xi Numeric: tuning parameter for initial ridge regression
#' @param eps Numeric: algorithm stops when the relative change in any coefficient is less than \code{eps} (default is \code{1E-6})
#' @param max.iter Numeric: maximum iterations to achieve convergence (default is 1000)
#'
#' @details The \code{pshBAR} function penalizes the log-partial likelihood of the proportional subdistribution hazards model
#' from Fine and Gray (1999) with the Broken Adaptive Ridge (BAR) penalty. A cyclic coordinate descent algorithm is used for implementation.
#' For stability, the covariate matrix \code{X} is standardized prior to implementation.
#'
#' Special cases: Fixing \code{xi} and \code{lambda} to 0 results in the standard competing risk regression using \code{crr}.
#' Fixing \code{lambda} to 0 and specifying \code{xi} will result in a ridge regression solution.
#' @return Returns a list of class \code{pshBAR}.
#'
#' @import survival cmprsk
#' @export
#' @useDynLib pshBAR standardize
#' @examples
#' set.seed(10)
#' ftime <- rexp(200)
#' fstatus <- sample(0:2, 200, replace = TRUE)
#' cov <- matrix(runif(1000), nrow = 200)
#' dimnames(cov)[[2]] <- c('x1','x2','x3','x4','x5')
#' fit <- pshBARL0(ftime, fstatus, cov, lambda = log(5), / 2, xi = log(5))
#' fit$coef
#' @references
#' Breheny, P. and Huang, J. (2011) Coordinate descent algorithms for nonconvex penalized regression, with applications to biological feature selection. \emph{Ann. Appl. Statist.}, 5: 232-253.
#' Fine J. and Gray R. (1999) A proportional hazards model for the subdistribution of a competing risk.  \emph{JASA} 94:496-509.

pshBARL0 <- function(ftime, fstatus, X, failcode = 1, cencode = 0,
                   lambda = 0, xi = 0,
                   eps = 1E-6,
                   max.iter = 1000){

  ## Error checking
  if(xi < 0) stop("xi must be a non-negative number.")
  if(max.iter < 1) stop("max.iter must be positive integer.")
  if(eps <= 0) stop("eps must be a positive number.")
  # Sort time
  n <- length(ftime)
  p <- ncol(X)
  d <- data.frame(ftime = ftime, fstatus = fstatus)
  if (!missing(X)) d$X <- as.matrix(X)
  d        <- d[order(d$ftime, -d$fstatus, decreasing = TRUE), ]
  ftime    <- d$ftime
  cenind   <- ifelse(d$fstatus == cencode, 1, 0)
  fstatus  <- ifelse(d$fstatus == failcode, 1, 2 * (1 - cenind))
  X <- d$X
  u <- do.call('survfit', list(formula = Surv(ftime, cenind) ~ 1,
                               data = data.frame(ftime, cenind)))

  # uuu is weight function (IPCW)
  u <- approx(c(0, u$time, max(u$time) * (1 + 10 * .Machine$double.eps)), c(1, u$surv, 0),
              xout = ftime * (1 - 100 * .Machine$double.eps), method = 'constant',
              f = 0, rule = 2)
  uuu <- u$y

  # Standardize design matrix here
  std    <- .Call("standardize", X, PACKAGE = "pshBAR")
  XX     <- std[[1]]
  center <- std[[2]]
  scale  <- std[[3]]
  nz <- which(scale > 1e-6)
  if (length(nz) != ncol(XX)) XX <- XX[ , nz, drop = FALSE]


  # If lambda is MISSING, create lambda path
  if(missing(lambda)) {
    stop("Lambda must be a non-negtive number or vector of non-negative numbers.")
  } else if(min(lambda) < 0) {
    stop("lambda must be a non-negative number.")
  } else {
    if(is.unsorted(lambda)) sort(lambda)
  }

   nlam <- length(lambda)

  ## Fit the PSH Ridge Model here w/ tuning parameter xi
  ridgeFit   <- .Call("ccd_ridge", XX, as.numeric(ftime), as.integer(fstatus), uuu,
                      xi, eps, as.integer(max.iter),
                      penalty.factor = rep(1, p), PACKAGE = "pshBAR")
  ridgeCoef  <- ridgeFit[[1]] / scale #Divide coeff estimates by sdev
  ridgeIter  <- ridgeFit[[3]]

  #Enter BAR Fit here (for different lambdas) keep everything in terms of standardized coefficients
  btmp <- ridgeFit[[1]]
  barFit <- .Call("ccd_bar", XX, as.numeric(ftime), as.integer(fstatus), uuu,
                      as.vector(lambda), as.double(eps), as.integer(max.iter),
                      as.vector(btmp), PACKAGE = "pshBAR")

  #Store results here
  coefMatrix  <- as.matrix(matrix(barFit[[1]], p)) / scale
  logLik      <- as.double(barFit[[2]][-1] / -2)
  logLik.null <- as.double(barFit[[2]][1] / -2)
  iter        <- as.integer(barFit[[3]])
  conv        <- as.integer(barFit[[8]])

  ## Output
  colnames(coefMatrix) <- round(lambda, 3)
  val <- structure(list(coef = coefMatrix,
                        logLik = logLik,
                        logLik.null = logLik.null,
                        iter = iter,
                        lambda = lambda,
                        converged = conv,
                        ridgeCoef = ridgeCoef,
                        xi = xi,
                        call = sys.call()),
                   class = "pshBAR")

  val
}
