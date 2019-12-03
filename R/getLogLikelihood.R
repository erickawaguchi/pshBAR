#' Get Log-Pseudo Likelihood
#'
#' @description Returns log-pseudo likelihood at a specific value of \code{beta}.
#'
#' @param ftime A vector of event/censoring times.
#' @param fstatus A vector with unique code for each event type and a separate code for censored observations.
#' @param X A matrix of fixed covariates (nobs x ncovs)
#' @param failcode Integer: code of \code{fstatus} that event type of interest (default is 1)
#' @param cencode Integer: code of \code{fstatus} that denotes censored observations (default is 0)
#' @param beta A vector of coefficient values at which to evaluate log-pseudo likelihood
#' @return Returns log-pseudo likelihood at \code{beta}.
#' @examples
#' library(cmprsk)
#' set.seed(10)
#' ftime <- rexp(200)
#' fstatus <- sample(0:2, 200, replace = TRUE)
#' cov <- matrix(runif(1000), nrow = 200)
#' dimnames(cov)[[2]] <- c('x1','x2','x3','x4','x5')
#' # fit <- crr(ftime, fstatus, cov) # crr does not work in R 3.6.0 as of 12/03/2019
#' fit$loglik
#' getLogLikelihood(ftime, fstatus, cov, beta = fit$coef)
#' @references
#'
#' Fine J. and Gray R. (1999) A proportional hazards model for the subdistribution of a competing risk.  \emph{JASA} 94:496-509.
#' @import survival
#' @export

getLogLikelihood <- function(ftime, fstatus, X, failcode = 1, cencode = 0,
                             beta = rep(0, dim(X)[2])){

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

  out <- .Call("evalLogLikelihood", as.double(X), as.double(ftime), as.integer(fstatus),
               as.double(uuu), as.double(beta), PACKAGE = "pshBAR")
 return(out)
}
