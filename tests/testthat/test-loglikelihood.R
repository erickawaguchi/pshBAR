test_that("getLogLikelihood returns same NULL likelihood as crr", {
  set.seed(10)
  ftime   <- rexp(50)
  fstatus <- sample(0:2, 50, replace = TRUE)
  cov     <- matrix(runif(250), nrow = 50)
  dimnames(cov)[[2]] <- c('x1', 'x2', 'x3', 'x4', 'x5')
  fit.crr <- crr(ftime, fstatus, cov)
  expect_equal(getLogLikelihood(ftime, fstatus, cov, beta = rep(0, 5)),
               fit.crr$loglik.null)
})

test_that("getLogLikelihood returns same likelihood as crr", {
  set.seed(10)
  ftime   <- rexp(50)
  fstatus <- sample(0:2, 50, replace = TRUE)
  cov     <- matrix(runif(250), nrow = 50)
  dimnames(cov)[[2]] <- c('x1', 'x2', 'x3', 'x4', 'x5')
  fit.crr <- crr(ftime, fstatus, cov)
  expect_equal(getLogLikelihood(ftime, fstatus, cov, beta = fit.crr$coef),
               fit.crr$loglik)
})
