library("testthat")
library("cmprsk")

test_that("pshBAR with no penalties gives same results as crr (small sample size)", {
  ftime   <- rexp(50)
  fstatus <- sample(0:2, 50, replace = TRUE)
  cov     <- matrix(runif(250), nrow = 50)
  dimnames(cov)[[2]] <- c('x1', 'x2', 'x3', 'x4', 'x5')

  fit.crr    <- crr(ftime, fstatus, cov)
  fit.BAR <- pshBAR(ftime, fstatus, cov, lambda = 0, xi = 0)
  expect_equal(as.vector(fit.crr$coef), as.vector(fit.BAR$coef), tolerance = 1E-4)
})

test_that("pshBARL0 with no penalties gives same results as crr (small sample size)", {
  ftime   <- rexp(200)
  fstatus <- sample(0:2, 200, replace = TRUE)
  cov     <- matrix(runif(1000), nrow = 200)
  dimnames(cov)[[2]] <- c('x1', 'x2', 'x3', 'x4', 'x5')

  fit.crr    <- crr(ftime, fstatus, cov)
  fit.BAR <- pshBARL0(ftime, fstatus, cov, lambda = 0, xi = 0)
  expect_equal(as.vector(fit.crr$coef), as.vector(fit.BAR$coef), tolerance = 1E-4)
})

