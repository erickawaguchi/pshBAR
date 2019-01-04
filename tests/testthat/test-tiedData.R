library("testthat")
library("cmprsk")

test_that("pshBAR works with tied data", {
  set.seed(100)
  ftime   <- round(rexp(50), 0) + 10
  fstatus <- sample(0:2, 50, replace = TRUE)
  cov     <- matrix(runif(250), nrow = 50)
  dimnames(cov)[[2]] <- c('x1', 'x2', 'x3', 'x4', 'x5')

  fit.crr    <- crr(ftime, fstatus, cov)
  fit.BAR    <- pshBAR(ftime, fstatus, cov, lambda = 0, xi = 0)
  fit.BAR2   <- pshBARL0(ftime, fstatus, cov, lambda = 0, xi = 0)
  expect_equal(as.vector(fit.crr$coef), as.vector(fit.BAR$coef), tolerance = 1E-6)
  expect_equal(as.vector(fit.crr$coef), as.vector(fit.BAR2$coef), tolerance = 1E-6)
})

