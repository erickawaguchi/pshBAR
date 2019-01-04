library("testthat")

test_that("pshBARL0 and pshBAR yield similar estimates for small sample sizes (up to 1E-5)", {
  set.seed(2019)
  ftime   <- rexp(50)
  fstatus <- sample(0:2, 50, replace = TRUE)
  cov     <- matrix(runif(250), nrow = 50)
  dimnames(cov)[[2]] <- c('x1', 'x2', 'x3', 'x4', 'x5')
  lambda  <- 0.1
  xi      <- 0.2
  a <- pshBAR(ftime, fstatus, cov, lambda = lambda, xi = xi)
  b <- pshBARL0(ftime, fstatus, cov, lambda = lambda, xi = xi)
  expect_equal(a$coef, b$coef, tolerance = 1E-5)
})

