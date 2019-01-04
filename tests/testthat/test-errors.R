library("testthat")

test_that("pshBAR throws error for negative value of xi", {
  set.seed(10)
  ftime   <- rexp(50)
  fstatus <- sample(0:2, 50, replace = TRUE)
  cov     <- matrix(runif(250), nrow = 50)
  dimnames(cov)[[2]] <- c('x1', 'x2', 'x3', 'x4', 'x5')

  expect_that(pshBAR(ftime, fstatus, cov, lambda = 0, xi = -0.1), throws_error())
})

test_that("pshBAR throws error for negative value of lambda", {
  set.seed(10)
  ftime   <- rexp(50)
  fstatus <- sample(0:2, 50, replace = TRUE)
  cov     <- matrix(runif(250), nrow = 50)
  dimnames(cov)[[2]] <- c('x1', 'x2', 'x3', 'x4', 'x5')

  expect_that(pshBAR(ftime, fstatus, cov, lambda = -0.1, xi = 0), throws_error())
})


test_that("pshBARL0 throws error for negative value of xi", {
  set.seed(10)
  ftime   <- rexp(50)
  fstatus <- sample(0:2, 50, replace = TRUE)
  cov     <- matrix(runif(250), nrow = 50)
  dimnames(cov)[[2]] <- c('x1', 'x2', 'x3', 'x4', 'x5')

  expect_that(pshBARL0(ftime, fstatus, cov, lambda = 0, xi = -0.1), throws_error())
})

test_that("pshBARL0 throws error for negative value of lambda", {
  set.seed(10)
  ftime   <- rexp(50)
  fstatus <- sample(0:2, 50, replace = TRUE)
  cov     <- matrix(runif(250), nrow = 50)
  dimnames(cov)[[2]] <- c('x1', 'x2', 'x3', 'x4', 'x5')

  expect_that(pshBARL0(ftime, fstatus, cov, lambda = -0.1, xi = 0), throws_error())
})
