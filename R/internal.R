## These are internal functions that we do not want users to use

#Taken from crrp and grpreg packages:
#- Code below is by Patrick Breheny from grpreg package.
orthogonalize <- function(X, group) {
  n <- nrow(X)
  J <- max(group)
  V <- vector("list", J)
  XX <- matrix(0, nrow = nrow(X), ncol = ncol(X))
  for (j in seq_along(numeric(J))) {
    ind <- which(group == j)
    if (length(ind) == 0) next
    SVD <- svd(X[, ind, drop = FALSE], nu = 0)
    r <- which(SVD$d > 1e-10)
    V[[j]] <- sweep(SVD$v[, r, drop = FALSE], 2, sqrt(n) / SVD$d[r], "*")
    XX[,ind[r]] <- X[,ind] %*% V[[j]]
  }
  nz <- !apply(XX == 0, 2, all)
  XX <- XX[, nz, drop = FALSE]
  attr(XX, "T") <- V
  attr(XX, "group") <- group[nz]
  XX
}

unorthogonalize <- function(b, XX, group) {
  ind <- !sapply(attr(XX, "T"), is.null)
  T <- bdiag(attr(XX, "T")[ind])
  as.matrix(T %*% b)
}
