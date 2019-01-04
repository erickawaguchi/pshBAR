#include <math.h>
#include <Rmath.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R.h>
#include <R_ext/Applic.h>
#include <stdlib.h>
#define LEN sizeof(double)

//Define sgn function: sgn(z) = 1 if z > 0, -1 if z < 0, 0 if z = 0
double sgn(double z) {
  double s = 0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  return(s);
}

//Standardize design matrix
SEXP standardize(SEXP X_) {
  // Declarations
  int n = nrows(X_);
  int p = ncols(X_);
  SEXP XX_, c_, s_;
  PROTECT(XX_ = allocMatrix(REALSXP, n, p));
  PROTECT(c_ = allocVector(REALSXP, p));
  PROTECT(s_ = allocVector(REALSXP, p));
  double *X = REAL(X_);
  double *XX = REAL(XX_);
  double *c = REAL(c_);
  double *s = REAL(s_);

  for (int j = 0; j < p; j++) {

    // Center (Calculate mean and subtract)
    c[j] = 0;
    for (int i = 0; i < n; i++) {
      c[j] += X[j * n + i];
    }
    c[j] = c[j] / n;
    for (int i = 0; i < n; i++) XX[j * n + i] = X[j * n + i] - c[j];

    // Scale (Calculate sdev and divide)
    s[j] = 0;
    for (int i = 0; i < n; i++) {
      s[j] += pow(XX[j * n + i], 2);
    }
    s[j] = sqrt(s[j] / n);
    for (int i = 0; i < n; i++) XX[j * n + i] = XX[j * n + i] / s[j];
  }

  // Return list
  SEXP res;
  PROTECT(res = allocVector(VECSXP, 3));
  SET_VECTOR_ELT(res, 0, XX_); // Standardized design matrix
  SET_VECTOR_ELT(res, 1, c_); // Mean
  SET_VECTOR_ELT(res, 2, s_); // Standard deviations
  UNPROTECT(4);
  return(res);
}

////////////////////////////////////////////////////////////////////////////////////////
// Calculate Log-Partial Likelihood
double getLogLikelihood(double *t2, int *ici, double *x, int ncov, int nin, double *wt, double *b)
{
  // Look at Eq (2) from Fu et al. 2017.
  int i, j, i2;
  const int p = ncov,  n = nin;
  double accNum1[n]; //accumulate the backwards numerator
  double accNum2[n]; //acumulate the foreward numerator (weighted)
  double accSum[n]; //accumulate sum over both accNum1 and accNum2
  double tmp1 = 0; //track backward sum for uncensored events risk set
  double tmp2 = 0; //track forward sum for competing risks risk set
  double loglik = 0; //store loglik

  double eta[n]; //calculate eta in this step
  for(i = 0; i < n; i++) {
    //initialize values to 0
    eta[i] = 0;
    accNum1[i] = 0;
    accNum2[i] = 0;

    for (j = 0; j < p; j++)
      eta[i] += b[j] * x[n * j + i];
  }

  //Note: t2, ici, and x should be ordered in DECREASING order. (First time is largest)
  //Backward Scan
  for (i = 0; i < n; i++)
  {
    tmp1 += exp(eta[i]); //track cumulative sum over 1:n
    if (ici[i] != 1) {
      // if subject is not an event then accNum[i] = 0;
      accNum1[i] = 0;
    } else {
      loglik += eta[i];
      accNum1[i] = tmp1;
    }

    //Forward Scan (To take into account the competing risks component)
    i2 = (n - 1) - i;
    if (ici[i2] == 2) {
      tmp2 += exp(eta[i2]) / wt[i2];
    }
    if (ici[i2] != 1) {
      accNum2[i2] = 0;
    } else {
      accNum2[i2] = wt[i2] * tmp2; //Sort accNum2 in same order as backward scan
    }
  }

  //taking into account ties
  for(i = 0; i < n; i++) {
    i2 = (n - 1) - i;
    if(ici[i2] == 2 || ici[i2 - 1] != 1 || i2 == 0) continue;
    if(t2[i2] == t2[i2 - 1]) {
      accNum1[i2 - 1] = accNum1[i2];
    }
  }


  //calculate loglik
  for(i = 0; i < n; i++) {
    accSum[i] = 0;
    if (ici[i] != 1) continue;
    accSum[i] = accNum1[i] + accNum2[i];
    loglik  -= log(accSum[i]);
  }
  return loglik;
}



SEXP evalLogLikelihood(SEXP x_, SEXP t2_, SEXP ici_, SEXP wt_, SEXP beta_) {
  //Declaration
  int n = length(t2_);
  int p = length(x_) / n;
  //int L = length(lambda);

  // initialize
  double *x = REAL(x_);
  double *t2 = REAL(t2_);
  double *wt = REAL(wt_);
  double *b = REAL(beta_);
  int *ici = INTEGER(ici_);
  double loglik = getLogLikelihood(t2, ici, x, p, n, wt, b);
  return(ScalarReal(loglik));
}


// Criterion for convergence: All coefficients must pass the following |(b_new - b_old) / b_old| < eps
int checkConvergence(double *beta, double *beta_old, double eps, int p) {
  int converged = 1;
  for (int j = 0; j < p; j++) {
    if (fabs((beta[j] - beta_old[j]) / beta_old[j]) > eps) {
      converged = 0;
      break;
    }
  }
  return(converged);
}

// Create penalty function for ridge regression. Lasso is here as outline
// Edit: Added into BAR function directly.
// See Simon et al. 2011 (pg. 4)
double ridge(double z, double l1, double v) {
  return(z / (v + l1));
}


// Weighted cross product of y with jth column of x
double wcrossprod(double *X, double *y, double *w, int n, int j) {
  int nn = n * j;
  double val = 0;
  for (int i = 0; i < n; i++) val += X[nn + i] * y[i] * w[i];
  return(val);
}

// Weighted sum of squares of jth column of X
double wsqsum(double *X, double *w, int n, int j) {
  int nn = n * j;
  double val = 0;
  for (int i = 0; i < n; i++) val += w[i] * pow(X[nn + i], 2);
  return(val);
}


///////////////////////////////////////////////////////////////////////////////////////
SEXP cleanupCRR(double *a, double *eta, double *st, double *w, double *diffBeta, double *accNum1, double *accNum2, double *accSum,
                SEXP beta, SEXP Dev, SEXP iter, SEXP residuals, SEXP score, SEXP hessian, SEXP linpred) {
  Free(a);
  Free(eta);
  Free(st);
  Free(w);
  Free(diffBeta);
  Free(accNum1);
  Free(accNum2);
  Free(accSum);
  SEXP res;
  PROTECT(res = allocVector(VECSXP, 7));
  SET_VECTOR_ELT(res, 0, beta); //coefficient estimates
  SET_VECTOR_ELT(res, 1, Dev); //deviance = -2*loglik
  SET_VECTOR_ELT(res, 2, iter); //iterations until convergence
  SET_VECTOR_ELT(res, 3, residuals); //residuals
  SET_VECTOR_ELT(res, 4, score); //gradient
  SET_VECTOR_ELT(res, 5, hessian); //hessian
  SET_VECTOR_ELT(res, 6, linpred); //hessian
  UNPROTECT(8);
  return(res);
}


//////////////////////////////////////////////////////////////////////////////////
//start cordinate descent

//x_ = design matrix
//t2_ = failtime
//ici_ = censoring vector
//wt_ = weight (uuu)
//lambda = tuning parameter
//esp_ = epsilon (thershold)
//max_iter_ = max iterations
//multiplier = penalty.factor (what to reweight lambda by)

SEXP ccd_ridge(SEXP x_, SEXP t2_, SEXP ici_, SEXP wt_, SEXP lambda_,
               SEXP esp_, SEXP max_iter_, SEXP multiplier) {

  //Declaration
  int n = length(t2_);
  int p = length(x_) / n;

  //Output
  SEXP res, beta, Dev, iter, residuals, score, hessian, linpred;
  PROTECT(beta = allocVector(REALSXP, p));
  double *b = REAL(beta);
  for (int j = 0; j < p; j++) b[j] = 0;
  PROTECT (score = allocVector(REALSXP, n));
  double *s = REAL(score);
  for (int i = 0; i < n; i++) s[i] = 0;
  PROTECT (hessian = allocVector(REALSXP, n));
  double *h = REAL(hessian);
  for (int i = 0; i < n; i++) h[i] = 0;
  PROTECT(residuals = allocVector(REALSXP, n));
  double *r = REAL(residuals);
  PROTECT(Dev = allocVector(REALSXP, 1));
  for (int i = 0; i < 1; i++) REAL(Dev)[i] = 0;
  PROTECT(iter = allocVector(INTSXP, 1));
  for (int i = 0; i < 1; i++) INTEGER(iter)[i] = 0;
  PROTECT(linpred = allocVector(REALSXP, n));
  double *lp = REAL(linpred);
  for (int i = 0; i <  n; i++) lp[i] = 0;

  //Intermediate quantities for internal use (must be freed afterwards!)
  double *a = Calloc(p, double); // Beta from previous iteration
  for (int j = 0; j < p; j++) a[j] = 0;
  double *st = Calloc(n, double);
  for (int i = 0; i < n; i++) st[i] = 0;
  double *w = Calloc(n, double);
  for ( int i = 0; i < n; i++) w[i] = 0;
  double *eta = Calloc(n, double);
  for (int i = 0; i < n; i++) eta[i] = 0;
  double *diffBeta = Calloc(p, double);
  for (int j = 0; j < p; j++) diffBeta[j] = 1;
  double *accNum1 = Calloc(n, double); //accumulate the backwards numerator
  for (int i = 0; i < n; i++) accNum1[i] = 0;
  double *accNum2 = Calloc(n, double); //acumulate the foreward numerator (weighted)
  for (int i = 0; i < n; i++) accNum2[i] = 0;
  double *accSum = Calloc(n, double); //accumulate sum over both accNum1 and accNum2
  for (int i = 0; i < n; i++) accSum[i] = 0;


  //Pointers for R values
  double *x = REAL(x_);
  double *t2 = REAL(t2_);
  double *wt = REAL(wt_);
  int *ici = INTEGER(ici_);
  double lam = REAL(lambda_)[0];
  double esp = REAL(esp_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(multiplier);

  //internal storage
  double nullDev; //to store null deviance
  double grad, hess, l1, shift,  si, delta;
  int converged; //convergence check
  int i, j, i2; //for loop indices
  double tmp1 = 0; //track backward sum for uncensored events risk set
  double tmp2 = 0; //track forward sum for competing risks risk set
  //end of declaration;

  //Start regression
  //calculate null deviance
  nullDev = -2 * getLogLikelihood(t2, ici, x, p, n, wt, a);

  //start
  while (INTEGER(iter)[0] < max_iter) {
    if (REAL(Dev)[0] - nullDev > 0.99 * nullDev) break;

    INTEGER(iter)[0]++;

    //Reset values
    tmp1 = 0, tmp2 = 0;
    for(i = 0; i < n; i++) {
      st[i] = 0;
      w[i] = 0;
    }

    //Backward Scan
    for (i = 0; i < n; i++)
    {
      tmp1 += exp(eta[i]); //track cumulative sum over 1:n
      if (ici[i] != 1) {
        accNum1[i] = 0;
      } else {
        accNum1[i] = tmp1;
      }

      //Forward Scan (To take into account the competing risks component)
      i2 = (n - 1) - i;
      if (ici[i2] == 2) {
        tmp2 += exp(eta[i2]) / wt[i2];
      }
      if (ici[i2] != 1) {
        accNum2[i2] = 0;
      } else {
        accNum2[i2] = wt[i2] * tmp2; //Sort accNum2 in same order as backward scan
      }
    }


    //calculate risk set for denominator (adjusted for ties)
    for(i = 0; i < n; i++) {
      if (ici[i] != 1) continue;
      accSum[i] = accNum1[i] + accNum2[i];
    }

    //taking into account ties
    for(i = 0; i < n; i++) {
      i2 = (n - 1) - i;
      if(ici[i2] == 2 || ici[i2 - 1] != 1 || i2 == 0) continue;
      if(t2[i2] == t2[i2 - 1]) {
        accSum[i2 - 1] = accSum[i2];
      }
    }


    //calculate score and hessian here
    double tmp1 = 0; tmp2 = 0; //reset temporary vals

    //linear scan for non-competing risks (backwards scan)
    for(i = n; i > 0; i--) {
      if(ici[i - 1] == 1) {
        tmp1 += 1 / accSum[i - 1];
        tmp2 += 1 / pow(accSum[i - 1], 2);
        accNum1[i - 1] = tmp1;
        accNum2[i - 1] = tmp2;
      } else {
        accNum1[i - 1] = tmp1;
        accNum2[i - 1] = tmp2;
      }
    }

    //Fix ties here:
    for(i = 0; i < n; i++) {
      //only needs to be adjusted consective event times
      if(ici[i] != 1 || ici[i + 1] != 1 || i == (n - 1)) continue;
      if(t2[i] == t2[i + 1]) {
        accNum1[i + 1] = accNum1[i];
        accNum2[i + 1] = accNum2[i];
      }
    }


    //Store into st and w so we can reuse accNum1 and accNum2
    for(i = 0; i < n; i++) {
      st[i] = accNum1[i] * exp(eta[i]);
      w[i] = accNum2[i] * pow(exp(eta[i]), 2);
    }

    //Perform linear scan for competing risks
    tmp1 = 0; tmp2 = 0; //reset tmp vals
    for(i = 0; i < n; i++) {
      accNum1[i] = 0;
      accNum2[i] = 0;
      if(ici[i] == 1) {
        tmp1 += wt[i] / accSum[i];
        tmp2 += pow(wt[i] / accSum[i], 2);
      }
      if(ici[i] != 2) continue;
      accNum1[i] = tmp1;
      accNum2[i] = tmp2;
    }

    //Now combine to produce score and hessian
    for(i = 0; i < n; i++) {
      //First, update st and w and then get score and hessian
      st[i] += accNum1[i] * (exp(eta[i]) / wt[i]);
      w[i] += accNum2[i] * pow(exp(eta[i]) / wt[i], 2);
    }

    for(i= 0; i < n; i++) {
      w[i] = (st[i] - w[i]);
      if(ici[i] != 1) {
        st[i] = - st[i];
      } else {
        st[i] = (1 - st[i]);
      }
    }

    //end calculation of score and hessian

    for (i = 0; i < n; i++){
      if (w[i] == 0) r[i] = 0;
      else r[i] = st[i] / w[i];
    }

    // calculate xwr and xwx & update beta_j
    for (j = 0; j < p; j++) {
      grad = -wcrossprod(x, r, w, n, j); // jth component of gradient [l'(b)]
      hess = wsqsum(x, w, n, j); // jth component of hessian [l''(b)]
      l1 = lam * m[j]; //divide by n since we are minimizing the following: -(1/n)l(beta) + lambda * p(beta)
      delta = -(grad + a[j] * l1) / (hess + l1);

      // Do one dimensional ridge update.
      // Employ trust region as in Genkin et al. (2007) for quadratic approximation.
      b[j] = a[j] + sgn(delta) * fmin(fabs(delta), diffBeta[j]);
      diffBeta[j] = fmax(2 * fabs(delta), diffBeta[j] / 2);

      // Update r
      shift = b[j] - a[j];
      if (shift != 0) {
        for (int i = 0; i < n; i++) {
          si = shift * x[j * n + i];
          r[i] -= si;
          eta[i] += si;
        }
      } //end shift
    } //for j = 0 to (p - 1)

    // Check for convergence
    converged = checkConvergence(b, a, esp, p);
    for (j = 0; j < p; j++)
      a[j] = b[j];

    //Calculate deviance
    REAL(Dev)[0] = -2 * getLogLikelihood(t2, ici, x, p, n, wt, a);

    for (i = 0; i < n; i++){
      lp[i] = eta[i];
      s[i] = st[i];
      h[i] = w[i];
    }
    if (converged)  break;
  } //for while loop
  res = cleanupCRR(a, eta, st, w, diffBeta, accNum1, accNum2, accSum, beta, Dev, iter, residuals, score, hessian, linpred);
  return(res);
}

////////////////////////////////////////////////////
// FAST L_0-BAR
////////////////////////////////////////////////////
int checkFastBarConvergence(double *beta, double *beta_old, double eps, int l, int p) {
  int converged = 1;
  for (int j = 0; j < p; j++) {
    if (fabs((beta[l * p + j] - beta_old[j])) > eps) {
      converged = 0;
      break;
    }
  }
  return(converged);
}

SEXP cleanupNewCRR(double *a, double *e,  double *eta, double *st, double *w, double *accNum1, double *accNum2, double *accSum,
                   SEXP beta, SEXP Dev, SEXP iter, SEXP residuals, SEXP score, SEXP hessian, SEXP linpred, SEXP converged) {
  // Free up all intermediate step variables
  Free(a);
  Free(e);
  Free(eta);
  Free(st);
  Free(w);
  Free(accNum1);
  Free(accNum2);
  Free(accSum);
  SEXP res;
  PROTECT(res = allocVector(VECSXP, 8));
  SET_VECTOR_ELT(res, 0, beta); //coefficient estimates
  SET_VECTOR_ELT(res, 1, Dev); //deviance = -2*loglik
  SET_VECTOR_ELT(res, 2, iter); //iterations until convergence
  SET_VECTOR_ELT(res, 3, residuals); //residuals
  SET_VECTOR_ELT(res, 4, score); //gradient
  SET_VECTOR_ELT(res, 5, hessian); //hessian
  SET_VECTOR_ELT(res, 6, linpred); //hessian
  SET_VECTOR_ELT(res, 7, converged); //check convergence
  UNPROTECT(9);
  return(res);
}

double newBarL0(double h, double g, double b, double l) {
  double tmp;
  double s = 0;
  tmp = h * b + g;
  if (tmp > 0) s = 1;
  else if (tmp < 0) s = -1;
  if (fabs(tmp) < 2 * sqrt(h * l)) return(0);
  else return((tmp + s * sqrt(pow(tmp, 2) - 4 * l * h)) / (2 * h));
}


// CCD BAR L_0 or L_1
SEXP ccd_bar(SEXP x_, SEXP t2_, SEXP ici_, SEXP wt_, SEXP lambda_,
             SEXP esp_, SEXP max_iter_, SEXP beta0_) {

  //Declaration
  int n = length(t2_);
  int p = length(x_) / n;
  int L = length(lambda_);

  //Output
  SEXP res, beta, Dev, iter, residuals, score, hessian, converged, linpred;
  PROTECT(beta = allocVector(REALSXP, L * p));
  double *b = REAL(beta);
  for (int j = 0; j < (L * p); j++) b[j] = 0;
  PROTECT (score = allocVector(REALSXP, L * n));
  double *s = REAL(score);
  for (int i = 0; i < (L * n); i++) s[i] = 0;
  PROTECT (hessian = allocVector(REALSXP, L * n));
  double *h = REAL(hessian);
  for (int i = 0; i <  (L * n); i++) h[i] = 0;
  PROTECT(residuals = allocVector(REALSXP, n));
  double *r = REAL(residuals);
  PROTECT(Dev = allocVector(REALSXP, L + 1));
  for (int i = 0; i < (L + 1); i++) REAL(Dev)[i] = 0;
  PROTECT(iter = allocVector(INTSXP, L));
  for (int i = 0; i < L; i++) INTEGER(iter)[i] = 0;
  PROTECT(converged = allocVector(INTSXP, L));
  for (int i = 0; i < L; i++) INTEGER(converged)[i] = 0;
  PROTECT(linpred = allocVector(REALSXP, n));
  double *lp = REAL(linpred);
  for (int i = 0; i <  n; i++) lp[i] = 0;

  //Intermediate quantities for internal use (must be freed afterwards!)
  double *a = Calloc(p, double); // Beta from previous iteration
  for (int j = 0; j < p; j++) a[j] = 0;
  double *st = Calloc(n, double);
  for (int i = 0; i < n; i++) st[i] = 0;
  double *w = Calloc(n, double);
  for ( int i = 0; i < n; i++) w[i] = 0;
  double *eta = Calloc(n, double);
  for (int i = 0; i < n; i++) eta[i] = 0;
  double *e = Calloc(n, double);
  for (int i = 0; i < n; i++) e[i] = 0;
  double *diffBeta = Calloc(p, double);
  for (int j = 0; j < p; j++) diffBeta[j] = 1;
  double *accNum1 = Calloc(n, double); //accumulate the backwards numerator
  for (int i = 0; i < n; i++) accNum1[i] = 0;
  double *accNum2 = Calloc(n, double); //acumulate the foreward numerator (weighted)
  for (int i = 0; i < n; i++) accNum2[i] = 0;
  double *accSum = Calloc(n, double); //accumulate sum over both accNum1 and accNum2
  for (int i = 0; i < n; i++) accSum[i] = 0;


  //Pointers for R values
  double *x = REAL(x_);
  double *t2 = REAL(t2_);
  double *wt = REAL(wt_);
  int *ici = INTEGER(ici_);
  double *lam = REAL(lambda_);
  double esp = REAL(esp_)[0];
  int max_iter = INTEGER(max_iter_)[0];
  double *m = REAL(beta0_);

  //internal storage
  double nullDev; //to store null deviance
  double grad, hess, shift, si;
  int i, j, i2; //for loop indices
  double tmp1 = 0; //track backward sum for uncensored events risk set
  double tmp2 = 0; //track forward sum for competing risks risk set
  //end of declaration;
  //initialization

  //Initialize beta_0 = beta^(0)

  //Initialize eta with beta_0
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      e[i] += m[j] * x[j * n + i];
    }
  }

  nullDev = -2 * getLogLikelihood(t2, ici, x, p, n, wt, a); // Calculate null deviance at beta = 0
  REAL(Dev)[0] = nullDev; //Store initial loglikelihood as first element in Dev

  // Initialize a and eta using m and e. (Warm starts will be used afterwards)
  for (int j = 0; j < p; j++) a[j] = m[j];
  for (int i = 0; i < n; i++) eta[i] = e[i];

  //Outer loop for each lambda
  for(int l = 0; l < L; l++) {

    //Start algorithm here
    while (INTEGER(iter)[l] < max_iter) {

      if (REAL(Dev)[l + 1] - nullDev > 0.99 * nullDev) break;

      INTEGER(iter)[l]++;

      //Reset values
      tmp1 = 0, tmp2 = 0;
      for(i = 0; i < n; i++) {
        st[i] = 0;
        w[i] = 0;
      }

      //Backward Scan
      for (i = 0; i < n; i++)
      {
        tmp1 += exp(eta[i]); //track cumulative sum over 1:n
        if (ici[i] != 1) {
          accNum1[i] = 0;
        } else {
          st[i] += 1;
          accNum1[i] = tmp1;
        }

        //Forward Scan (To take into account the competing risks component)
        i2 = (n - 1) - i;
        if (ici[i2] == 2) {
          tmp2 += exp(eta[i2]) / wt[i2];
        }
        if (ici[i2] != 1) {
          accNum2[i2] = 0;
        } else {
          accNum2[i2] = wt[i2] * tmp2; //Sort accNum2 in same order as backward scan
        }
      }

      //calculate risk set for denominator
      for(i = 0; i < n; i++) {
        if (ici[i] != 1) continue;
        accSum[i] = accNum1[i] + accNum2[i];
      }

      for(i = 0; i < n; i++) {
        i2 = (n - 1) - i;
        if(ici[i2] == 2 || ici[i2 - 1] != 1 || i2 == 0) continue;
        if(t2[i2] == t2[i2 - 1]) {
          accSum[i2 - 1] = accSum[i2];
        }
      }



      //calculate score and hessian here
      tmp1 = 0; tmp2 = 0; //reset temporary vals

      //linear scan for non-competing risks (backwards scan)
      for(i = n; i > 0; i--) {
        if(ici[i - 1] == 1) {
          tmp1 += 1 / accSum[i - 1];
          tmp2 += 1 / pow(accSum[i - 1], 2);
          accNum1[i - 1] = tmp1;
          accNum2[i - 1] = tmp2;
        } else if (ici[i - 1] != 1) {
          accNum1[i - 1] = tmp1;
          accNum2[i - 1] = tmp2;
        }
      }

      //Fix ties here:
      for(i = 0; i < n; i++) {
        //only needs to be adjusted consective event times
        if(ici[i] != 1 || ici[i + 1] != 1 || i == (n - 1)) continue;
        if(t2[i] == t2[i + 1]) {
          accNum1[i + 1] = accNum1[i];
          accNum2[i + 1] = accNum2[i];
        }
      }


      //Store into st and w so we can reuse accNum1 and accNum2
      for(i = 0; i < n; i++) {
        st[i] = accNum1[i] * exp(eta[i]);
        w[i] = accNum2[i] * pow(exp(eta[i]), 2);
      }

      //Perform linear scan for competing risks
      tmp1 = 0; tmp2 = 0; //reset tmp vals
      for(i = 0; i < n; i++) {
        accNum1[i] = 0;
        accNum2[i] = 0;
        if(ici[i] == 1) {
          tmp1 += wt[i] / accSum[i];
          tmp2 += pow(wt[i] / accSum[i], 2);
        }
        if(ici[i] != 2) continue;
        accNum1[i] = tmp1;
        accNum2[i] = tmp2;
      }

      //Now combine to produce score and hessian
      for(i = 0; i < n; i++) {
        //First, update st and w and then get score and hessian
        st[i] += accNum1[i] * (exp(eta[i]) / wt[i]);
        w[i] += accNum2[i] * pow(exp(eta[i]) / wt[i], 2);
      }

      for(i= 0; i < n; i++) {
        w[i] = (st[i] - w[i]);
        if(ici[i] != 1) {
          st[i] = - st[i];
        } else {
          st[i] = (1 - st[i]);
        }
      }
      //end calculation of score and hessian

      for (i = 0; i < n; i++){
        if (w[i] == 0) r[i] = 0;
        else r[i] = st[i] / w[i];
      }

      // calculate xwr and xwx & update beta_j
      for (j = 0; j < p; j++) {

        grad = wcrossprod(x, r, w, n, j); // jth component of gradient
        hess = wsqsum(x, w, n, j); // jth component of hessian [l''(b)]
        //New beta_j update
        b[l * p + j] = newBarL0(hess, grad, a[j], lam[l]);

        // Update r
        shift = b[l * p + j] - a[j];
        if (shift != 0) {
          for (i = 0; i < n; i++) {
            si = shift * x[j * n + i]; //low-rank update of beta
            r[i] -= si;
            eta[i] += si;
          }
        } //end shift

      } // End cyclic coordinate-wise optimization

      // Check for convergence (b = current est. a = old estimate)
      INTEGER(converged)[l] = checkFastBarConvergence(b, a, esp, l, p);

      for (j = 0; j < p; j++)
        a[j] = b[l * p + j]; //redefine old est as new est

      //Calculate deviance
      REAL(Dev)[l + 1] = -2 * getLogLikelihood(t2, ici, x, p, n, wt, a);
      for (i = 0; i < n; i++){
        s[l * n + i] = st[i];
        h[l * n + i] = w[i];
        lp[i] = eta[i];
      }
      if (INTEGER(converged)[l])  break;
    } //BAR iterate for l^th lambda
  } // Cycle through all lambda

  res = cleanupNewCRR(a, e, eta, st, w, accNum1, accNum2, accSum, beta, Dev, iter, residuals, score, hessian, linpred, converged);
  return(res);
}
