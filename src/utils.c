#include <math.h>
#include <Rmath.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R.h>
#include <R_ext/Applic.h>
#include <stdlib.h>
#define LEN sizeof(double)

//////////////////////////////////////////////////////////////////////////////////////
// ERIC S. KAWAGUCHI
//////////////////////////////////////////////////////////////////////////////////////
// Utilities for Package

// Weighted cross product of y with jth column of x (x'diag(w)y )
double wcrossprod(double *X, double *y, double *w, int n, int j)
{
    int nn = n * j;
    double val = 0;
    for (int i = 0; i < n; i++) val += X[nn + i] * y[i] * w[i];
    return(val);
}

// Weighted sum of squares of jth column of X (x'diag(w)x)
double wsqsum(double *X, double *w, int n, int j)
{
    int nn = n * j;
    double val = 0;
    for (int i = 0; i < n; i++) val += w[i] * pow(X[nn + i], 2);
    return(val);
}

//Define sgn function: sgn(z) = 1 if z > 0, -1 if z < 0, 0 if z = 0
double sgn(double z)
{
  double s = 0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  return(s);
}

// Criterion for convergence: All coefficients must pass the following |(b_new - b_old) / b_old| < eps
int checkConvergence(double *beta, double *beta_old, double eps, int p)
{
    int converged = 1;
    for (int j = 0; j < p; j++) {
        if (fabs((beta[j] - beta_old[j]) / beta_old[j]) > eps) {
            converged = 0;
            break;
        }
    }
    return(converged);
}

// Criterion for convergence for BAR: Max relative error should be less than eps
int checkFastBarConvergence(double *beta, double *beta_old, double eps, int l, int p)
{
    int converged = 1;
    for (int j = 0; j < p; j++) {
        if (fabs((beta[l * p + j] - beta_old[j])) > eps) {
            converged = 0;
            break;
        }
    }
    return(converged);
}

//Cyclic BAR penalty
double newBarL0(double h, double g, double b, double l)
{
    double tmp;
    double s = 0;
    tmp = h * b + g;
    if (tmp > 0) s = 1;
    else if (tmp < 0) s = -1;
    if (fabs(tmp) < 2 * sqrt(h * l)) return(0);
    else return((tmp + s * sqrt(pow(tmp, 2) - 4 * l * h)) / (2 * h));
}

// Calculate Log-Partial Likelihood
double getLogLikelihood(double *t2, int *ici, double *x, int ncov, int nin, double *wt, double *b)
{
    // Look at Eq (2) from Fu et al. 2017.
    const int p = ncov,  n = nin;
    double tmp1 = 0; //track backward sum for uncensored events risk set
    double tmp2 = 0; //track forward sum for competing risks risk set
    double loglik = 0; //store loglik

    //Pointers to be freed later
    double *eta = Calloc(n, double); //accumulate sum over both accNum1 and accNum2
    for (int i = 0; i < n; i++) eta[i] = 0;
    double *accSum = Calloc(n, double); //accumulate sum over both accNum1 and accNum2
    for (int i = 0; i < n; i++) accSum[i] = 0;

    for (int i =  0; i < n; i++) {
        //initialize values to 0
        eta[i] = 0;
        accSum[i] = 0;

        for (int j = 0; j < p; j++)
            eta[i] += b[j] * x[n * j + i];
    }

    //Note: t2, ici, and x should be ordered in DECREASING order. (First time is largest)
    //Backward Scan [O(n)]
    for (int i = 0; i < n; i++)
    {
        tmp1 += exp(eta[i]); //track cumulative sum over 1:n
        if (ici[i] != 1) {
            // if subject is not an event then accNum[i] = 0;
            accSum[i] = 0;
        } else {
            loglik += eta[i];
            accSum[i] = tmp1;
        }
    }

    //Forward Scan (To take into account the competing risks component) [O(n)]
    for (int i2 = (n - 1); i2 >= 0; i2--) {
        if (ici[i2] == 2) {
            tmp2 += exp(eta[i2]) / wt[i2];
        }
        if (ici[i2] != 1) continue;
        accSum[i2] += wt[i2] * tmp2;
    }


    //taking into account ties [O(n)]
    for (int i2 = (n - 1); i2 >= 0; i2--) {
        if (ici[i2] == 2 || i2 == 0 || ici[i2 - 1] != 1) continue;
        if (t2[i2] == t2[i2 - 1]) {
            accSum[i2 - 1] = accSum[i2];
        }
    }


    //calculate loglik [O(n)]
    for (int i =  0; i < n; i++) {
        if (ici[i] != 1) continue;
        loglik  -= log(accSum[i]);
    }

    Free(eta);
    Free(accSum);
    return loglik;
}


//////////////////////////////////////////////////////////////////////////////////////

//Get score and hessian for finding lambda values:
void getScoreAndHessian (double *t2, int *ici, int *nin, double *wt, double *eta, double *st, double *w, double *r)
{

    const int n = nin[0];
    int i, i2; //for loop indices
    double tmp1 = 0; //track backward sum for uncensored events risk set
    double tmp2 = 0; //track forward sum for competing risks risk set
    //end of declaration;

    double *accNum1 = Calloc(n, double); //accumulate the backwards numerator
    for (int i = 0; i < n; i++) accNum1[i] = 0;
    double *accNum2 = Calloc(n, double); //acumulate the foreward numerator (weighted)
    for (int i = 0; i < n; i++) accNum2[i] = 0;
    double *accSum = Calloc(n, double); //accumulate sum over both accNum1 and accNum2
    for (int i = 0; i < n; i++) accSum[i] = 0;
    //initialization

    //Backward Scan [O(n)]
    for (i = 0; i < n; i++){
        st[i] = 0;
        w[i] = 0;
        r[i] = 0;
        tmp1 += exp(eta[i]); //track cumulative sum over 1:n
        if (ici[i] != 1) {
            // if subject is not an event then accNum[i] = 0;
            accSum[i] = 0;
        } else {
            accSum[i] = tmp1;
        }
    }

    //Forward Scan (To take into account the competing risks component) [O(n)]
    for(i2 = (n - 1); i2 >= 0; i2--) {
        if (ici[i2] == 2) {
            tmp2 += exp(eta[i2]) / wt[i2];
        }
        if (ici[i2] != 1) continue;
        accSum[i2] += wt[i2] * tmp2;
    }


    //taking into account ties [O(n)]
    for(i2 = (n - 1); i2 >= 0; i2--) {
        if (ici[i2] == 2 || i2 == 0 || ici[i2 - 1] != 1) continue;
        if (t2[i2] == t2[i2 - 1]) {
            accSum[i2 - 1] = accSum[i2];
        }
    }


    //calculate score and hessian here
    tmp1 = 0; tmp2 = 0; //reset temporary vals

    //linear scan for non-competing risks (backwards scan)
    for (int i =  (n - 1); i >= 0; i--) {
        if (ici[i] == 1) {
            tmp1 += 1 / accSum[i];
            tmp2 += 1 / pow(accSum[i], 2);
            accNum1[i] = tmp1;
            accNum2[i] = tmp2;
        } else {
            accNum1[i] = tmp1;
            accNum2[i] = tmp2;
        }
    }

    //Fix ties here:
    for (int i =  0; i < n; i++) {
        //only needs to be adjusted consective event times
        if (ici[i] != 1 || i == (n - 1) || ici[i + 1] != 1) continue;
        if (t2[i] == t2[i + 1]) {
            accNum1[i + 1] = accNum1[i];
            accNum2[i + 1] = accNum2[i];
        }
    }


    //Store into st and w so we can reuse accNum1 and accNum2
    for (int i =  0; i < n; i++) {
        st[i] = accNum1[i] * exp(eta[i]);
        w[i] = accNum2[i] * pow(exp(eta[i]), 2);
    }

    //Perform linear scan for competing risks
    tmp1 = 0; tmp2 = 0; //reset tmp vals
    for (int i =  0; i < n; i++) {
        accNum1[i] = 0;
        accNum2[i] = 0;
        if (ici[i] == 1) {
            tmp1 += wt[i] / accSum[i];
            tmp2 += pow(wt[i] / accSum[i], 2);
        }
        if (ici[i] != 2) continue;
        accNum1[i] = tmp1;
        accNum2[i] = tmp2;
    }

    //Now combine to produce score and hessian
    for (int i =  0; i < n; i++) {
        //First, update st and w and then get score and hessian
        st[i] += accNum1[i] * (exp(eta[i]) / wt[i]);
        w[i] += accNum2[i] * pow(exp(eta[i]) / wt[i], 2);
    }

    for (int i =  0; i < n; i++) {
        w[i] = (st[i] - w[i]);
        if (ici[i] != 1) {
            st[i] = - st[i];
        } else {
            st[i] = (1 - st[i]);
        }
        if (w[i] == 0) r[i] = 0;
        else r[i] = st[i] / w[i];
    }

    free(accNum1);
    free(accNum2);
    free(accSum);
}
