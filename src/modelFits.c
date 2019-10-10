#include <math.h>
#include <Rmath.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R.h>
#include <R_ext/Applic.h>
#include <stdlib.h>
#include "utils.h"
#include "sexp.h"
#define LEN sizeof(double)


//////////////////////////////////////////////////////////////////////////////////////
//L2-penalized Fine-Gray regression via Coordinate Descent
SEXP ccd_ridge(SEXP x_, SEXP t2_, SEXP ici_, SEXP wt_, SEXP lambda_,
               SEXP esp_, SEXP max_iter_, SEXP multiplier)
{

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
        for (int i = 0; i < n; i++) {
            st[i] = 0;
            w[i] = 0;
        }

        //Backward Scan [O(n)]
        for (int i = 0; i < n; i++)
        {
            tmp1 += exp(eta[i]); //track cumulative sum over 1:n
            if (ici[i] != 1) {
                // if subject is not an event then accNum[i] = 0;
                accSum[i] = 0;
            } else {
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


        //calculate score and hessian here
        double tmp1 = 0; tmp2 = 0; //reset temporary vals

        //linear scan for non-competing risks (backwards scan)
        for (int i = (n - 1); i >= 0; i--) {
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
        for (int i = 0; i < n; i++) {
            //only needs to be adjusted consective event times
            if (ici[i] != 1 || i == (n - 1) || ici[i + 1] != 1) continue;
            if (t2[i] == t2[i + 1]) {
                accNum1[i + 1] = accNum1[i];
                accNum2[i + 1] = accNum2[i];
            }
        }


        //Store into st and w so we can reuse accNum1 and accNum2
        for (int i = 0; i < n; i++) {
            st[i] = accNum1[i] * exp(eta[i]);
            w[i] = accNum2[i] * pow(exp(eta[i]), 2);
        }

        //Perform linear scan for competing risks
        tmp1 = 0; tmp2 = 0; //reset tmp vals
        for (int i = 0; i < n; i++) {
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
        for (int i = 0; i < n; i++) {
            //First, update st and w and then get score and hessian
            st[i] += accNum1[i] * (exp(eta[i]) / wt[i]);
            w[i] += accNum2[i] * pow(exp(eta[i]) / wt[i], 2);
        }

        for (int i = 0; i < n; i++) {
            w[i] = (st[i] - w[i]);
            if (ici[i] != 1) {
                st[i] = - st[i];
            } else {
                st[i] = (1 - st[i]);
            }
        }

        //end calculation of score and hessian

        for (int i = 0; i < n; i++){
            if (w[i] == 0) r[i] = 0;
            else r[i] = st[i] / w[i];
        }

        // calculate xwr and xwx & update beta_j
        for (int j = 0; j < p; j++) {
            grad = -wcrossprod(x, r, w, n, j) / n; // jth component of gradient [l'(b)]
            hess = wsqsum(x, w, n, j) / n; // jth component of hessian [l''(b)]
            l1 = lam * m[j] / n; //divide by n since we are minimizing the following: -(1/n)l(beta) + lambda * p(beta)
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
        for (int j = 0; j < p; j++)
            a[j] = b[j];

        //Calculate deviance
        REAL(Dev)[0] = -2 * getLogLikelihood(t2, ici, x, p, n, wt, a);

        for (int i = 0; i < n; i++){
            lp[i] = eta[i];
            s[i] = st[i];
            h[i] = w[i];
        }
        if (converged)  break;
    } //for while loop
    res = cleanupCRR(a, eta, st, w, diffBeta, accNum1, accNum2, accSum, beta, Dev, iter, residuals, score, hessian, linpred);
    return(res);
}

//////////////////////////////////////////////////////////////////////////////////////
//BAR Fine-Gray regression via Coordinate-wise update algorithm
SEXP ccd_bar(SEXP x_, SEXP t2_, SEXP ici_, SEXP wt_, SEXP lambda_,
             SEXP esp_, SEXP max_iter_, SEXP beta0_)
{

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
    for (int l = 0; l < L; l++) {

        //Start algorithm here
        while (INTEGER(iter)[l] < max_iter) {

            if (REAL(Dev)[l + 1] - nullDev > 0.99 * nullDev) break;

            INTEGER(iter)[l]++;

            //Reset values
            tmp1 = 0, tmp2 = 0;
            for (int i = 0; i < n; i++) {
                st[i] = 0;
                w[i] = 0;
            }

            //Backward Scan [O(n)]
            for (int i = 0; i < n; i++)
            {
                tmp1 += exp(eta[i]); //track cumulative sum over 1:n
                if (ici[i] != 1) {
                    // if subject is not an event then accNum[i] = 0;
                    accSum[i] = 0;
                } else {
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


            //calculate score and hessian here
            double tmp1 = 0; tmp2 = 0; //reset temporary vals

            //linear scan for non-competing risks (backwards scan)
            for (int i = (n - 1); i >= 0; i--) {
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
            for (int i = 0; i < n; i++) {
                //only needs to be adjusted consective event times
                if (ici[i] != 1 || i == (n - 1) || ici[i + 1] != 1) continue;
                if (t2[i] == t2[i + 1]) {
                    accNum1[i + 1] = accNum1[i];
                    accNum2[i + 1] = accNum2[i];
                }
            }


            //Store into st and w so we can reuse accNum1 and accNum2
            for (int i = 0; i < n; i++) {
                st[i] = accNum1[i] * exp(eta[i]);
                w[i] = accNum2[i] * pow(exp(eta[i]), 2);
            }

            //Perform linear scan for competing risks
            tmp1 = 0; tmp2 = 0; //reset tmp vals
            for (int i = 0; i < n; i++) {
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
            for (int i = 0; i < n; i++) {
                //First, update st and w and then get score and hessian
                st[i] += accNum1[i] * (exp(eta[i]) / wt[i]);
                w[i] += accNum2[i] * pow(exp(eta[i]) / wt[i], 2);
            }

            for (int i = 0; i < n; i++) {
                w[i] = (st[i] - w[i]);
                if (ici[i] != 1) {
                    st[i] = - st[i];
                } else {
                    st[i] = (1 - st[i]);
                }
            }
            //end calculation of score and hessian

            for (int i = 0; i < n; i++){
                if (w[i] == 0) r[i] = 0;
                else r[i] = st[i] / w[i];
            }

            // calculate xwr and xwx & update beta_j
            for (int j = 0; j < p; j++) {

                grad = wcrossprod(x, r, w, n, j) / n; // jth component of gradient
                hess = wsqsum(x, w, n, j) / n; // jth component of hessian [l''(b)]
                //New beta_j update
                b[l * p + j] = newBarL0(hess, grad, a[j], lam[l] / n);

                // Update r
                shift = b[l * p + j] - a[j];
                if (shift != 0) {
                    for (int i = 0; i < n; i++) {
                        si = shift * x[j * n + i]; //low-rank update of beta
                        r[i] -= si;
                        eta[i] += si;
                    }
                } //end shift

            } // End cyclic coordinate-wise optimization

            // Check for convergence (b = current est. a = old estimate)
            INTEGER(converged)[l] = checkFastBarConvergence(b, a, esp, l, p);

            for (int j = 0; j < p; j++)
                a[j] = b[l * p + j]; //redefine old est as new est

            //Calculate deviance
            REAL(Dev)[l + 1] = -2 * getLogLikelihood(t2, ici, x, p, n, wt, a);
            for (int i = 0; i < n; i++){
                s[l * n + i] = st[i];
                h[l * n + i] = w[i];
                lp[i] = eta[i];
            }
            if (INTEGER(converged)[l])  break;
        } //BAR iterate for l^th lambda
    } // Cycle through all lambda

    res = cleanupNewCRR(a, e, eta, st, w, diffBeta, accNum1, accNum2, accSum, beta, Dev, iter, residuals, score, hessian, linpred, converged);
    return(res);
}


//////////////////////////////////////////////////////////////////////////////////////

