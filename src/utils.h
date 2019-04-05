#ifndef UTILS_H
#define UTILS_H

// This is the content of the .h file, which is where the declarations go
double wcrossprod(double *X, double *y, double *w, int n, int j);
double wsqsum(double *X, double *w, int n, int j);
double sgn(double z);
int checkConvergence(double *beta, double *beta_old, double eps, int p);
int checkFastBarConvergence(double *beta, double *beta_old, double eps, int l, int p);
double newBarL0(double h, double g, double b, double l);
double getLogLikelihood(double *t2, int *ici, double *x, int ncov, int nin, double *wt, double *b);
void getBreslowJumps(double *t2, int *ici, double *x, int *ncov, int *nin, double *wt, double *b, double *bj);
void getScoreAndHessian (double *t2, int *ici, int *nin, double *wt, double *eta, double *st, double *w, double *r);
// This is the end of the header guard
#endif
