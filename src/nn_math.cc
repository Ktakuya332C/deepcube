#include "nn_math.h"
#include <cblas.h>
#include <random>

std::random_device rd;
std::mt19937 gen(rd());

double relu(double x) {
  if (x > 0.0) {
    return x;
  } else {
    return 0.0;
  }
}

double step(double x) {
  if (x > 0.0) {
    return 1.0;
  } else {
    return 0.0;
  }
}

double sign(double x) {
  if (x > 0.0) {
    return 1.0;
  } else {
    return -1.0;
  }
}

double uniform(double lower, double upper) {
  std::uniform_real_distribution<> dis(lower, upper);
  return dis(gen);
}

void naive_mv(const double* A, const double* x, int n, int m, double* y) {
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      y[i] += A[i*m + j] * x[j];
    }
  }
};

void cblas_mv(const double* A, const double* x, int n, int m, double* y) {
  cblas_dgemv(CblasRowMajor, CblasNoTrans, n, m, 1, A, m, x, 1, 1, y, 1);
};

void naive_vm(const double* A, const double* x, int n, int m, double* y) {
  for (int i=0; i<m; i++) {
    for (int j=0; j<n; j++) {
      y[i] += A[j*m + i] * x[j];
    }
  }
}

void cblas_vm(const double* A, const double* x, int n, int m, double* y) {
  cblas_dgemv(CblasRowMajor, CblasTrans, n, m, 1, A, m, x, 1, 1, y, 1);
}

void naive_mm(const double* A, const double* B,
    int n, int l, int m, double alpha, double* C) {
  for (int i=0; i<n; i++) {
    for (int j=0; j<m; j++) {
      for (int k=0; k<l; k++) {
        C[i*m + j] += alpha * A[i*l + k] * B[k*m + j];
      }
    }
  }
};

void cblas_mm(const double* A, const double* B,
    int n, int l, int m, double alpha, double* C) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              n, m, l, alpha, A, l, B, m, 1, C, m);
};
