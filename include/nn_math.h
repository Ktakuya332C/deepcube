#ifndef NN_MATH_H_
#define NN_MATH_H_

double relu(double x);
double step(double x);
double sign(double x);
double uniform(double lower, double upper);
void softmax(const double* in_values, double* out_values, int length);
void calc_max(const double*array, int size, double *max_value, int *max_idx);

// y += Ax, A \in M(n, m), row major
void naive_mv(const double* A, const double* x, int n, int m, double* y);

// y += Ax, A \in M(n, m), row major
void cblas_mv(const double* A, const double* x, int n, int m, double* y);

// y += A^T x, A \in M(n, m), row major
void naive_vm(const double* A, const double* x, int n, int m, double* y);

// y += A^T x, A \in M(n, m), row major
void cblas_vm(const double* A, const double* x, int n, int m, double* y);

// C += alpha * AB, A \in M(n, l), B \in M(l, m), row major
void naive_mm(const double* A, const double* B,
    int n, int l, int m, double alpha, double* C);

// C += aloha * AB, A \in M(n, l), B \in M(l, m), row major
void cblas_mm(const double* A, const double* B,
    int n, int l, int m, double alpha, double* C);

#endif // NN_MATH_H_
