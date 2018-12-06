#include "nn_math.h"
#include <cassert>
#include <cmath>
#include <iostream>

void test_softmax() {
  double in_values[5] = {1, 2, -1, 3, 5};
  double out_values[5];
  softmax(in_values, out_values, 5);
  
  double expected[5] = {
      0.01518815, 0.04128566, 0.00205549, 0.11222606, 0.82924464};
  for (int i=0; i<5; i++) {
    assert(std::fabs(out_values[i] - expected[i]) < 1e-4);
  }
}

void test_naive_mv() {
  double A[15] = {
    1, 2, 3, 4, 5,
    2, 3, 4, 5, 6,
    3, 4, 5, 6, 7
  };
  double x[5] = { -1, 1, -1, 1, -1 };
  double y[3] = {1, 0, 0};
  
  double expected[3] = { -2, -4, -5 };
  naive_mv(A, x, 3, 5, y);
  for (int i=0; i<3; i++) {
    assert( y[i] == expected[i] );
  }
}

void test_cblas_mv() {
  double A[15] = {
    1, 2, 3, 4, 5,
    2, 3, 4, 5, 6,
    3, 4, 5, 6, 7
  };
  double x[5] = { -1, 1, -1, 1, -1 };
  double y[3] = {1, 0, 0};
  
  double expected[3] = { -2, -4, -5 };
  cblas_mv(A, x, 3, 5, y);
  for (int i=0; i<3; i++) {
    assert( y[i] == expected[i] );
  }
}

void test_naive_vm() {
  double A[15] = {
    1, 2, 3, 4, 5,
    2, 3, 4, 5, 6,
    3, 4, 5, 6, 7
  };
  double x[3] = { -1, -1, 1 };
  double y[5] = {1, 0, 0, 0, 0};
  
  double expected[5] = { 1, -1, -2, -3, -4 };
  naive_vm(A, x, 3, 5, y);
  for (int i=0; i<5; i++) {
    assert( y[i] == expected[i] );
  }
}

void test_cblas_vm() {
  double A[15] = {
    1, 2, 3, 4, 5,
    2, 3, 4, 5, 6,
    3, 4, 5, 6, 7
  };
  double x[3] = { -1, -1, 1 };
  double y[5] = {1, 0, 0, 0, 0};
  
  double expected[5] = { 1, -1, -2, -3, -4 };
  cblas_vm(A, x, 3, 5, y);
  for (int i=0; i<5; i++) {
    assert( y[i] == expected[i] );
  }
}

void test_naive_mm() {
  double A[15] = {
    1, 2, 3, 4, 5,
    2, 3, 4, 5, 6,
    3, 4, 5, 6, 7
  };
  double B[20] = {
    -1, 1, -1, 1,
    2, -2, 2, -2,
    -3, 3, -3, 3,
    1, -1, 1, -1,
    -2, 2, -2, 2
  };
  double C[12] = {
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };
  double alpha = 1;
  
  double expected[12] = {
    -11, 12, -12, 12,
    -15, 15, -15, 15,
    -18, 18, -18, 18
  };
  naive_mm(A, B, 3, 5, 4, alpha, C);
  for (int i=0; i<12; i++) {
    assert( C[i] == expected[i] );
  }
}

void test_cblas_mm() {
  double A[15] = {
    1, 2, 3, 4, 5,
    2, 3, 4, 5, 6,
    3, 4, 5, 6, 7
  };
  double B[20] = {
    -1, 1, -1, 1,
    2, -2, 2, -2,
    -3, 3, -3, 3,
    1, -1, 1, -1,
    -2, 2, -2, 2
  };
  double C[12] = {
    1, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };
  double alpha = 1;
  
  double expected[12] = {
    -11, 12, -12, 12,
    -15, 15, -15, 15,
    -18, 18, -18, 18
  };
  cblas_mm(A, B, 3, 5, 4, alpha, C);
  for (int i=0; i<12; i++) {
    assert( C[i] == expected[i] );
  }
}

int main() {
  std::cout << "Start tests" << std::endl;
  test_softmax();
  test_naive_mv();
  test_cblas_mv();
  test_naive_vm();
  test_cblas_vm();
  test_naive_mm();
  test_cblas_mm();
  std::cout << "Succeed all tests" << std::endl;
}