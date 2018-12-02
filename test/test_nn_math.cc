#include "nn_math.h"
#include <cassert>
#include <iostream>

void test_naive_mv() {
  double A[15] = {
    1, 2, 3, 4, 5,
    2, 3, 4, 5, 6,
    3, 4, 5, 6, 7
  };
  double x[5] = { -1, 1, -1, 1, -1 };
  double y[3] = {0, 0, 0};
  
  double expected[3] = { -3, -4, -5 };
  naive_mv(A, x, 3, 5, y);
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
  double y[5] = {0, 0, 0, 0, 0};
  
  double expected[5] = { 0, -1, -2, -3, -4 };
  naive_vm(A, x, 3, 5, y);
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
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0
  };
  double alpha = 1;
  
  double expected[12] = {
    -12, 12, -12, 12,
    -15, 15, -15, 15,
    -18, 18, -18, 18
  };
  naive_mm(A, B, 3, 5, 4, alpha, C);
  for (int i=0; i<12; i++) {
    assert( C[i] == expected[i] );
  }
}

int main() {
  std::cout << "Start tests" << std::endl;
  test_naive_mv();
  test_naive_vm();
  test_naive_mm();
  std::cout << "Succeed all tests" << std::endl;
}