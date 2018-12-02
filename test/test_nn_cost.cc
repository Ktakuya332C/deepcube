#include "nn_cost.h"
#include <cassert>
#include <cmath>
#include <iostream>

void test_squared_error_grad() {
  double input[1] = {1.0};
  double target = 3.0;
  double feedbacks[1];
  
  double cost = squared_error_grad(input, target, feedbacks);
  assert(cost == 2.0);
  assert(feedbacks[0] == -2.0);
}

void test_cross_entropy_loss_grad() {
  double input[4] = {1, 2, 3, 4};
  double target_idx = 2;
  double feedbacks[4];
  
  double cost = cross_entropy_loss_grad(input, target_idx, 4, feedbacks);
  assert(std::fabs(cost - 1.4401896985611953) < 1e-4);
  double expected[4] = {0.0320586 , 0.08714432, -0.76311718, 0.64391426};
  for (int i=0; i<4; i++) {
    assert(std::fabs(feedbacks[i] - expected[i]) < 1e-4);
  }
}

int main() {
  std::cout << "Start tests" << std::endl;
  test_squared_error_grad();
  test_cross_entropy_loss_grad();
  std::cout << "Succeed all tests" << std::endl;
}