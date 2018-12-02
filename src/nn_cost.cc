#include "nn_cost.h"
#include <cmath>
#include <cfloat>
#include <algorithm>

double squared_error_grad(
    double const *const input, double target, double *const feedbacks) {
  feedbacks[0] = input[0] - target;
  return (input[0] - target) * (input[0] - target) / 2.0;
}

double cross_entropy_loss_grad(double const *const input,
    int target_idx, int n_class, double *const feedbacks) {
  double max_input = *std::max_element(input, input+n_class);
  double sum_exp = 0.0;
  for (int i=0; i<n_class; i++) {
    sum_exp += exp(input[i] - max_input);
  }
  for (int i=0; i<n_class; i++) {
    feedbacks[i] = exp(input[i] - max_input) / sum_exp;
    if ( i == target_idx) feedbacks[i] -= 1.0;
  }
  return -input[target_idx] + max_input + log(sum_exp);
}
