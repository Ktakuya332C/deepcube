#ifndef NN_COST_H_
#define NN_COST_H_

double squared_error_grad(
    double const *const input, double target, double *const feedbacks);
double cross_entropy_loss_grad(double const *const input,
    int target_idx, int n_class, double *const feedbacks);

#endif // NN_COST_H_