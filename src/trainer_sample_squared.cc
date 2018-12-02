#include <iostream>
#include "nn_cost.h"
#include "nn_layer.h"

int main() {
  double inputs[4][3] = {
    {0, 0, 1},
    {1, 1, 1},
    {1, 0, 1},
    {0, 1, 1}
  };
  double output[4] = {0, 1, 1, 0};
  
  InputLayer input_layer(3);
  DenseLayer dense_layer1(3, &input_layer);
  dense_layer1.init_params();
  ReluLayer relu_layer(&dense_layer1);
  DenseLayer dense_layer2(1, &relu_layer);
  dense_layer2.init_params();
  
  for (int epoch=0; epoch<100; epoch++) {
    double cost = 0.0;
    for (int i=0; i<4; i++) {
      for (int j=0; j<3; j++) {
        input_layer.activations[j] = inputs[i][j];
      }
      dense_layer2.forward();
      cost += squared_error_grad(
          dense_layer2.activations, output[i], dense_layer2.feedbacks);
      dense_layer2.backward(1.0);
      dense_layer2.zero_states();
    }
    dense_layer2.apply_grad(0.1);
    dense_layer2.zero_grad();
    std::cout << "Epoch" << epoch+1 << " cost " << cost / 4.0 << std::endl;
  }
  
}