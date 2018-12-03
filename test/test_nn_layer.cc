#include "nn_layer.h"
#include <cmath>
#include <cassert>
#include <iostream>

void test_input_layer_forward() {
  InputLayer input_layer(5);
  double in_activations[5] = {1, 2, 3, 4, 5};
  for (int i=0; i<5; i++) {
    input_layer.activations[i] = in_activations[i];
  }
  
  input_layer.forward();
  double expected[5] = {1, 2, 3, 4, 5};
  for (int i=0; i<5; i++) {
    assert( input_layer.activations[i] == expected[i] );
  }
}

void test_dense_layer_forward() {
  InputLayer input_layer(5);
  double in_activations[5] = {1, -1, 1, -1, 1};
  for (int i=0; i<5; i++) {
    input_layer.activations[i] = in_activations[i];
  }
  
  DenseLayer dense_layer(4, &input_layer);
  double weights[20] = {
    1, 2, 3, 4, 5,
    2, 3, 4, 5, 6,
    3, 4, 5, 6, 7,
    4, 5, 6, 7, 8
  };
  double bias[4] = {0, 0, 0, 0};
  dense_layer.init_params(weights, bias);
  
  dense_layer.forward();
  double expected[4] = { 3, 4, 5, 6 };
  for (int i=0; i<4; i++) {
    assert( dense_layer.activations[i] == expected[i] );
  }
}

void test_dense_layer_backward() {
  InputLayer input_layer(5);
  
  DenseLayer dense_layer(4, &input_layer);
  double weights[20] = {
    1, 2, 3, 4, 5,
    2, 3, 4, 5, 6,
    3, 4, 5, 6, 7,
    4, 5, 6, 7, 8
  };
  double bias[4] = {0, 0, 0, 0};
  dense_layer.init_params(weights, bias);
  double in_feedbacks[4] = {-2, 1, -1, 1};
  for (int i=0; i<4; i++) {
    dense_layer.feedbacks[i] = in_feedbacks[i];
  }
  
  double alpha = 1;
  
  dense_layer.backward(alpha);
  double expected[5] = { 1, 0, -1, -2, -3 };
  for (int i=0; i<4; i++) {
    assert( input_layer.feedbacks[i] == expected[i] );
  }
}

void test_dense_layer_save_load() {
  InputLayer input_layer(5);
  double in_activations[5] = {1, -1, 1, -1, 1};
  for (int i=0; i<5; i++) {
    input_layer.activations[i] = in_activations[i];
  }
  
  DenseLayer dense_layer(4, &input_layer);
  dense_layer.init_params();
  dense_layer.forward();
  
  double activations[4];
  for (int i=0; i<4; i++) {
    activations[i] = dense_layer.activations[i];
  }
  
  assert(dense_layer.save("/tmp/", "test_dense_layer_save_load"));
  dense_layer.init_params();
  assert(dense_layer.load("/tmp/", "test_dense_layer_save_load"));
  
  dense_layer.zero_states();
  for (int i=0; i<5; i++) {
    input_layer.activations[i] = in_activations[i];
  }
  dense_layer.forward();
  
  for (int i=0; i<4; i++) {
    assert(std::fabs(activations[i] -  dense_layer.activations[i]) < 1e-3);
  }
}

void test_relu_layer_forward() {
  InputLayer input_layer(5);
  double in_activations[5] = {-1, 2, -1, 1, -4};
  for (int i=0; i<5; i++) {
    input_layer.activations[i] = in_activations[i];
  }
  
  ReluLayer relu_layer(&input_layer);
  
  relu_layer.forward();
  double expected[5] = { 0, 2, 0, 1, 0 };
  for (int i=0; i<5; i++) {
    assert( relu_layer.activations[i] == expected[i] );
  }
}

void test_relu_layer_backward() {
  InputLayer input_layer(5);
  double in_activations[5] = {-1, 3, -1, 1, -1};
  for (int i=0; i<5; i++) {
    input_layer.activations[i] = in_activations[i];
  }
  
  ReluLayer relu_layer(&input_layer);
  double in_feedbacks[5] = {1, -1, -1, 2, 1};
  for (int i=0; i<5; i++) {
    relu_layer.feedbacks[i] = in_feedbacks[i];
  }
  
  double alpha = 1;
  
  relu_layer.forward();
  relu_layer.backward(alpha);
  double expected[5] = { 0, -1, 0, 2, 0 };
  for (int i=0; i<4; i++) {
    assert( input_layer.feedbacks[i] == expected[i] );
  }
}

int main() {
  std::cout << "Start tests" << std::endl;
  test_input_layer_forward();
  test_dense_layer_forward();
  test_dense_layer_backward();
  test_dense_layer_save_load();
  test_relu_layer_forward();
  test_relu_layer_backward();
  std::cout << "Succeed all tests" << std::endl;
}