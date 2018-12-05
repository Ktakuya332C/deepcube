#include <iostream>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "cube.h"
#include "nn_layer.h"

const std::string save_dir = "/tmp/";
const int n_scramble = 1;
const int max_depth = 100;
const double c = 0.0;
const double nu = 0.0;

struct Node {
  int N[n_move];
  double L[n_move];
  double W[n_move];
  double P[n_move];
  Node *A[n_move];
  Node(double *P_in) {
    std::fill(N, N+n_move, 0);
    std::fill(L, L+n_move, 0);
    std::fill(W, W+n_move, 0.0);
    std::copy(P, P+n_move, P_in);
    std::fill(A, A+n_move, nullptr);
  }
};

double search(Node *cur_node, Cube &cur_cube,
    InputLayer &input, AbstractLayer &value, AbstractLayer &policy) {
  if (cur_node) {
    int sum_N = 0;
    for (int i=0; i<n_move; i++) {
      sum_N += cur_node->N[i];
    }
    double U[n_move];
    for (int i=0; i<n_move; i++) {
      U[i] = c * cur_node->P[i] * sqrt(sum_N) / (1 + cur_node->N[i]);
    }
    double Q[n_move];
    for (int i=0; i<n_move; i++) {
      Q[i] = cur_node->W[i] - cur_node->L[i];
    }
    int max_idx = 0;
    for (int i=1; i<n_move; i++) {
      if (U[i] + Q[i] > U[max_idx] + Q[max_idx]) max_idx = i;
    }
    cur_node->L[max_idx] += nu;
    
    cur_cube.rotate(static_cast<Move>(max_idx));
    double pred_value = search(
        cur_node->A[max_idx], cur_cube, input, value, policy);
    cur_node->W[max_idx] = std::max(cur_node->W[max_idx], pred_value);
    return pred_value;
    
  } else { // if cur_node is nullptr
    cur_cube.get_state(input.activations);
    policy.forward();
    cur_node = new Node(policy.activations);
    policy.zero_states();
    value.forward();
    double pred_value = value.activations[0];
    value.zero_states();
    return pred_value;
  }
}

int main() {
  // Prepare network
  InputLayer input_layer(state_size);
  DenseLayer dense_layer1(512, &input_layer);
  dense_layer1.load(save_dir, "dense_layer1");
  ReluLayer relu_layer1(&dense_layer1);
  DenseLayer dense_layer2(256, &relu_layer1);
  dense_layer2.load(save_dir, "dense_layer2");
  ReluLayer relu_layer2(&dense_layer2);
  
  DenseLayer dense_layer_p1(256, &relu_layer2);
  dense_layer_p1.load(save_dir, "dense_layer_p1");
  ReluLayer relu_layer_p1(&dense_layer_p1);
  DenseLayer dense_layer_p2(n_move, &relu_layer_p1);
  dense_layer_p2.load(save_dir, "dense_layer_p2");
  
  DenseLayer dense_layer_v1(256, &relu_layer2);
  dense_layer_v1.load(save_dir, "dense_layer_v1");
  ReluLayer relu_layer_v1(&dense_layer_v1);
  DenseLayer dense_layer_v2(1, &relu_layer_v1);
  dense_layer_v2.load(save_dir, "dense_layer_v2");
  
  // Scramble the cube
  Cube original;
  for (int i=0; i<n_scramble; i++) {
    original.rotate_random();
  }
  
  // MCTS
  Cube cube;
  Node *root = nullptr;
  for (int i=0; i<max_depth; i++) {
    cube.restore(original);
    search(root, cube, input_layer, dense_layer_v2, dense_layer_p2);
    if (cube.is_solved()) {
      std::cout << "Solved the cube" << std::endl;
      return 0;
    }
  }
  std::cout << "Could not solve the cube" << std::endl;
  return 0;
}