#include <iostream>
#include <string>
#include <cmath>
#include <map>
#include <numeric>
#include <algorithm>
#include "cube.h"
#include "nn_math.h"
#include "nn_layer.h"

const std::string save_dir = "data/";
const int n_scramble = 6;
const int max_try = 1000;
const double c = 5.0;

std::map<Move, std::string> move_str {
  {U_CC, "U'"}, {U_CW, "U"}, {D_CC, "D'"}, {D_CW, "D"},
  {F_CC, "F'"}, {F_CW, "F"}, {B_CC, "B'"}, {B_CW, "B"},
  {R_CC, "R'"}, {R_CW, "R"}, {L_CC, "L'"}, {L_CW, "L"},
};

struct Node {
  bool is_visited;
  int N[n_move];
  double L[n_move];
  double W[n_move];
  double P[n_move];
  Node *A[n_move];
  Node() {
    is_visited = false;
    std::fill(N, N+n_move, 0);
    std::fill(W, W+n_move, 0.0);
    std::fill(P, P+n_move, 0.0);
    std::fill(A, A+n_move, nullptr);
  }
};

double search(Move *history, int *depth, Node *cur_node, Cube *cur_cube,
    InputLayer &input, AbstractLayer &value, AbstractLayer &policy) {
  
  if (cur_node->is_visited) {
    // Calculate U
    int sum_N = 0;
    for (int i=0; i<n_move; i++) {
      sum_N += cur_node->N[i];
    }
    double U[n_move];
    for (int i=0; i<n_move; i++) {
      U[i] = c * cur_node->P[i] * sqrt(sum_N) / (1 + cur_node->N[i]);
    }
    
    // Calculate Q
    double Q[n_move];
    for (int i=0; i<n_move; i++) {
      Q[i] = cur_node->W[i] - cur_node->L[i];
    }
    
    // Determine which action to take
    int max_idx = 0;
    for (int i=1; i<n_move; i++) {
      if (U[i] + Q[i] > U[max_idx] + Q[max_idx]) max_idx = i;
    }
    
    // Rotate cube
    Move move = static_cast<Move>(max_idx);
    cur_cube->rotate(move);
    history[*depth] = move;
    *depth += 1;
    
    // Get feedback from child nodes
    double pred_value = search(history, depth,
        cur_node->A[max_idx], cur_cube, input, value, policy);
    cur_node->W[max_idx] = std::max(cur_node->W[max_idx], pred_value);
    cur_node->N[max_idx] += 1;
    return pred_value;
    
  } else {
    cur_node->is_visited = true;
    
    // Create child nodes
    for (int i=0; i<n_move; i++) {
      cur_node->A[i] = new Node();
    }
    
    // Fill values of P
    cur_cube->get_state(input.activations);
    policy.forward();
    softmax(policy.activations, cur_node->P, n_move);
    policy.zero_states();
    
    // Feedback predicted value
    cur_cube->get_state(input.activations);
    value.forward();
    double pred_value = value.activations[0];
    std::fill(cur_node->W, cur_node->W+n_move, pred_value);
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
  std::cout << "Scramble the cube. The order of moves is" << std::endl;
  Cube original;
  for (int i=0; i<n_scramble; i++) {
    Move move = original.rotate_random();
    std::cout << move_str[move] << " ";
  }
  std::cout << std::endl;
  
  // MCTS
  Cube cube;
  Node root;
  Move history[max_try];
  int depth = 0;
  for (int i=0; i<max_try; i++) {
    cube.restore(original);
    depth = 0;
    
    search(history, &depth, &root, &cube,
        input_layer, dense_layer_v2, dense_layer_p2);
    
    if (cube.is_solved()) {
      std::cout << "Solved the cube. The order of moves is" << std::endl;
      for (int i=0; i<depth; i++) {
        std::cout << move_str[history[i]] << " ";
      }
      std::cout << std::endl;
      return 0;
    }
    
  }
  std::cout << "Could not solve the cube" << std::endl;
  
  // Need to free memories allocated for nodes ...
  
  return 0;
}