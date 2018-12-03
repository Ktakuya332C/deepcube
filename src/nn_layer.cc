#include "nn_layer.h"
#include "nn_math.h"
#include <cmath>
#include <fstream>
#include <algorithm>

/**
* AbstractLayer
*/

AbstractLayer::AbstractLayer(int n_neurons, AbstractLayer *const prev_layer)
    : n_neurons(n_neurons), prev_layer_(prev_layer) {
  activations = new double[n_neurons];
  std::fill(activations, activations+n_neurons, 0.0);
  feedbacks = new double[n_neurons];
  std::fill(feedbacks, feedbacks+n_neurons, 0.0);
}

AbstractLayer::~AbstractLayer() {
  delete[] activations;
  delete[] feedbacks;
}

void AbstractLayer::apply_grad(double lr) {
  prev_layer_->apply_grad(lr);
}

void AbstractLayer::zero_grad() {
  prev_layer_->zero_grad();
}

void AbstractLayer::zero_states() {
  std::fill(activations, activations+n_neurons, 0.0);
  std::fill(feedbacks, feedbacks+n_neurons, 0.0);
  prev_layer_->zero_states();
}

/*
* InputLayer
*/

void InputLayer::zero_states() {
  std::fill(activations, activations+n_neurons, 0.0);
  std::fill(feedbacks, feedbacks+n_neurons, 0.0);
}

/*
* ReluLayer
*/

ReluLayer::ReluLayer(AbstractLayer *const prev_layer)
    : AbstractLayer(prev_layer->n_neurons, prev_layer) {};

void ReluLayer::forward() {
  prev_layer_->forward();
  for (int i=0; i<n_neurons; i++) {
    activations[i] = relu(prev_layer_->activations[i]);
  }
}

void ReluLayer::backward(double alpha) {
  for (int i=0; i<n_neurons; i++) {
    prev_layer_->feedbacks[i] = feedbacks[i] * step(prev_layer_->activations[i]);
  }
  prev_layer_->backward(alpha);
}

/*
* DenseLayer
*/

DenseLayer::DenseLayer(int n_neurons, AbstractLayer *const prev_layer)
    : AbstractLayer(n_neurons, prev_layer) {
  weights_ = new double[n_neurons * prev_layer_->n_neurons];
  std::fill(weights_, weights_ + n_neurons*prev_layer_->n_neurons, 0.0);
  bias_ = new double[n_neurons];
  std::fill(bias_, bias_ + n_neurons, 0.0);
  weight_grads_ = new double[n_neurons * prev_layer_->n_neurons];
  std::fill(weight_grads_, weight_grads_ + n_neurons*prev_layer_->n_neurons, 0.0);
  bias_grads_ = new double[n_neurons];
  std::fill(bias_grads_, bias_grads_ + n_neurons, 0.0);
}

DenseLayer::~DenseLayer() {
  delete[] weights_;
  delete[] bias_;
  delete[] weight_grads_;
  delete[] bias_grads_;
}

void DenseLayer::forward() {
  prev_layer_->forward();
  cblas_mv(weights_, prev_layer_->activations,
      n_neurons, prev_layer_->n_neurons, activations);
  for (int i=0; i<n_neurons; i++) {
    activations[i] += bias_[i];
  }
}

void DenseLayer::backward(double alpha) {
  // Calculate gradients of this layer
  cblas_mm(feedbacks, prev_layer_->activations,
      n_neurons, 1, prev_layer_->n_neurons, alpha, weight_grads_);
  for (int i=0; i<n_neurons; i++) {
    bias_grads_[i] += alpha * feedbacks[i];
  }
  // Calculate gradients of previous layer
  cblas_vm(weights_, feedbacks,
      n_neurons, prev_layer_->n_neurons, prev_layer_->feedbacks);
  prev_layer_->backward(alpha);
}

void DenseLayer::init_params() {
  // Implements glorot uniform initialization
  double limit = sqrt(6.0 / (n_neurons + prev_layer_->n_neurons));
  for (int i=0; i<n_neurons*prev_layer_->n_neurons; i++) {
    weights_[i] = uniform(-limit, limit);
  }
  for (int i=0; i<n_neurons; i++) {
    bias_[i] = 0.0;
  }
}

void DenseLayer::init_params(
    double const *const weights, double const *const bias) {
  int prev_n_neurons = prev_layer_->n_neurons;
  for (int i=0; i<n_neurons; i++) {
    for (int j=0; j<prev_n_neurons; j++) {
      weights_[i*prev_n_neurons + j] = weights[i*prev_n_neurons + j];
    }
    bias_[i] = bias[i];
  }
}

void DenseLayer::apply_grad(double lr) {
  // Implements Rprop
  for (int i=0; i<n_neurons*prev_layer_->n_neurons; i++) {
    weights_[i] -= lr * sign(weight_grads_[i]);
  }
  for (int i=0; i<n_neurons; i++) {
    bias_[i] -= lr * sign(bias_grads_[i]);
  }
  prev_layer_->apply_grad(lr);
}

void DenseLayer::zero_grad() {
  for (int i=0; i<n_neurons*prev_layer_->n_neurons; i++) {
    weight_grads_[i] = 0.0;
  }
  for (int i=0; i<n_neurons; i++) {
    bias_grads_[i] = 0.0;
  }
  prev_layer_->zero_grad();
}

bool DenseLayer::save(std::string dir_path, std::string base_name) {
  const std::string path_w = dir_path + base_name + "_weight";
  const std::string path_b = dir_path + base_name + "_bias";
  std::ofstream out_w(path_w, std::ios::out | std::ios::binary);
  std::ofstream out_b(path_b, std::ios::out | std::ios::binary);
  if (out_w.is_open() && out_b.is_open()) {
    out_w.write((char*)weights_,
        sizeof(double)*n_neurons*prev_layer_->n_neurons);
    out_b.write((char*)bias_,
        sizeof(double)*n_neurons);
    out_w.close();
    out_b.close();
    return true;
  } else {
    return false;
  }
}

bool DenseLayer::load(std::string dir_path, std::string base_name) {
  const std::string path_w = dir_path + base_name + "_weight";
  const std::string path_b = dir_path + base_name + "_bias";
  std::ifstream in_w(path_w, std::ios::in | std::ios::binary);
  std::ifstream in_b(path_b, std::ios::in | std::ios::binary);
  if (in_w.is_open() && in_b.is_open()) {
    in_w.read((char*)weights_,
        sizeof(double)*n_neurons*prev_layer_->n_neurons);
    in_b.read((char*)bias_,
        sizeof(double)*n_neurons);
    in_w.close();
    in_b.close();
    return true;
  } else {
    return false;
  }
}
