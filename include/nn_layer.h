#ifndef NN_LAYER_H_
#define NN_LAYER_H_

#include <string>

class AbstractLayer {
  
public:
  AbstractLayer(int n_neurons, AbstractLayer *const prev_layer);
  virtual ~AbstractLayer();
  
  virtual void forward() = 0;
  virtual void backward(double alpha) = 0;
  
  virtual void apply_grad(double lr);
  virtual void zero_grad();
  virtual void zero_states();

public:
  int const n_neurons;
  double *activations;
  double *feedbacks;

protected:
  AbstractLayer *const prev_layer_;

};

class InputLayer: public AbstractLayer {

public:
  InputLayer(int n_neurons): AbstractLayer(n_neurons, nullptr) {};
  ~InputLayer() {};
  
  void forward() override {};
  void backward(double alpha) override {};
  
  void apply_grad(double lr) override {};
  void zero_grad() override {};
  void zero_states() override;
};

class ReluLayer: public AbstractLayer {

public:
  ReluLayer(AbstractLayer *const prev_layer);
  ~ReluLayer() {};
  
  void forward() override;
  void backward(double alpha) override;
};

class DenseLayer: public AbstractLayer {

public:
  DenseLayer(int n_neurons, AbstractLayer *const prev_layer);
  ~DenseLayer();
  
  void forward() override;
  void backward(double alpha) override;
  
  void init_params();
  void init_params(double const *const weights, double const *const bias);
  
  void apply_grad(double lr) override;
  void zero_grad() override;
  
  // dir_path must end with a slash (/)
  bool save(std::string dir_path, std::string base_name);
  bool load(std::string dir_path, std::string base_name);
  
private:
  double *weights_;
  double *bias_;
  double *weight_grads_;
  double *bias_grads_;

};

#endif // NN_LAYER_H_
