#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include <random>
#include <vector>
#include <memory>

#include "types.hpp"
#include "device_memory.cuh"
#include "maths.hpp"
#include "cuda_maths.cuh"
#include "tensor.hpp"

class Tensor;

void linear_layer(const Tensor* input, const Tensor* weight, const Tensor* bias, 
				  Tensor* out);
void linear_layer_backward(Tensor* input, Tensor* weight, Tensor* bias, const Tensor* out);
void softmax(const Tensor* input, Tensor* out); // add temperature to computations later
void softmax_backward(Tensor* input, const Tensor* out);
void mseloss(const Tensor* input, const Tensor* expected, Tensor* out);
void mseloss_backward(Tensor* input, const Tensor* expected, Tensor* out);
void relu(const Tensor* input, Tensor* output);
void relu_backward(Tensor* input, const Tensor* output);

void adam_step(std::vector<Tensor*>& parameters, std::vector<std::unique_ptr<Tensor>>& means, std::vector<std::unique_ptr<Tensor>>& squares,
				f32 w_decay, f32 lr, f32 eps, f32 b1, f32 b2, f32 b1_pow, f32 b2_pow);

void xavier_uniform(Tensor* parameters, u32 input_features, u32 output_features);

void zero_memory(Tensor* t);
void zero_memory_grad(Tensor* t);
void grad_fill_ones(Tensor* t);

#endif
