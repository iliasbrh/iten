#ifndef FUNCTIONAL_HPP
#define FUNCTIONAL_HPP

#include <random>

#include "maths.hpp"
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


void xavier_uniform(Tensor* parameters, u32 input_features, u32 output_features);


#endif
