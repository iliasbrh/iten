#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <cmath>
#include "types.h"

u32 multiply_vector(const std::vector<u32> &vec); // product of vector's elements, helper for tensor.size

/////////////////////////////
//       TRUE MATHS        //
// soon to be parallelized //
/////////////////////////////

// the cast arguments are to cast for batch size
// batch size can be replaced by num_lines since operations don't need to know if it's the batch size or anything else (useful for multi head attention layers later)

void _mat_add(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, 
			  b32 cast_A, b32 cast_B, b32 cast_out);
void _mat_sub(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, 
			  b32 cast_A, b32 cast_B, b32 cast_out);
void _mat_mul(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, u32 P, 
			  b32 cast_A, b32 cast_B, b32 cast_out, 
			  b32 zero_out = true);
void _mat_mul_transpose_A(const f32* A, const f32* B, f32* out, 
				         u32 n_lines, u32 M, u32 N, u32 P, 
						 b32 cast_A, b32 cast_B, b32 cast_out,
						 b32 zero_out = true);
void _mat_mul_transpose_B(const f32* A, const f32* B, f32* out, 
						 u32 n_lines, u32 M, u32 N, u32 P, 
						 b32 cast_A, b32 cast_B, b32 cast_out,
						 b32 zero_out = true);
void _linear_layer(const f32* input, const f32* w, const f32* b, f32* out,
				   u32 n_lines, u32 in_features, u32 out_features);
void _linear_layer_backward(const f32* input, const f32* w, const f32* downstream_grads,
						    f32* input_grad, f32* w_grad, f32* b_grad,
							u32 n_lines, u32 in_features, u32 out_features);
void _softmax(const f32* A, f32* out,
		      u32 n_lines, u32 last_dim); // softmax computed along the last dimension
void _softmax_backward(const f32* downstream_grads, const f32* softmaxs, f32* input_grad,
				       u32 n_lines, u32 last_dim);
void _mseloss(const f32* A, const f32* expected, f32* out,
		      u32 size); // size is the product of the vector size and the batch size (since anyway we average on both)
void _mseloss_backward(const f32* A, const f32* expected, 
				       const f32* downstream_grad, f32* input_grad,
					   u32 size);


// element-wise operations (matrix shape not needed as parameter)
void _relu(const f32* A, f32* out,
		       u32 size);
void _relu_backward(const f32* downstream_grads, const f32* input_data, f32* input_grad,
						u32 size);
void _mat_exp(const f32* A, f32* out, 
		     u32 size);
void _mat_sum(const f32* A, f32* out,
		    u32 size);
void _mat_scale(const f32* A, f32* out, 
		       f32 scaler, 
			   u32 size);


void _adam_step(f32* parameters_grad, f32* parameters, f32* means, f32* squares, 
				f32 w_decay, f32 lr, f32 eps, f32 b1, f32 b2, f32 b1_pow, f32 b2_pow,
				u32 size);


#endif
