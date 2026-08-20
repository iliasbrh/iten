#include "cuda_runtime.h"
#include "types.hpp"


__global__ void _mat_add_kernel(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, 
			  b32 cast_A, b32 cast_B, b32 cast_out);
__global__ void _mat_sub_kernel(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, 
			  b32 cast_A, b32 cast_B, b32 cast_out);
__global__ void _mat_mul_kernel(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, u32 P, 
			  b32 cast_A, b32 cast_B, b32 cast_out, 
			  b32 zero_out = true);
__global__ void _mat_mul_transpose_A_kernel(const f32* A, const f32* B, f32* out, 
				         u32 n_lines, u32 M, u32 N, u32 P, 
						 b32 cast_A, b32 cast_B, b32 cast_out,
						 b32 zero_out = true);
__global__ void _mat_mul_transpose_B_kernel(const f32* A, const f32* B, f32* out, 
						 u32 n_lines, u32 M, u32 N, u32 P, 
						 b32 cast_A, b32 cast_B, b32 cast_out,
						 b32 zero_out = true);
__global__ void _linear_layer_kernel(const f32* input, const f32* w, const f32* b, f32* out,
				   u32 n_lines, u32 in_features, u32 out_features);
__global__ void _linear_layer_backward_kernel(const f32* input, const f32* w, const f32* downstream_grads,
						    f32* input_grad, f32* w_grad, f32* b_grad,
							u32 n_lines, u32 in_features, u32 out_features);
__global__ void _softmax_kernel(const f32* A, f32* out,
		      u32 n_lines, u32 last_dim);
__global__ void _softmax_backward_kernel(const f32* downstream_grads, const f32* softmaxs, f32* input_grad,
				       u32 n_lines, u32 last_dim);
__global__ void _mseloss_kernel(const f32* A, const f32* expected, f32* out,
		      u32 size);
__global__ void _mseloss_backward_kernel(const f32* A, const f32* expected, 
				       const f32* downstream_grad, f32* input_grad,
					   u32 size);


__global__ void _relu_kernel(const f32* A, f32* out,
		       u32 size);
__global__ void _relu_backward_kernel(const f32* downstream_grads, const f32* input_data, f32* input_grad,
						u32 size);
__global__ void _mat_exp_kernel(const f32* A, f32* out, 
		     u32 size);
__global__ void _mat_sum_kernel(const f32* A, f32* out,
		    u32 size);
__global__ void _mat_scale_kernel(const f32* A, f32* out, 
		       f32 scaler, 
			   u32 size);


__global__ void _adam_step_kernel(f32* parameters_grad, f32* parameters, f32* means, f32* squares, 
				f32 w_decay, f32 lr, f32 eps, f32 b1, f32 b2, f32 b1_pow, f32 b2_pow,
				u32 size);
__global__ void _set_value_kernel(f32* data, f32 value, u32 size);
