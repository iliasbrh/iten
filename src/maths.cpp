#include "maths.hpp"

u32 multiply_vector(const std::vector<u32> &vec) {
		u32 result = 1;
		for (const u32 element : vec) {result *= element;}
		return result;
}

void _mat_add(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, 
			  b32 cast_A, b32 cast_B, b32 cast_out) {
		for (u32 b = 0; b < n_lines; b++)
				for (u32 row = 0; row < M; row++)
						for (u32 col = 0; col < N; col++)
								out[(1-cast_out)*b*M*N+row*N+col] = 
										A[(1-cast_A)*b*M*N+row*N+col] 
										+ B[(1-cast_B)*b*M*N+row*N+col];
}
void _mat_sub(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, 
			  b32 cast_A, b32 cast_B, b32 cast_out) {
		for (u32 b = 0; b < n_lines; b++)
				for (u32 row = 0; row < M; row++)
						for (u32 col = 0; col < N; col++)
								out[(1-cast_out)*b*M*N+row*N+col] = 
										A[(1-cast_A)*b*M*N+row*N+col] 
										- B[(1-cast_B)*b*M*N+row*N+col];
}
void _mat_mul(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, u32 P, 
			  b32 cast_A, b32 cast_B, b32 cast_out,
			  b32 zero_out) {
		if (zero_out) memset(out, 0.0f, n_lines*M*P*sizeof(f32));

		for (u32 b = 0; b < n_lines; b++)
				for (u32 k=0; k < N; k++)
						for (u32 row = 0; row < M; row++)
								for (u32 col = 0; col < P; col++)
										out[(1-cast_out)*b*M*P+row*P+col] += 
												A[(1-cast_A)*b*M*N + row*N + k]
												* B[(1-cast_B)*b*N*P + k*P + col];
}
void _mat_mul_transpose_A(const f32* A, const f32* B, f32* out, 
				         u32 n_lines, u32 M, u32 N, u32 P, 
						 b32 cast_A, b32 cast_B, b32 cast_out,
						 b32 zero_out) {
		// A of shape (N, M), B of shape (N, P), out of shape (M, P)
		if (zero_out) memset(out, 0.0f, n_lines*M*P*sizeof(f32));

		for (u32 b = 0; b < n_lines; b++)
				for (u32 k=0; k < N; k++)
						for (u32 row = 0; row < M; row++)
								for (u32 col = 0; col < P; col++)
										out[(1-cast_out)*b*M*P+row*P+col] += 
												A[(1-cast_A)*b*M*N + k*M + row]
												* B[(1-cast_B)*b*N*P + k*P + col];
}
void _mat_mul_transpose_B(const f32* A, const f32* B, f32* out, 
						 u32 n_lines, u32 M, u32 N, u32 P, 
						 b32 cast_A, b32 cast_B, b32 cast_out,
						 b32 zero_out) {
		if (zero_out) memset(out, 0.0f, n_lines*M*P*sizeof(f32));

		for (u32 b = 0; b < n_lines; b++)
				for (u32 row = 0; row < M; row++)
						for (u32 col = 0; col < P; col++)
								for (u32 k=0; k < N; k++)
										out[(1-cast_out)*b*M*P+row*P+col] += 
												A[(1-cast_A)*b*M*N + row*N + k]
												* B[(1-cast_B)*b*N*P + col*N + k];
}
void _linear_layer(const f32* input, const f32* w, const f32* b, f32* out,
				   u32 n_lines, u32 in_features, u32 out_features) {
		// normally input is of shape (n_lines, in_features)
		// more complicated shapes will come with multi-head attention layers
		_mat_mul(w, input, out,
				 n_lines, out_features, in_features, 1,
				 1, 0, 0);
		_mat_add(b, out, out,
				 n_lines, out_features, 1,
				 1, 0, 0);
}
void _linear_layer_backward(const f32* input, const f32* w, const f32* downstream_grads,
						    f32* input_grad, f32* w_grad, f32* b_grad,
							u32 n_lines, u32 in_features, u32 out_features) {
		// input grad
		_mat_mul_transpose_A(w, downstream_grads, input_grad,
						     n_lines, in_features, out_features, 1,
							 1, 0, 0,
							 0);

		// weight grad
		_mat_mul_transpose_B(downstream_grads, input, w_grad,
						     n_lines, out_features, 1, in_features,
							 0, 0, 1,
							 0);

		// bias grad
		_mat_add(b_grad, downstream_grads, b_grad,
				 n_lines, out_features, 1,
				 1, 0, 1);
}
void _softmax(const f32* A, f32* out,
		      u32 n_lines, u32 last_dim) { // softmax computed along the last dimension
		// find maximums for softmax subtraction trick
		f32* maximums = new f32[n_lines];
		for (u32 line = 0; line < n_lines; line++) {
				maximums[line] = 0.0f; // enough to avoid exponential overflow
				for (u32 k = 0; k < last_dim; k++)
						maximums[line] = fmaxf(maximums[line], A[line*last_dim + k]);
		}
		
		// subtraction
		for (u32 line = 0; line < n_lines; line++)
				for (u32 k = 0; k < last_dim; k++)
						out[line*last_dim + k] = A[line*last_dim + k] - maximums[line];
		delete[] maximums;

		// exp
		_mat_exp(out, out, n_lines*last_dim);

		// sum along lines
		f32* sums = new f32[n_lines];
		for (u32 line = 0; line < n_lines; line++) {
				sums[line] = 0.0f;
				for (u32 k = 0; k < last_dim; k++)
						sums[line] += out[line*last_dim + k];
		}
		
		// division
		for (u32 line = 0; line < n_lines; line++)
				for (u32 k = 0; k < last_dim; k++)
						out[line*last_dim + k] /= sums[line];
		delete[] sums;
}
void _softmax_backward(const f32* downstream_grads, const f32* softmaxs, f32* input_grad,
				       u32 n_lines, u32 last_dim) {
		f32* dot_prods = new f32[n_lines];
		for (u32 line = 0; line < n_lines; line++) {
				dot_prods[line] = 0.0f;
				for (u32 k = 0; k < last_dim; k++)
						dot_prods[line] += 
								downstream_grads[line*last_dim + k]
								* softmaxs[line*last_dim + k];
		}

		for (u32 line = 0; line < n_lines; line++)
				for (u32 k = 0; k < last_dim; k++)
						input_grad[line*last_dim + k] += 
								softmaxs[line*last_dim + k] *
								(downstream_grads[line*last_dim + k] - dot_prods[line]);
		delete[] dot_prods;
}
void _mseloss(const f32* A, const f32* expected, f32* out,
		      u32 size) { // size is the product of the vector size and the n_lines (since anyway we average on both)
		*out = 0.0f;
		for (u32 i = 0; i < size; i++)
				*out += powf(A[i] - expected[i], 2);
		*out *= 1.0f / (f32)size;
}
void _mseloss_backward(const f32* A, const f32* expected, const f32* downstream_grad, 
				       f32* input_grad, 
					   u32 size) {
		f32 scaler = 1.0f / (f32)size;
		for (u32 i = 0; i < size; i++)
				input_grad[i] += 2.0f * (A[i] - expected[i]) * scaler * (*downstream_grad);
				// using += instead of = because in backward passes we accumulate gradients (in case of multiple contributions through, let's say, residual connections e.g.)
}
void _relu(const f32* A, f32* out,
		       u32 size) {
		for (u32 i = 0; i < size; i++)
				out[i] = fmaxf(0.0f, A[i]);
}
void _relu_backward(const f32* downstream_grads, const f32* input_data, f32* input_grad,
						u32 size) {
		for (u32 i=0; i < size; i++)
				input_grad[i] += downstream_grads[i] * (input_data[i] > 0);
}
void _mat_exp(const f32* A, f32* out, 
		     u32 size) {
		for (u32 i = 0; i < size; i++)
				out[i] = expf(A[i]);
}
void _mat_sum(const f32* A, f32* out,
		    u32 size) {
		*out = 0.0f;
		for (u32 i = 0; i < size; i++)
				*out += A[i];
}
void _mat_scale(const f32* A, f32* out, 
		       f32 scaler, 
			   u32 size) {
		for (u32 i = 0; i < size; i++)
				out[i] = A[i] * scaler;
}


void _adam_step(f32* parameters_grad, f32* parameters, f32* means, f32* squares, 
				f32 w_decay, f32 lr, f32 eps, f32 b1, f32 b2, f32 b1_pow, f32 b2_pow,
				u32 size) {
		f32 grad, mean_hat, square_hat;

		for (u32 j = 0; j < size; j++) {
				grad = parameters_grad[j];
				parameters[j] -= w_decay * lr * parameters[j];
				means[j] = b1 * means[j] + (1.0f-b1) * grad;
				squares[j] = b2 * squares[j] + (1.0f-b2) * powf(grad, 2);
				mean_hat = means[j] / (1.0f - b1_pow);
				square_hat = squares[j] / (1.0f - b2_pow);
						
				parameters[j] -= lr * mean_hat / (sqrtf(square_hat) + eps);
		}
}












