#include "cuda_maths.cuh"


__global__ void _mat_add_kernel(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, 
			  b32 cast_A, b32 cast_B, b32 cast_out) {
		u32 col = blockDim.x * blockIdx.x + threadIdx.x;
		u32 row = blockDim.y * blockIdx.y + threadIdx.y;
		u32 b = blockDim.z * blockIdx.z + threadIdx.z;

		u32 stride = b*M*N;
		u32 idx = row*N+col;

		if (col < N && row < M)
				out[(1-cast_out)*stride+idx] =
						A[(1-cast_A)*stride+idx] + B[(1-cast_B)*stride+idx];
}
__global__ void _mat_sub_kernel(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, 
			  b32 cast_A, b32 cast_B, b32 cast_out) {
		u32 col = blockDim.x * blockIdx.x + threadIdx.x;
		u32 row = blockDim.y * blockIdx.y + threadIdx.y;
		u32 b = blockDim.z * blockIdx.z + threadIdx.z;

		u32 stride = b*M*N;
		u32 idx = row*N+col;

		if (col < N && row < M)
				out[(1-cast_out)*stride+idx] =
						A[(1-cast_A)*stride+idx] - B[(1-cast_B)*stride+idx];
}
__global__ void _mat_mul_kernel(const f32* A, const f32* B, f32* out, 
		      u32 n_lines, u32 M, u32 N, u32 P, 
			  b32 cast_A, b32 cast_B, b32 cast_out, 
			  b32 zero_out = true) {
		// naive implementation for now
		u32 col = blockDim.x * blockIdx.x + threadIdx.x;
		u32 row = blockDim.y * blockIdx.y + threadIdx.y;
		u32 b = blockDim.z * blockIdx.z + threadIdx.z;

		u32 stride_out = b*M*P;
		u32 stride_A = b*M*N;
		u32 stride_B = b*N*P;
		u32 idx = row*P+col;
		
		if (col < P && row < M) {
				out[(1-cast_out)*stride_out+idx] *= zero_out;
				f32 sum = 0.0f;
				for (u32 k = 0; k < N; k++)
						sum +=
								A[(1-cast_A)*stride_A+row*N+k]
								* B[(1-cast_B)*stride_B+k*P+col];
				out[(1-cast_out)*stride_out+idx] += sum; 
		}
}
__global__ void _mat_mul_transpose_A_kernel(const f32* A, const f32* B, f32* out, 
				         u32 n_lines, u32 M, u32 N, u32 P, 
						 b32 cast_A, b32 cast_B, b32 cast_out,
						 b32 zero_out = true) {
		u32 col = blockDim.x * blockIdx.x + threadIdx.x;
		u32 row = blockDim.y * blockIdx.y + threadIdx.y;
		u32 b = blockDim.z * blockIdx.z + threadIdx.z;

		u32 stride_out = b*M*P;
		u32 stride_A = b*M*N;
		u32 stride_B = b*N*P;
		u32 idx = row*P+col;
		
		if (col < P && row < M) {
				out[(1-cast_out)*stride_out+idx] *= zero_out;
				f32 sum = 0.0f;
				for (u32 k = 0; k < N; k++)
						sum +=
								A[(1-cast_A)*stride_A+k*M+row]
								* B[(1-cast_B)*stride_B+k*P+col];
				out[(1-cast_out)*stride_out+idx] += sum;
		}
}
__global__ void _mat_mul_transpose_B_kernel(const f32* A, const f32* B, f32* out, 
						 u32 n_lines, u32 M, u32 N, u32 P, 
						 b32 cast_A, b32 cast_B, b32 cast_out,
						 b32 zero_out = true) {
		u32 col = blockDim.x * blockIdx.x + threadIdx.x;
		u32 row = blockDim.y * blockIdx.y + threadIdx.y;
		u32 b = blockDim.z * blockIdx.z + threadIdx.z;

		u32 stride_out = b*M*P;
		u32 stride_A = b*M*N;
		u32 stride_B = b*N*P;
		u32 idx = row*P+col;
		
		if (col < P && row < M) {
				out[(1-cast_out)*stride_out+idx] *= zero_out;
				f32 sum = 0.0f;
				for (u32 k = 0; k < N; k++)
						sum +=
								A[(1-cast_A)*stride_A+row*N+k]
								* B[(1-cast_B)*stride_B+col*N+k];
				out[(1-cast_out)*stride_out+idx] += sum;
		}
}
__global__ void _linear_layer_kernel(const f32* input, const f32* w, const f32* b, f32* out,
				   u32 n_lines, u32 in_features, u32 out_features) {
		// not implemented for now
		// we just call mat_mul and mat_add, we will change that later
}
__global__ void _linear_layer_backward_kernel(const f32* input, const f32* w, const f32* downstream_grads,
						    f32* input_grad, f32* w_grad, f32* b_grad,
							u32 n_lines, u32 in_features, u32 out_features) {
		// not implemented for now
		// we call the mat mul transposes, that will change later
}
__global__ void _softmax_kernel(const f32* A, f32* out,
		      u32 n_lines, u32 last_dim) {
		// to run with a single block for the last dim so that i don't have to do atomic adds or atomic maxs on global memory
		// instead i use grid stride to stay in the same block and have the same shared memory
		// so gridDim.x = 1, and thus blockIdx.x = 0
		// running with 1024 threads on the x axis (so 1 on the y axis) to counterbalance the issue
		u32 last_idx = threadIdx.x;
		u32 line = blockDim.y * blockIdx.y + threadIdx.y;
		u32 grid_stride = blockDim.x;

		extern __shared__ float maximums[];
		extern __shared__ float sums[];

		if (last_idx < last_dim && line < n_lines) {
				maximums[last_idx] = A[line*last_dim + last_idx];
				for (u32 i=last_idx+grid_stride; i<last_dim; i+=grid_stride)
						maximums[last_idx] = fmaxf(maximums[last_idx], A[line*last_dim + i]);
		}
		else {
				maximums[last_idx] = 0.0f;
		}

		u32 binary_divide = last_dim;
		i32 add_last = binary_divide%2;
		while (binary_divide > 1) {
				if (last_idx == 0 && add_last && line < n_lines) 
						maximums[0] = fmaxf(maximums[0], maximums[binary_divide-1]);
				binary_divide /= 2;
				if (last_idx < binary_divide && line < n_lines)
						maximums[last_idx] = fmaxf(maximums[last_idx], maximums[last_idx+binary_divide]);

				__syncthreads();
				add_last = binary_divide%2;
		}

		// the max value of each line is now stored in maximums[0]

		if (last_idx < last_dim && line < n_lines)
				out[line*last_dim + last_idx] = __expf(A[line*last_dim + last_idx] - maximums[0]);

		if (last_idx < last_dim && line < n_lines) {
				sums[last_idx] = out[line*last_dim + last_idx];
				for (u32 i=last_idx+grid_stride; i<last_dim; i+=grid_stride)
						sums[last_idx] += out[line*last_dim + i];
		}
		else {
				sums[last_idx] = 0.0f;
		}

		u32 binary_divide = last_dim;
		i32 add_last = binary_divide%2;
		while (binary_divide > 1) {
				if (last_idx == 0 && add_last && line < n_lines) 
						sums[0] += sums[binary_divide-1];
				__syncthreads();
				binary_divide /= 2;
				if (last_idx < binary_divide && line < n_lines)
						sums[last_idx] += sums[last_idx+binary_divide];
				add_last = binary_divide % 2;

				__syncthreads();
		}

		// sum of the line stored in sums[0]

		if (last_idx < last_dim && line < n_lines)
				out[line*last_dim + last_idx] /= sums[0];
}
__global__ void _softmax_backward_kernel(const f32* downstream_grads, const f32* softmaxs, f32* input_grad,
				       u32 n_lines, u32 last_dim) {
		// kernel called with 1024 threadsPerBlock on the x axis and a single block, 1 threadPerBlock on the y axis
		u32 last_idx = threadIdx.x;
		u32 line = blockDim.y * blockIdx.y + threadIdx.y; // blockIdx.y = 0 but i write it like that for clarity
		u32 grid_stride = blockDim.x * gridDim.x; // gridDim.x = 1 and blockDim.x = 1024 so it's just 1024

		extern __shared__ f32 scalars[]; // to compute the dot product of downstream_grads and softmaxs

		if (last_idx < last_dim && line < n_lines) {
				scalars[last_idx] = downstream_grads[line*last_dim + last_idx] * softmaxs[line*last_dim + last_idx];
				for (u32 i=last_idx+grid_stride; i<last_dim; i+=grid_stride)
						scalars[last_idx] += downstream_grads[line*last_dim+i] * softmaxs[line*last_dim + i];
		}
		else {
				scalars[last_idx] = 0.0f;
		}

		u32 binary_divide = last_dim;
		i32 add_last = binary_divide%2;
		while (binary_divide > 1) {
				if (add_last && last_idx == 0)
						scalars[0] += scalars[binary_divide - 1];
				__syncthreads();
				binary_divide /= 2;
				add_last = binary_divide % 2;
				if (last_idx < binary_divide && line < n_lines)
						scalars[last_idx] += scalars[last_idx + binary_divide];
				__syncthreads();
		}

		// scalars[0] now contains the dot product

		if (last_idx < last_dim && line < n_lines)
				for (u32 i=last_idx; i<last_dim; i+=grid_stride)
						input_grad[line*last_dim + i] = softmaxs[line*last_dim + i] * (downstream_grads[line*last_dim + i] - scalars[0]);
}
__global__ void _mseloss_kernel(const f32* A, const f32* expected, f32* out,
		      u32 size) {
		u32 idx = blockDim.x * blockIdx.x + threadIdx.x;
		u32 tid = threadIdx.x;
		u32 grid_stride = blockDim.x * gridDim.x;

		extern __shared__ float sums[];

		f32 scalar = 1.0f / (f32)size; // to aovid having to scale with one single thread at the end because it would cause race conditions
								       // across different blocks (we cant synchronize different blocks)

		if (idx < size) {
				f32 diff = A[idx] - expected[idx];
				sums[tid] = diff*diff*scalar; // float exponent with __powf can't work for negative values (because it uses log) so we just multiply by itself
				for (u32 i=idx+grid_stride; i<size; i+=grid_stride)
						diff = A[i] - expected[i];
						sums[tid] += diff*diff*scalar;
		}
		else {
				sums[tid] = 0.0f; // flat out the garbage to zero to not bother adding it afterwards, the threads will run anyway
		}

		u32 binary_divide = blockDim.x;
		i32 add_last = binary_divide%2;
		while (binary_divide > 1) {
				if (tid == 0 && add_last) 
						sums[0] += sums[binary_divide-1];
				binary_divide /= 2;
				if (tid < binary_divide)
						sums[tid] += sums[tid+binary_divide];

				__syncthreads();
				add_last = binary_divide%2;
		}

		if (tid == 0) atomicAdd(&out[0], sums[0]);

}
__global__ void _mseloss_backward_kernel(const f32* A, const f32* expected, 
				       const f32* downstream_grad, f32* input_grad,
					   u32 size) {
		u32 idx = blockDim.x * blockIdx.x + threadIdx.x;
		
		f32 scalar = 1.0f / (f32)size;

		if (idx < size)
				input_grad[idx] = downstream_grad[idx] * 2.0f * scalar * (A[idx] - expected[idx]);
}


__global__ void _relu_kernel(const f32* A, f32* out,
		       u32 size) {
		u32 idx = blockDim.x * blockIdx.x + threadIdx.x;

		if (idx < size)
				out[idx] = fmaxf(A[idx], 0.0f);
}
__global__ void _relu_backward_kernel(const f32* downstream_grads, const f32* input_data, f32* input_grad,
						u32 size) {
		u32 idx = blockDim.x * blockIdx.x + threadIdx.x;

		if (idx < size)
				input_grad[idx] = downstream_grads[idx] * (input_data[idx] > 0.0f);
}
__global__ void _mat_exp_kernel(const f32* A, f32* out, 
		     u32 size) {}
__global__ void _mat_sum_kernel(const f32* A, f32* out,
		    u32 size) {}
__global__ void _mat_scale_kernel(const f32* A, f32* out, 
		       f32 scalar, 
			   u32 size) {}


__global__ void _adam_step_kernel(f32* parameters_grad, f32* parameters, f32* means, f32* squares, 
				f32 w_decay, f32 lr, f32 eps, f32 b1, f32 b2, f32 b1_pow, f32 b2_pow,
				u32 size) {

}











