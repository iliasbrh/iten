#include "functional.cuh"
#include "cuda_runtime.h"

// bpg stands for blocksPerGrid, to compute easily how much blocks are needed to complete
// everything
#define bpg(dim, threads) (((threads) + (dim) - 1) / (threads))


// MOVE THE ZERO OUT INTO THE FUNCTIONAL FUNCTIONS INSTEAD OF DOING IT IN THE KERNELS

// deal with devices, and shape mismatch, later 
void linear_layer(const Tensor* input, const Tensor* weight, const Tensor* bias, 
				  Tensor* out) {
		_check_same_device({input->device, weight->device, bias->device, out->device});
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 in_features = input->shape[input_dim-1];
		u32 out_features = out->shape[input_dim-1];

		if (input->device == CPU)
				_linear_layer(input->data, weight->data, bias->data, out->data,
				      n_lines, in_features, out_features);
		else if (input->device == CUDA) {
				// input and output are column vectors so 1 single column
				dim3 threadsPerBlock(1, 64, 16);
				dim3 blocksPerGrid(1, bpg(out_features, 64), bpg(n_lines, 16));

				
				_mat_mul_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								weight->data, input->data, out->data,
								n_lines, out_features, in_features, 1,
								1, 0, 0,
								1);

				_mat_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								out->data, bias->data, out->data,
								n_lines, out_features, 1,
								0, 1, 0);
				cudaDeviceSynchronize();
		}
		else std::cout << "Invalid device for computation" << std::endl;
}
void linear_layer_backward(Tensor* input, Tensor* weight, Tensor* bias, const Tensor* out) {
		_check_same_device({input->device, weight->device, bias->device, out->device});
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 in_features = input->shape[input_dim-1];
		u32 out_features = out->shape[input_dim-1];
		
		if (input->device == CPU)
				_linear_layer_backward(input->data, weight->data, out->grad,
						       input->grad, weight->grad, bias->grad,
							   n_lines, in_features, out_features);
		else if (input->device == CUDA) {
				{
				dim3 threadsPerBlock(1, 64, 16); // really bad to have 1 as threadPerBlock for the x axis since it does not allow memory coalescing... that will do it for now
				dim3 blocksPerGrid(1, bpg(in_features, 64), bpg(n_lines, 16));
				
				_mat_mul_transpose_A_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								weight->data, out->grad, input->grad,
								n_lines, in_features, out_features, 1,
								1, 0, 0,
								0);
				}
				
				{
				dim3 threadsPerBlock(16, 16, 4);
				dim3 blocksPerGrid(bpg(in_features, 16), bpg(out_features, 16), bpg(n_lines, 4));
				_mat_mul_transpose_B_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								out->grad, input->data, weight->grad,
								n_lines, out_features, 1, in_features,
								0, 0, 1,
								0);
				}

				{
				dim3 threadsPerBlock(1, 64, 16);
				dim3 blocksPerGrid(1, bpg(out_features, 64), bpg(n_lines, 16));
				_mat_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								bias->grad, out->grad, bias->grad,
								n_lines, out_features, 1,
								1, 0, 1);
				cudaDeviceSynchronize();
				}
		}
		else std::cout << "Invalid device for computation" << std::endl;
}
void softmax(const Tensor* input, Tensor* out) {
		_check_same_device({input->device, out->device});
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 last_dim = input->shape[input_dim-1];
		
		if (input->device == CPU)
				_softmax(input->data, out->data,
				 n_lines, last_dim);
		else if (input->device == CUDA) {
				dim3 threadsPerBlock(1024, 1);
				dim3 blocksPerGrid(1, n_lines);
				_softmax_kernel<<<blocksPerGrid, threadsPerBlock, 2*threadsPerBlock.x*sizeof(f32)>>>(
								input->data, out->data,
								n_lines, last_dim);
				cudaDeviceSynchronize();
		}
		else std::cout << "Invalid device for computation" << std::endl;
}
void softmax_backward(Tensor* input, const Tensor* out) {
		_check_same_device({input->device, out->device});
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 last_dim = input->shape[input_dim-1];
		
		if (input->device == CPU)
				_softmax_backward(out->grad, out->data, input->grad,
						  n_lines, last_dim);
		else if (input->device == CUDA) {
				dim3 threadsPerBlock(1024, 1);
				dim3 blocksPerGrid(1, n_lines);
				size_t sharedMemBytes = threadsPerBlock.x * sizeof(f32);
				_softmax_backward_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(
								out->grad, out->data, input->grad,
								n_lines, last_dim);
				cudaDeviceSynchronize();
		}
		else std::cout << "Invalid device for computation" << std::endl;
}
void mseloss(const Tensor* input, const Tensor* expected, Tensor* out) {
		_check_same_device({input->device, expected->device, out->device});
		u32 size = input->size;

		if (input->device == CPU)
				_mseloss(input->data, expected->data, out->data,
				 size);
		else if (input->device == CUDA) {
				i32 threadsPerBlock = 256;
				i32 blocksPerGrid = bpg(size, 256);
				size_t sharedMemBytes = threadsPerBlock * sizeof(f32);

				cudaMemset(out->data, 0, sizeof(f32));

				_mseloss_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(
								input->data, expected->data, out->data,
								size);
				cudaDeviceSynchronize();
		}
		else std::cout << "Invalid device for computation" << std::endl;
}
void mseloss_backward(Tensor* input, const Tensor* expected, Tensor* out) {
		_check_same_device({input->device, expected->device, out->device});
		u32 size = input->size;
		
		if (input->device == CPU)
				_mseloss_backward(input->data, expected->data,
						 out->grad, input->grad,
						 size);
		else if (input->device == CUDA) {
				i32 threadsPerBlock = 256;
				i32 blocksPerGrid = bpg(size, 256);
				_mseloss_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								input->data, expected->data, out->grad, input->grad,
								size);
				cudaDeviceSynchronize();
		}
		else std::cout << "Invalid device for computation" << std::endl;
}
void crossentropyloss(const Tensor* input, const Tensor* expected, Tensor* out) {
		_check_same_device({input->device, expected->device, out->device});
		u32 size = input->size;

		if (input->device == CPU)
				_crossentropyloss(input->data, expected->data, out->data,
				 size);
		else if (input->device == CUDA) {
				i32 threadsPerBlock = 256;
				i32 blocksPerGrid = bpg(size, 256);
				size_t sharedMemBytes = threadsPerBlock * sizeof(f32);

				cudaMemset(out->data, 0, sizeof(f32));

				_crossentropyloss_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(
								input->data, expected->data, out->data,
								size);
				cudaDeviceSynchronize();
		}
		else std::cout << "Invalid device for computation" << std::endl;
}
void crossentropyloss_backward(Tensor* input, const Tensor* expected, Tensor* out) {
		_check_same_device({input->device, expected->device, out->device});
		u32 size = input->size;
		
		if (input->device == CPU)
				_crossentropyloss_backward(input->data, expected->data,
						 out->grad, input->grad,
						 size);
		else if (input->device == CUDA) {
				i32 threadsPerBlock = 256;
				i32 blocksPerGrid = bpg(size, 256);
				_crossentropyloss_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								input->data, expected->data, out->grad, input->grad,
								size);
				cudaDeviceSynchronize();
		}
		else std::cout << "Invalid device for computation" << std::endl;
}
void relu(const Tensor* input, Tensor* output) {
		_check_same_device({input->device, output->device});
		u32 size = input->size;

		if (input->device == CPU)
				_relu(input->data, output->data,
		      size);
		else if (input->device == CUDA) {
				i32 threadsPerBlock = 256;
				i32 blocksPerGrid = bpg(size, 256);
				_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								input->data, output->data,
								size);
				cudaDeviceSynchronize();
		}
		else std::cout << "Invalid device for computation" << std::endl;
}
void relu_backward(Tensor* input, const Tensor* output) {
		_check_same_device({input->device, output->device});
		u32 size = input->size;
		
		if (input->device == CPU)
				_relu_backward(output->grad, input->data, input->grad,
				       size);
		else if (input->device == CUDA) {
				i32 threadsPerBlock = 256;
				i32 blocksPerGrid = bpg(size, 256);
				_relu_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								output->grad, input->data, input->grad,
								size);
				cudaDeviceSynchronize();
		}
		else std::cout << "Invalid device for computation" << std::endl;
}


void adam_step(std::vector<Tensor*>& parameters, std::vector<std::unique_ptr<Tensor>>& means, std::vector<std::unique_ptr<Tensor>>& squares,
				f32 w_decay, f32 lr, f32 eps, f32 b1, f32 b2, f32 b1_pow, f32 b2_pow) {
		for (u32 i = 0; i < parameters.size(); i++)
				_check_same_device({parameters[i]->device, means[i].get()->device, squares[i].get()->device});
		u32 n_tensors = (u32)parameters.size();

		if (parameters[0]->device == CPU) 
				for (u32 i = 0; i < n_tensors; i++)
						_adam_step(parameters[i]->grad, parameters[i]->data,
								means[i].get()->data, squares[i].get()->data,
								w_decay, lr, eps, b1, b2, b1_pow, b2_pow,
								parameters[i]->size);
		else if (parameters[0]->device == CUDA) {
				u32 pool_size = 8;
				cudaStream_t streams[pool_size];
				for (u32 i=0; i<pool_size; i++)
						cudaStreamCreate(&streams[i]);

				cudaDeviceSynchronize();
				// round robin for all parameter tensors
				for (u32 i=0; i<n_tensors; i++) {
						i32 stream_idx = i % pool_size;
						i32 threadsPerBlock = 256;
						i32 blocksPerGrid = bpg(parameters[i]->size, 256);
						_adam_step_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[stream_idx]>>>(
								parameters[i]->grad, parameters[i]->data, means[i].get()->data, squares[i].get()->data,
								w_decay, lr, eps, b1, b2, b1_pow, b2_pow,
								parameters[i]->size);
				}
				
				cudaDeviceSynchronize();

				for (u32 i=0; i<pool_size; i++)
						cudaStreamDestroy(streams[i]);
		}
		else std::cout << "Invalid device for computation" << std::endl;
}


void xavier_uniform(Tensor* parameters, u32 input_features, u32 output_features) {
		f32 bound = sqrtf(6.0f)/sqrtf((f32)input_features + (f32)output_features);

		if (parameters->device == CPU) {
				std::random_device rd{};
				std::mt19937 gen{rd()};
				std::uniform_real_distribution<f32> distribution(-bound, bound); 
				for (u64 i=0; i<parameters->size; i++)
						parameters->data[i] = distribution(gen);
		}
		// else cuda later
}


void zero_memory(Tensor* t) {
		if (t->device == CPU)
				memset(t->data, 0, t->size*sizeof(f32));
		else if (t->device == CUDA)
				cudaMemset(t->data, 0, t->size*sizeof(f32));
				cudaDeviceSynchronize();
}

void zero_memory_grad(Tensor* t) {
		if (t->device == CPU)
				memset(t->grad, 0, t->size*sizeof(f32));
		else if (t->device == CUDA) {
				size_t mem_size = (size_t)t->size * sizeof(f32);
				cudaMemset(t->grad, 0, mem_size);
				cudaDeviceSynchronize();
		}
		else
				std::cout << "Invalid device for computation" << std::endl;
}

void grad_fill_ones(Tensor* t) {
		if (t->device == CPU) {
				for (u64 i=0; i<t->size; i++)
						t->grad[i] = 1.0f;
		}
		if (t->device == CUDA) {
				i32 threads = 256;
				i32 blocks = bpg(t->size, threads);
				_set_value_kernel<<<blocks, threads>>>(t->grad, 1.0f, t->size);
				cudaDeviceSynchronize();
		}
}

