#include "functional.hpp"
#include "cuda_runtime.h"

// bpg stands for blocksPerGrid, to compute easily how much blocks are needed to complete
// everything
#define bpg(dim, threads) (((threads) + (dim) - 1) / (dim))


// MOVE THE ZERO OUT INTO THE FUNCTIONAL FUNCTIONS INSTEAD OF DOING IT IN THE KERNELS

// deal with devices, and shape mismatch, later 
void linear_layer(const Tensor* input, const Tensor* weight, const Tensor* bias, 
				  Tensor* out) {
		_check_same_device({input->dev, weight->dev, bias->dev, out->dev});
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 in_features = input->shape[input_dim-1];
		u32 out_features = out->shape[input_dim-1];

		if (input->dev == CPU)
				_linear_layer(input->data, weight->data, bias->data, out->data,
				      n_lines, in_features, out_features);
		else if (input->dev == CUDA) {
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
		else (std::cout << "Invalid device for computation" << endl;)
}
void linear_layer_backward(Tensor* input, Tensor* weight, Tensor* bias, const Tensor* out) {
		_check_same_device({input->dev, weight->dev, bias->dev, out->dev});
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 in_features = input->shape[input_dim-1];
		u32 out_features = out->shape[input_dim-1];
		
		if (input->dev == CPU)
				_linear_layer_backward(input->data, weight->data, out->grad,
						       input->grad, weight->grad, bias->grad,
							   n_lines, in_features, out_features);
		else if (input->dev == CUDA) {
				dim3 threadsPerBlock(1, 64, 16); // really bad to have 1 as threadPerBlock for the x axis since it does not allow memory coalescing... that will do it for now
				dim3 blocksPerGrid(1, bpg(in_features, 64), bpg(n_lines, 16));
				
				_mat_mul_transpose_A_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								weight->data, out->grad, input->grad,
								n_lines, in_features, out_features, 1,
								1, 0, 0,
								0);

				dim3 threadsPerBlock(16, 16, 4);
				dim3 blocksPerGrid(bpg(in_features, 16), bpg(out_features, 16), bpg(n_lines, 4));
				_mat_mul_transpose_B_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								out->grad, input->data, weight->grad,
								n_lines, out_features, 1, in_features,
								0, 0, 1,
								0);

				dim3 threadsPerBlock(1, 64, 16);
				dim3 blocksPerGrid(1, bpg(out_features, 64), bpg(n_lines, 16));
				_mat_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								bias->grad, out->grad, bias->grad,
								n_lines, out_features, 1,
								1, 0, 1);
				cudaDeviceSynchronize();
		}
		else (std::cout << "Invalid device for computation" << endl;)
}
void softmax(const Tensor* input, Tensor* out) {
		_check_same_device({input->dev, out->dev});
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 last_dim = input->shape[input_dim-1];
		
		if (input->dev == CPU)
				_softmax(input->data, out->data,
				 n_lines, last_dim);
		else if (input->dev == CUDA) {
				dim3 threadsPerBlock(64, 16);
				dim3 blocksPerGrid(bpg(last_dim, 64), bpg(n_lines, 16));
				_softmax_kernel<<<blocksPerGrid, threadsPerBlock, 2*threadsPerBlock*sizeof(f32)>>>(
								input->data, out->data,
								n_lines, last_dim);
				cudaDeviceSynchronize();
		}
		else (std::cout << "Invalid device for computation" << endl;)
}
void softmax_backward(Tensor* input, const Tensor* out) {
		_check_same_device({input->dev, out->dev});
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 last_dim = input->shape[input_dim-1];
		
		if (input->dev == CPU)
				_softmax_backward(out->grad, out->data, input->grad,
						  n_lines, last_dim);
		else if (input->dev == CUDA) {
				dim3 threadsPerBlock(64, 16);
				dim3 blocksPerGrid(bpg(last_dim, 64), bpg(n_lines, 16));
				_softmax_backward_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(f32)>>>(
								out->grad, out->data, input->grad,
								n_lines, last_dim);
				cudaDeviceSynchronize();
		}
		else (std::cout << "Invalid device for computation" << endl;)
}
void mseloss(const Tensor* input, const Tensor* expected, Tensor* out) {
		_check_same_device({input->dev, expected->dev, out->dev});
		u32 size = input->size;

		if (input->dev == CPU)
				_mseloss(input->data, expected->data, out->data,
				 size);
		else if (input->dev == CUDA) {
				i32 threadsPerBlock = 256;
				i32 blocksPerGrid = bpg(size, 256);
				_mseloss_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(f32)>>>(
								input->data, expected->data, out->data,
								size);
				cudaDeviceSynchronize();
		}
		else (std::cout << "Invalid device for computation" << endl;)
}
void mseloss_backward(Tensor* input, const Tensor* expected, Tensor* out) {
		_check_same_device({input->dev, expected->dev, out->dev});
		u32 size = input->size;
		
		if (input->dev == CPU)
				_mseloss_backward(input->data, expected->data,
						 out->grad, input->grad,
						 size);
		else if (input->dev == CUDA) {
				i32 threadsPerBlock = 256;
				i32 blocksPerGrid = bpg(size, 256);
				_mseloss_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								input->data, expected->data, out->grad, input->grad,
								size);
				cudaDeviceSynchronize();
		}
		else (std::cout << "Invalid device for computation" << endl;)
}
void relu(const Tensor* input, Tensor* output) {
		_check_same_device({input->dev, output->dev});
		u32 size = input->size;

		if (input->dev == CPU)
				_relu(input->data, output->data,
		      size);
		else if (input->dev == CUDA) {
				i32 threadsPerBlock = 256;
				i32 blocksPerGrid = bpg(size, 256);
				_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								input->data, out->data,
								size);
				cudaDeviceSynchronize();
		}
		else (std::cout << "Invalid device for computation" << endl;)
}
void relu_backward(Tensor* input, const Tensor* output) {
		_check_same_device({input->dev, output->dev});
		u32 size = input->size;
		
		if (input->dev == CPU)
				_relu_backward(output->grad, input->data, input->grad,
				       size);
		else if (input->dev == CUDA) {
				i32 threadsPerBlock = 256;
				i32 blocksPerGrid = bpg(size, 256);
				_relu_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(
								out->grad, input->data, input->grad,
								size);
				cudaDeviceSynchronize();
		}
		else (std::cout << "Invalid device for computation" << endl;)
}


void adam_step(std::vector<Tensor*>& parameters, std::vector<std::unique_ptr<Tensor>>& means, std::vector<std::unique_ptr<Tensor>>& squares,
				f32 w_decay, f32 lr, f32 eps, f32 b1, f32 b2, f32 b1_pow, f32 b2_pow) {
		for (u32 i = 0; i < parameters.size(); i++)
				_check_same_device({parameters[i]->dev, means[i].get()->dev, squares[i].get()->dev});
		u32 n_tensors = (u32)parameters.size();

		if (parameters[0]->dev == CPU) 
				for (u32 i = 0; i < n_tensors; i++)
						_adam_step(parameters[i]->grad, parameters[i]->data,
								means[i].get()->data, squares[i].get()->data,
								w_decay, lr, eps, b1, b2, b1_pow, b2_pow,
								parameters[i]->size);
		else if (parameters[0]->dev == CUDA) {
				u32 pool_size = 8;
				cudaStream_t streams[pool_size];
				for (u32 i=0; i<pool_size; i++)
						cudaStreamCreate(&streams[i]);

				// round robin for all parameter tensors
				for (u32 i=0; i<n_tensors; i++) {
						i32 stream_idx = i % pool_size;
						i32 threadsPerBlock = 256;
						i32 blocksPerGrid = bpg(parameters[i]->size, 256);
						_adam_step_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream_idx>>>(
								parameters[i]->grad, parameters[i]->data, means[i].get()->data, squares[i].get()->data,
								w_decay, lr, eps, b1, b2, b1_pow, b2_pow,
								parameters[i]->size);
				}

				cudaDeviceSynchronize();
		}
		else (std::cout << "Invalid device for computation" << endl;)
}


void xavier_uniform(Tensor* parameters, u32 input_features, u32 output_features) {
		f32 bound = sqrtf(6.0f)/sqrtf((f32)input_features + (f32)output_features);

		if (parameters->dev == CPU) {
				std::random_device rd{};
				std::mt19937 gen{rd()};
				std::uniform_real_distribution<f32> distribution(-bound, bound); 
				for (u64 i=0; i<parameters->size; i++)
						parameters->data[i] = distribution(gen);
		}
		// else cuda later
}


void zero_memory(Tensor* t) {
		if (t->dev == CPU)
				memset(t->data, 0.0f, t->size*sizeof(f32));
		else
				cudaMemset(t->data, 0.0f, t->size*sizeof(f32));
}

void zero_memory_grad(Tensor* t) {
		if (t->dev == CPU)
				memset(t->grad, 0.0f, t->size*sizeof(f32));
		else
				cudaMemset(t->grad, 0.0f, t->size*sizeof(f32));
}

