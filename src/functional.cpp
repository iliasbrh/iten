#include "functional.hpp"

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
		// else here call kernel	
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
		// else here call kernel
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
		// else cuda
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
		// else cuda
}
void mseloss(const Tensor* input, const Tensor* expected, Tensor* out) {
		_check_same_device({input->dev, expected->dev, out->dev});
		u32 size = input->size;

		if (input->dev == CPU)
				_mseloss(input->data, expected->data, out->data,
				 size);
		// else cuda
}
void mseloss_backward(Tensor* input, const Tensor* expected, Tensor* out) {
		_check_same_device({input->dev, expected->dev, out->dev});
		u32 size = input->size;
		
		if (input->dev == CPU)
				_mseloss_backward(input->data, expected->data,
						 out->grad, input->grad,
						 size);
		// else cuda
}
void relu(const Tensor* input, Tensor* output) {
		_check_same_device({input->dev, output->dev});
		u32 size = input->size;

		if (input->dev == CPU)
				_relu(input->data, output->data,
		      size);
		// else cuda
}
void relu_backward(Tensor* input, const Tensor* output) {
		_check_same_device({input->dev, output->dev});
		u32 size = input->size;
		
		if (input->dev == CPU)
				_relu_backward(output->grad, input->data, input->grad,
				       size);
		// else cuda
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
		// else cuda
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
		// else cuda
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

