#include "functional.hpp"

// deal with devices, and shape mismatch, later 
void linear_layer(const Tensor* input, const Tensor* weight, const Tensor* bias, 
				  Tensor* out) {
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 in_features = input->shape[input_dim-1];
		u32 out_features = out->shape[input_dim-1];

		_linear_layer(input->data, weight->data, bias->data, out->data,
				      n_lines, in_features, out_features);
}
void linear_layer_backward(Tensor* input, Tensor* weight, Tensor* bias, const Tensor* out) {
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 in_features = input->shape[input_dim-1];
		u32 out_features = out->shape[input_dim-1];
		
		_linear_layer_backward(input->data, weight->data, out->grad,
						       input->grad, weight->grad, bias->grad,
							   n_lines, in_features, out_features);
}
void softmax(const Tensor* input, Tensor* out) {
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 last_dim = input->shape[input_dim-1];
		
		_softmax(input->data, out->data,
				 n_lines, last_dim);
}
void softmax_backward(Tensor* input, const Tensor* out) {
		u32 input_dim = (u32)input->shape.size();
		
		u32 n_lines = 1;
		for (u32 k = 0; k < input_dim-1; k++) n_lines *= input->shape[k];
		u32 last_dim = input->shape[input_dim-1];
		
		_softmax_backward(out->grad, out->data, input->grad,
						  n_lines, last_dim);
}
void mseloss(const Tensor* input, const Tensor* expected, Tensor* out) {
		u32 size = input->size;

		_mseloss(input->data, expected->data, out->data,
				 size);
}
void mseloss_backward(Tensor* input, const Tensor* expected, Tensor* out) {
		u32 size = input->size;

		_mseloss_backward(input->data, expected->data,
						 out->grad, input->grad,
						 size);
}
void relu(const Tensor* input, Tensor* output) {
		u32 size = input->size;

		_relu(input->data, output->data,
		      size);
}
void relu_backward(Tensor* input, const Tensor* output) {
		u32 size = input->size;

		_relu_backward(output->grad, input->data, input->grad,
				       size);
}


void adam_step(std::vector<Tensor*>& parameters, std::vector<std::unique_ptr<Tensor>>& means, std::vector<std::unique_ptr<Tensor>>& squares,
				f32 w_decay, f32 lr, f32 eps, f32 b1, f32 b2, f32 b1_pow, f32 b2_pow) {
		u32 n_tensors = (u32)parameters.size();

		for (u32 i = 0; i < n_tensors; i++)
				_adam_step(parameters[i]->grad, parameters[i]->data,
						   means[i].get()->data, squares[i].get()->data,
						   w_decay, lr, eps, b1, b2, b1_pow, b2_pow,
						   parameters[i]->size);
}


void xavier_uniform(Tensor* parameters, u32 input_features, u32 output_features) {
		f32 bound = sqrtf(6.0f)/sqrtf((f32)input_features + (f32)output_features);

		std::random_device rd{};
		std::mt19937 gen{rd()};
		std::uniform_real_distribution<f32> distribution(-bound, bound); 
		for (u64 i=0; i<parameters->size; i++)
				parameters->data[i] = distribution(gen);
}




