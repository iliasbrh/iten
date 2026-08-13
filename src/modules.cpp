#include "modules.hpp"


//////////////////
// Linear layer //
//////////////////

Linear::Linear(const std::vector<u64> &activation_shape, u64 input_features, u64 output_features) {
		this->activation = new Tensor(activation_shape);
		this->weight = new Tensor({output_features, input_features});
		init_parameters(this->weight, input_features);
		this->bias = new Tensor({output_features, 1});
		init_parameters(this->bias, input_features);

		this->parameters.push_back(this->weight);
		this->parameters.push_back(this->bias);
}
Linear::~Linear() {
		delete this->weight;
		delete this->bias;
}

Tensor* Linear::forward(Tensor* input) {
		if (this->activation->requires_grad)
				memset(this->activation->grad, 0.0f, this->activation->size * sizeof(f32));

		mat_mul(this->weight->data, input->data, this->activation->data, this->weight->shape[0], this->weight->shape[1], input->shape[1], true);
		mat_add(this->activation->data, this->bias->data, this->activation->data, this->activation->shape[0], this->activation->shape[1]);

		// node creation
		if (this->weight->grad_node) delete this->weight->grad_node;
		if (this->bias->grad_node) delete this->bias->grad_node;
		if (this->activation->grad_node) delete this->activation->grad_node;

		this->weight->grad_node = new StartNode(this->weight);
		this->bias->grad_node = new StartNode(this->bias);
		this->activation->grad_node = new LinearNode(input, this->weight, this->bias, this->activation);

		return this->activation;
}


////////////////
// ReLU Layer //
////////////////

ReLU::ReLU(const std::vector<u64>& activation_shape) {
		this->activation = new Tensor(activation_shape);
}

Tensor* ReLU::forward(Tensor* input) {
		if (this->activation->requires_grad)
				memset(this->activation->grad, 0.0f, this->activation->size * sizeof(f32));

		// replace with a utils function
		for (u64 i=0; i<input->size; i++)
				this->activation->data[i] = fmaxf(input->data[i], 0.0f);

		if (this->activation->grad_node) delete this->activation->grad_node;
		this->activation->grad_node = new ReLUNode(input, this->activation);

		return this->activation;
}


///////////////////
// Softmax Layer //
///////////////////

Softmax::Softmax(const std::vector<u64> &activation_shape, f32 temperature) {
		this->activation = new Tensor(activation_shape);
		this->temp = temperature;
}

Tensor* Softmax::forward(Tensor* input) {
		// take care of the temperature later
		if (this->activation->requires_grad)
				memset(this->activation->grad, 0.0f, this->activation->size * sizeof(f32));

		// implementing the subtraction to avoid exponential overflow
		f32 max_val = input->data[0];
		for (u64 i=1; i<input->size; i++)
				max_val = fmaxf(max_val, input->data[i]);
		for (u64 i=0; i<input->size; i++)
				this->activation->data[i] = input->data[i] - max_val;

		mat_exp(this->activation->data, this->activation->data, this->activation->size);
		f32 divider_ctx = mat_sum(this->activation->data, this->activation->size);
		mat_scale(this->activation->data, this->activation->data, 1.0f/divider_ctx, this->activation->size);

		if (this->activation->grad_node) delete this->activation->grad_node;
		this->activation->grad_node = new SoftmaxNode(input, this->activation, divider_ctx, this->temp);

		return this->activation;
}


/////////////
// MSELoss //
/////////////

MSELoss::MSELoss(const std::vector<u64> &activation_shape) {
		this->activation = new Tensor(activation_shape);
}

Tensor* MSELoss::forward(Tensor* input, Tensor* expected_output) {
		if (this->activation->requires_grad)
				memset(this->activation->grad, 0.0f, this->activation->size * sizeof(f32));
		// do it with utils function later
		this->activation->data[0] = 0.0f;
		for (u64 i=0; i<input->size; i++)
				this->activation->data[0] += powf((input->data[i] - expected_output->data[i]), 2) / (f32)input->size;

		if (this->activation->grad_node) delete this->activation->grad_node;
		this->activation->grad_node = new MSELossNode(input, expected_output, this->activation);

		return this->activation;
}





