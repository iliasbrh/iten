#include "modules.hpp"

Module::~Module() {
		if (this->activation)
				delete this->activation;
}

void Module::to(device_t device) {
		for (Tensor* param : this->parameters)
				param->to(device);
		for (Module* sub_module : this->sub_layers)
				sub_module->to(device);
}

//////////////////
// Linear layer //
//////////////////

Linear::Linear(u32 input_features, u32 output_features, device_t target_device) {
		this->weight = new Tensor({output_features, input_features}, target_device);
		xavier_uniform(this->weight, input_features, output_features);

		this->bias = new Tensor({output_features, 1}, target_device);
		xavier_uniform(this->bias, input_features, output_features);

		this->parameters = {this->weight, this->bias};

		this->activation = nullptr;
}
Linear::~Linear() {
		delete this->weight;
		delete this->bias;
}

Tensor* Linear::forward(Tensor* input) {
		// memory allocation
		if (!this->activation) {
				std::vector<u32> output_shape = input->shape;
				output_shape[output_shape.size() - 1] = this->weight->shape[0];
				this->activation = new Tensor(output_shape, input->device);
		}

		// zero grad (not done with parameters, for which it is done with optimizer)
		if (this->activation->requires_grad)
				zero_memory_grad(this->activation);

		// computation

		linear_layer(input, this->weight, this->bias, this->activation);

		// building graph 
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

ReLU::ReLU() {
		this->activation = nullptr;
}

Tensor* ReLU::forward(Tensor* input) {
		if (!this->activation)
				this->activation = new Tensor(input->shape, input->device);
		if (this->activation->requires_grad)
				zero_memory_grad(this->activation);

		relu(input, this->activation);

		if (this->activation->grad_node) delete this->activation->grad_node;
		this->activation->grad_node = new ReLUNode(input, this->activation);

		return this->activation;
}


///////////////////
// Softmax Layer //
///////////////////

Softmax::Softmax(f32 temperature) {
		this->activation = nullptr;
		this->temp = temperature;
}

Tensor* Softmax::forward(Tensor* input) {
		if (!this->activation)
				this->activation = new Tensor(input->shape, input->device);
		if (this->activation->requires_grad)
				zero_memory_grad(this->activation);

		softmax(input, this->activation);

		// building graph
		if (this->activation->grad_node) delete this->activation->grad_node;
		this->activation->grad_node = new SoftmaxNode(input, this->activation, this->temp);

		return this->activation;
}


/////////////
// MSELoss //
/////////////

MSELoss::MSELoss() {
		this->activation = nullptr;
}

Tensor* MSELoss::forward(Tensor* input, Tensor* expected_output) {
		if (!this->activation)
				this->activation = new Tensor({1}, input->device);
		if (this->activation->requires_grad)
				zero_memory_grad(this->activation);

		mseloss(input, expected_output, this->activation);

		if (this->activation->grad_node) delete this->activation->grad_node;
		this->activation->grad_node = new MSELossNode(input, expected_output, this->activation);

		return this->activation;
}


//////////////////////
// CrossEntropyLoss //
//////////////////////

CrossEntropyLoss::CrossEntropyLoss() {
		this->activation = nullptr;
}

Tensor* CrossEntropyLoss::forward(Tensor* input, Tensor* expected_output) {
		if (!this->activation)
				this->activation = new Tensor({1}, input->device);
		if (this->activation->requires_grad)
				zero_memory_grad(this->activation);

		crossentropyloss(input, expected_output, this->activation);

		if (this->activation->grad_node) delete this->activation->grad_node;
		this->activation->grad_node = new CrossEntropyLossNode(input, expected_output, this->activation);

		return this->activation;
}





