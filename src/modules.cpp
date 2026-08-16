#include "modules.hpp"
#include <iostream>
using namespace std;

Module::~Module() {
		if (this->activation)
				delete this->activation;
}

//////////////////
// Linear layer //
//////////////////

Linear::Linear(u32 input_features, u32 output_features) {
		this->weight = new Tensor({output_features, input_features});
		xavier_uniform(this->weight, input_features, output_features);

		this->bias = new Tensor({output_features, 1});
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
				this->activation = new Tensor(output_shape);
		}

		// zero grad (not done with parameters, for which it is done with optimizer)
		if (this->activation->requires_grad)
				memset(this->activation->grad, 0.0f, this->activation->size * sizeof(f32));
		
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
				this->activation = new Tensor(input->shape);
		if (this->activation->requires_grad)
				memset(this->activation->grad, 0.0f, this->activation->size * sizeof(f32));

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
				this->activation = new Tensor(input->shape);
		if (this->activation->requires_grad)
				memset(this->activation->grad, 0.0f, this->activation->size * sizeof(f32));

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
				this->activation = new Tensor({1});
		if (this->activation->requires_grad)
				memset(this->activation->grad, 0.0f, this->activation->size * sizeof(f32));

		mseloss(input, expected_output, this->activation);

		if (this->activation->grad_node) delete this->activation->grad_node;
		this->activation->grad_node = new MSELossNode(input, expected_output, this->activation);

		return this->activation;
}





