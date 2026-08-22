#include "tensor.hpp"

/////////////
// Tensors //
/////////////

Tensor::Tensor(std::vector<u32> tensor_shape, device_t target_device, b32 is_input, b32 req_grad) {
		this->shape = tensor_shape;
		this->size = multiply_vector(tensor_shape);
		this->requires_grad = req_grad;
		
		this->data = _alloc(this->size*sizeof(f32), target_device);

		if (this->requires_grad) 
		{
				this->grad = _alloc(this->size*sizeof(f32), target_device);
		}
		else {this->grad = nullptr;}

		if (is_input)
			   this->grad_node = new StartNode(this);
		else {this->grad_node = nullptr;}

		this->device = target_device;
}

Tensor::~Tensor() {
		_free(this->data, this->device);
		if (this->grad)
				_free(this->grad, this->device);

		if (this->grad_node)
				delete this->grad_node;
}

void Tensor::to(device_t dev) {
		if (this->device == CPU && dev == CUDA) {
				this->device = CUDA;
				this->data = _move_to(this->data, this->size, CUDA);
				if (this->requires_grad)
					this->grad = _move_to(this->grad, this->size, CUDA);
		}
		if (this->device == CUDA && dev == CPU) {
				this->device = CPU;
				this->data = _move_to(this->data, this->size, CPU);
				if (this->requires_grad)
					this->grad = _move_to(this->grad, this->size, CPU);
		}
}

void Tensor::backward() { // to reimplement for a proper reverse topological graph
		grad_fill_ones(this);
		
		// building bfs backpass graph
		std::vector<Node*> bfs_visited;
		std::queue<Node*> q;
		q.push(this->grad_node);
		
		Node* running_node;
		while (!q.empty()) {
				running_node = q.front();
				q.pop();
				bfs_visited.push_back(running_node);
				for (auto parent : running_node->parents) {
						q.push(parent);
				}
		}

		// applying gradients
		for (auto node : bfs_visited) {
				node->apply();
		}
}


/////////////////////////
// Computational Nodes //
/////////////////////////

StartNode::StartNode(Tensor* tensor_node) {
		this->tensor = tensor_node;
}

void StartNode::apply() {}


LinearNode::LinearNode(Tensor* input, Tensor* weight, Tensor* bias, Tensor* tensor_node) {
		this->tensor = tensor_node;
		this->parents = {input->grad_node, weight->grad_node, bias->grad_node};

		this->weight_ctx = weight;
		this->bias_ctx = bias;
		this->input_ctx = input;
}

void LinearNode::apply() {
		linear_layer_backward(this->input_ctx, this->weight_ctx, this->bias_ctx,
						      this->tensor);
}

ReLUNode::ReLUNode(Tensor* input, Tensor* tensor_node) {
		this->tensor = tensor_node;
		this->parents = {input->grad_node};

		this->input_ctx = input;
}

void ReLUNode::apply() {
		relu_backward(this->input_ctx, this->tensor);
}


SoftmaxNode::SoftmaxNode(Tensor* input, Tensor* tensor_node, f32 temp) {
		this->tensor = tensor_node;
		this->parents = {input->grad_node};

		this->input_ctx = input;
		this->temperature = temp;
}

void SoftmaxNode::apply() {
		// do it temperature wise in the future
		// in fact, temperature is never used during training so adding it to backward passes would be useless. the only time where we set it to something other than 1 is during inference
		softmax_backward(this->input_ctx, this->tensor);
}


MSELossNode::MSELossNode(Tensor* input, Tensor* expected_output, Tensor* loss) {
		this->tensor = loss;
		this->parents = {input->grad_node}; // no need to backpropagate to the expected output

		this->input_ctx = input;
		this->expected_output_ctx = expected_output;
}

void MSELossNode::apply() {
		mseloss_backward(this->input_ctx, this->expected_output_ctx, this->tensor);
}

void CrossEntropyLossNode::apply() {
		crossentropyloss_backward(this->input_ctx, this->expected_output_ctx, this->tensor);
}



