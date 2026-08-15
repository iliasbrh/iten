#include "tensor.hpp"

void init_parameters(Tensor* parameters, u64 input_features) {
		f32 bound = 1.0f/sqrtf((f32)input_features);

		std::random_device rd{};
		std::mt19937 gen{rd()};
		std::uniform_real_distribution<f32> distribution(-bound, bound); 
		for (u64 i=0; i<parameters->size; i++)
				parameters->data[i] = distribution(gen);
}


/////////////
// Tensors //
/////////////

Tensor::Tensor(std::vector<u64> tensor_shape, b32 is_input, b32 req_grad) {
		this->shape = tensor_shape;
		this->size = multiply_vector(tensor_shape);
		this->requires_grad = req_grad;
		
		this->data = (f32*)malloc(this->size * sizeof(f32));

		if (this->requires_grad) 
		{
				this->grad = (f32*)malloc(this->size * sizeof(f32));
				memset(this->grad, 0.0f, this->size*sizeof(f32));
		}
		else {this->grad = nullptr;}

		memset(this->data, 0.0f, this->size*sizeof(f32));

		if (is_input)
			   this->grad_node = new StartNode(this);
		else {this->grad_node = nullptr;}
}

Tensor::~Tensor() {
		free(this->data);
		if (this->grad)
				free(this->grad);

		if (this->grad_node)
				delete this->grad_node;
}

void Tensor::backward() {
		for (u64 i=0; i<this->size; i++)
				this->grad[i] = 1.0f;
		
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
		this->grad = tensor_node->grad;
}

void StartNode::apply() {}


LinearNode::LinearNode(Tensor* input, Tensor* weight, Tensor* bias, Tensor* tensor_node) {
		this->grad = tensor_node->grad;
		this->parents = {input->grad_node, weight->grad_node, bias->grad_node};

		this->weight_ctx = weight;
		this->bias_ctx = bias;
		this->input_ctx = input;
}

void LinearNode::apply() {
		// bias grad
		mat_add(this->bias_ctx->grad, this->grad, this->bias_ctx->grad, this->bias_ctx->shape[0], this->bias_ctx->shape[1]);

		// weight grad
		// weight of shape (O, I), input_ctx of shape (I, 1), output of shape (O, 1)
		mat_mul_transpose_B(this->grad, this->input_ctx->data, this->weight_ctx->grad, this->weight_ctx->shape[0], this->input_ctx->shape[1], this->weight_ctx->shape[1], false);

		// input grad
		mat_mul_transpose_A(this->weight_ctx->data, this->grad, this->input_ctx->grad, this->weight_ctx->shape[1], this->weight_ctx->shape[0], this->input_ctx->shape[1], false);
}


ReLUNode::ReLUNode(Tensor* input, Tensor* tensor_node) {
		this->grad = tensor_node->grad;
		this->parents = {input->grad_node};

		this->input_ctx = input;
}

void ReLUNode::apply() {
		// rewrite that with a mat_elementwise_mul function from utils
		for (u64 i=0; i<this->input_ctx->size; i++)
				this->input_ctx->grad[i] += this->input_ctx->data[i] > 0.0f ? this->grad[i] : 0.0f;
}


SoftmaxNode::SoftmaxNode(Tensor* input, Tensor* tensor_node, f32 div, f32 temp) {
		this->grad = tensor_node->grad;
		this->parents = {input->grad_node};

		this->input_ctx = input;
		this->output_ctx = tensor_node;
		this->divider = div;
		this->temperature = temp;
}

void SoftmaxNode::apply() {
		// do it temperature wise in the future, and make it with operations from utils (since they will be the ones to be parallelized in the future)
		// in fact, temperature is never used during training so adding it to backward passes would be useless. the only time where we set it to something other than 1 is during inference
		f32 dot_prod = 0.0f;
		for (u64 i=0; i<this->input_ctx->size; i++)
				dot_prod += this->grad[i] * this->output_ctx->data[i];

		for (u64 i=0; i<this->input_ctx->size; i++)
				this->input_ctx->grad[i] += this->output_ctx->data[i] * (this->grad[i] - dot_prod);
}


MSELossNode::MSELossNode(Tensor* input, Tensor* expected_output, Tensor* loss) {
		this->grad = loss->grad;
		this->parents = {input->grad_node}; 
		// no need to backpropagate to the expected output

		this->input_ctx = input;
		this->expected_output_ctx = expected_output;
}

void MSELossNode::apply() {
		// rewrite that with mat_subtract and mat_scale functions from utils
		for (u64 i=0; i<this->input_ctx->size; i++)
				this->input_ctx->grad[i] += 2.0f*(this->input_ctx->data[i] - this->expected_output_ctx->data[i]) * this->grad[0] / (f32)this->input_ctx->size; // the output of MSELoss is a single value
}





