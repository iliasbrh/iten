#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "types.hpp"
#include "functional.cuh"



class Node;
// as annoying as that seems, nodes need to have tensors as input of their constructors for context, because if they only had data pointers they would not know the shape of them
// so tensors have node attributes (to be able to run tensor.backward())
// and nodes need tensors as constructor inputs, it does not feel right but that will do it for now


class Tensor {
		public:
				std::vector<u32> shape;
				u32 size;
				device_t device;
				f32* data;

				b32 requires_grad; // if true, the constructor will allocate grads, and a node will be made during forward pass
				f32* grad;
				Node* grad_node;
				
				void to(device_t dev);
				void backward(); // turns this->grad to ones, then builds the operation graph via BFS on the nodes, then applies gradients in the right order

				
				Tensor(std::vector<u32> tensor_shape, device_t target_device=CPU, b32 is_input=false, b32 req_grad = true); // will return a tensor of zeros by default
				~Tensor();
};


class Node {
		protected:
				Tensor* tensor;

		public:
				std::vector<Node*> parents; // public so that .backward can climb back up
				virtual void apply() = 0;

				Node() = default;
				virtual ~Node() = default; 
};

class StartNode : public Node { // for nodes that are not children of any other node, e.g. inputs or parameters
		public:
				StartNode(Tensor* tensor_node);
				void apply() override;
};

class LinearNode : public Node {
		private:
				Tensor* weight_ctx;
				Tensor* bias_ctx;
				Tensor* input_ctx;

		public:
				LinearNode(Tensor* input, Tensor* weight, Tensor* bias, Tensor* tensor_node);
				void apply() override;
};

class ReLUNode : public Node {
		private:
				Tensor* input_ctx;

		public:
				ReLUNode(Tensor* input, Tensor* tensor_node);
				void apply() override;
};

class SoftmaxNode : public Node {
		private:
				Tensor* input_ctx;
				Tensor* output_ctx;
				f32 temperature;

		public:	
				SoftmaxNode(Tensor* input, Tensor* tensor_node, f32 temp = 1.0f);
				void apply() override;
};

class MSELossNode : public Node {
		private:
				Tensor* input_ctx;
				Tensor* expected_output_ctx;
		
		public:
				MSELossNode(Tensor* input, Tensor* expected_output, Tensor* loss);
				void apply() override;
};

class CrossEntropyLossNode : public Node {
		private:
				Tensor* input_ctx;
				Tensor* expected_output_ctx;

		public:
				CrossEntropyLossNode(Tensor* input, Tensor* expected_output, Tensor* loss);
				void apply() override;
}


#endif
