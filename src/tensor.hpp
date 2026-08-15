#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <cmath>
#include <queue>
#include <random>
#include "stdlib.h"

#include "types.h"
#include "maths.hpp"


class Node {
		protected:
				f32* grad;

		public:
				std::vector<Node*> parents; // public so that .backward can climb back up
				virtual void apply() = 0;

				Node() = default;
				virtual ~Node() = default; 
};

// as annoying as that seems, nodes need to have tensors as input of their constructors for context, because if they only had data pointers they would not know the shape of them
// so tensors have node attributes and nodes need tensors as constructor inputs, it does not feel right but that will do it for now


class Tensor {
		public:
				std::vector<u64> shape;
				u64 size;
				f32* data;

				b32 requires_grad; // if true, the constructor will allocate grads, and a node will be made during forward pass
				f32* grad;
				Node* grad_node;
				
				void backward(); // turns this->grad to ones, then builds the operation graph via BFS on the nodes, then applies gradients in the right order

				Tensor(std::vector<u64> tensor_shape, b32 is_input=false, b32 req_grad = true); // will return a tensor of zeros by default
				~Tensor();
};

void init_parameters(Tensor* parameters, u64 input_features);


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
				f32 divider; // keeping the sum of exponentials in memory

		public:	
				SoftmaxNode(Tensor* input, Tensor* tensor_node, f32 div, f32 temp = 1.0f);
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




#endif
