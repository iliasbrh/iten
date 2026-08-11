#ifndef MODULES_HPP
#define MODULES_HPP

#include <vector>
#include <cmath>
#include "tensor.hpp"

class Module {
		public:
				Tensor* activation; // useful to have this attribute so that post-layer activation's allocation is done by the constructor
				// i think in more advanced implementations the allocation is done dynamically during forward pass, thanks to linear allocators on cuda device
				std::vector<Tensor*> parameters;
				std::vector<Module*> sub_layers; // e.g. for attention that uses multiple linears

				virtual ~Module() = default;
				virtual Tensor* forward(Tensor* input) = 0; // returns the activation
};

class Linear : public Module {
		public:
				Tensor* weight;
				Tensor* bias;
				Linear(const std::vector<u64> &activation_shape, u64 input_features, u64 output_features);

				Tensor* forward(Tensor* input) override;
};

class ReLU : public Module {
		public:
				ReLU(const std::vector<u64> &activation_shape);

				Tensor* forward(Tensor* input) override;
};

class Softmax : public Module {
		public:
				f32 temperature;
				Softmax(const std::vector<u64> &activation_shape, f32 temperature = 1);

				Tensor* forward(Tensor* input) override;
};

class MSELoss : public Module {
		public:
				MSELoss(const std::vector<u64> &activation_shape);

				Tensor* forward(Tensor* input) override {return NULL;};
				Tensor* forward(Tensor* input, Tensor* expected_output);
};



#endif
