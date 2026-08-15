#ifndef MODULES_HPP
#define MODULES_HPP

#include <vector>
#include <cmath>
#include "tensor.hpp"

class Module {
		protected:
				Tensor* activation; 
				// useful to have this attribute so that post-layer activation's allocation is done by the constructor

		public:
				std::vector<Tensor*> parameters;
				std::vector<Module*> sub_layers; // e.g. for attention that uses multiple linears

				virtual ~Module() {delete this->activation;};
				virtual Tensor* forward(Tensor* input) = 0; // returns the activation
};

class Linear : public Module {
		private:
				Tensor* weight;
				Tensor* bias;

		public:
				Linear(const std::vector<u64> &activation_shape, u64 input_features, u64 output_features);
				~Linear();

				Tensor* forward(Tensor* input) override;
};

class ReLU : public Module {
		public:
				ReLU(const std::vector<u64> &activation_shape);

				Tensor* forward(Tensor* input) override;
};

class Softmax : public Module {
		private:
				f32 temp;

		public:
				Softmax(const std::vector<u64> &activation_shape, f32 temperature = 1.0f);

				Tensor* forward(Tensor* input) override;
};

class MSELoss : public Module {
		private:
				Tensor* forward(Tensor*) override {return nullptr;};

		public:
				MSELoss(const std::vector<u64> &activation_shape);
				
				Tensor* forward(Tensor* input, Tensor* expected_output);
};



#endif
