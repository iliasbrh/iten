#ifndef MODULES_HPP
#define MODULES_HPP

#include "types.hpp"
#include "tensor.hpp"
#include "functional.cuh"

class Module {
		protected:
				Tensor* activation; 
				// useful to have this attribute so that post-layer activation's allocation is done by the constructor

		public:
				std::vector<Tensor*> parameters;
				std::vector<Module*> sub_layers; // e.g. for attention that uses multiple linears
				void to(device_t device);

				virtual ~Module();
				virtual Tensor* forward(Tensor* input) = 0; // returns the activation

				// add a .to function to change device
				// this function will go through all sub_layers and parameters
};

class Linear : public Module {
		private:
				Tensor* weight;
				Tensor* bias;

		public:
				Linear(u32 input_features, u32 output_features, device_t target_device=CPU);
				~Linear();

				Tensor* forward(Tensor* input) override;
};

class ReLU : public Module {
		public:
				ReLU();

				Tensor* forward(Tensor* input) override;
};

class Softmax : public Module {
		private:
				f32 temp;

		public:
				Softmax(f32 temperature = 1.0f);

				Tensor* forward(Tensor* input) override;
};

class MSELoss : public Module {
		private:
				Tensor* forward(Tensor*) override {return nullptr;};

		public:
				MSELoss();
				
				Tensor* forward(Tensor* input, Tensor* expected_output);
};
 


#endif
