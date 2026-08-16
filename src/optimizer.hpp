#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include "maths.hpp"
#include "tensor.hpp"
#include "modules.hpp"


class Optimizer {
		protected:
				u32 parameter_count;
				f32 lr;
				std::vector<Tensor*> parameters;

		public:
				Optimizer() = default;
				virtual ~Optimizer() = default;

				virtual void step() = 0;
				void zero_grad();
};

class Adam : public Optimizer {
		private:
				f32 w_decay;
				f32 eps;

				f32 b1;
				f32 b2;

				f32 b1_pow;
				f32 b2_pow;

				std::vector<std::unique_ptr<Tensor>> means;
				std::vector<std::unique_ptr<Tensor>> squares;
		
		public:
				Adam(Module* model, f32 learning_rate=0.001f, f32 weight_decay=0.0f, f32 beta1=0.9f, f32 beta2=0.999f, f32 epsilon=1e-8f);

				void step() override;
};













#endif
