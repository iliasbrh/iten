#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include <vector>
#include "utils.hpp"
#include "tensor.hpp"
#include "modules.hpp"


class Optimizer {
		public:
				std::vector<Tensor*> parameters;
				u64 parameter_count;

				f32 lr;

				virtual void step() = 0;
				void zero_grad();

				Optimizer() = default;
				virtual ~Optimizer() = default;
};

class Adam : public Optimizer {
		public:
				f32 w_decay;
				f32 eps;

				f32 b1;
				f32 b2;

				std::vector<Tensor*> means;
				std::vector<Tensor*> squares;

				u64 running_step;

				void step() override;

				Adam(Module* model, f32 learning_rate=0.001f, f32 weight_decay=0.0f, f32 beta1=0.9f, f32 beta2=0.999f, f32 epsilon=1e-8f);
};













#endif
