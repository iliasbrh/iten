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

				f32 learning_rate;

				virtual void step() = 0;
				void zero_grad();

				Optimizer() = default;
				virtual ~Optimizer() = default;
};

class Adam : public Optimizer {
		public:
				f32 weight_decay;
				f32 eps;

				f32 beta1;
				f32 beta2;

				std::vector<Tensor*> means;
				std::vector<Tensor*> squares;

				u64 running_step = 1;

				void step() override;

				Adam(Module* model, f32 learning_rate=0.001, f32 weight_decay=0., f32 beta1=0.9, f32 beta2=0.999, f32 eps=1e-5);
};













#endif
