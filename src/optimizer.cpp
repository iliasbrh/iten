#include <vector>
#include <queue>
#include <memory>
#include "optimizer.hpp"


void Optimizer::zero_grad() {
		for (Tensor* param : this->parameters)
				memset(param->grad, 0.0f, param->size * sizeof(f32));
}

Adam::Adam(Module* model, f32 learning_rate, f32 weight_decay, f32 beta1, f32 beta2, f32 epsilon) {
		this->parameter_count = 0;

		this->w_decay = weight_decay;
		this->eps = epsilon;

		this->lr = learning_rate;
		this->b1 = beta1;
		this->b2 = beta2;
		this->b1_pow = beta1;
		this->b2_pow = beta2;

		// searching submodules of the model for parameters
		std::queue<Module*> q;
		q.push(model);

		Module* running_module;
		while (!q.empty()) {
				running_module = q.front();
				q.pop();
				for (Tensor* param : running_module->parameters) {
						this->parameters.push_back(param);
						this->parameter_count += param->size;
				}
				for (Module* submodule : running_module->sub_layers)
						q.push(submodule);
		}

		// allocation of means and squares
		for (Tensor* param : this->parameters) {
				this->means.push_back(std::make_unique<Tensor>(param->shape, false, false));
				this->squares.push_back(std::make_unique<Tensor>(param->shape, false, false));
		}
}

void Adam::step() {
		f32 grad;
		Tensor* param_tensor;
		f32* mean;
		f32* square;

		f32 mean_hat;
		f32 square_hat;

		for (u64 i=0; i<this->parameters.size(); i++) {
				param_tensor = this->parameters[i];
				mean = this->means[i].get()->data;
				square = this->squares[i].get()->data;
				for (u64 j=0; j<param_tensor->size; j++) {
						grad = param_tensor->grad[j];
						param_tensor->data[j] -= this->w_decay * this->lr * param_tensor->data[j];
						mean[j] = this->b1 * mean[j] + (1.0f-this->b1) * grad;
						square[j] = this->b2 * square[j] + (1.0f-this->b2) * powf(grad, 2);
						mean_hat = mean[j] / (1.0f - this->b1_pow);
						square_hat = square[j] / (1.0f - this->b2_pow);
						
						param_tensor->data[j] -= this->lr * mean_hat / (sqrtf(square_hat) + this->eps);
				}
		}

		this->b1_pow *= this->b1;
		this->b2_pow *= this->b2;
}


