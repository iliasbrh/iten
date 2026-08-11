#include <vector>
#include <queue>
#include "optimizer.hpp"


#include <iostream>
using namespace std;

void Optimizer::zero_grad() {
		for (Tensor* param : this->parameters)
				memset(param->grad, 0, param->size * sizeof(f32));
}

Adam::Adam(Module* model, f32 learning_rate, f32 weight_decay, f32 beta1, f32 beta2, f32 eps) {
		this->parameter_count = 0;

		this->weight_decay = weight_decay;
		this->eps = eps;

		this->learning_rate = learning_rate;
		this->beta1 = beta1;
		this->beta2 = beta2;

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
				this->means.push_back(new Tensor(param->shape, false, false));
				this->squares.push_back(new Tensor(param->shape, false, false));
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
				mean = this->means[i]->data;
				square = this->squares[i]->data;
				for (u64 j=0; j<param_tensor->size; j++) {
						grad = param_tensor->grad[j];
						param_tensor->data[j] -= this->weight_decay * this->learning_rate * param_tensor->data[j];
						mean[j] = this->beta1 * mean[j] + (1.-this->beta1) * grad;
						square[j] = this->beta2 * square[j] + (1.-this->beta2) * pow(grad, 2);
						mean_hat = mean[j] / (1. - pow(this->beta1, this->running_step));
						square_hat = square[j] / (1. - pow(this->beta2, this->running_step));
						
						param_tensor->data[j] -= this->learning_rate * mean_hat / (sqrt(square_hat) + this->eps);
				}
		}


		this->running_step += 1;
}
