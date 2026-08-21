#include "model.hpp"


BigModel::BigModel() {
		this->activation = nullptr;

		this->linear1 = new Linear(784, 800);
		this->linear2 = new Linear(800, 800);
		this->linear3 = new Linear(800, 10);

		this->relu1 = new ReLU();
		this->relu2 = new ReLU();

		this->softmax = new Softmax();

		// only tracking sub layers that have parameters, so no relu and no softmax
		this->sub_layers = {this->linear1, this->linear2, this->linear3};
}
BigModel::~BigModel() {
		delete this->linear1;
		delete this->linear2;
		delete this->linear3;
		delete this->relu1;
		delete this->relu2;
		delete this->softmax;
}
Tensor* BigModel::forward(Tensor* input) {
		Tensor* x;
		x = this->linear1->forward(input);
		x = this->relu1->forward(x);
		x = this->linear2->forward(x);
		x = this->relu2->forward(x);
		x = this->linear3->forward(x);
		x = this->softmax->forward(x);

		return x;
}
