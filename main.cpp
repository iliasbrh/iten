#include <vector>
#include <iostream>
#include <stdlib.h>

#include "src/types.h" 
#include "src/utils.hpp"
#include "src/tensor.hpp"
#include "src/modules.hpp"
#include "src/optimizer.hpp"
#include "read_ubyte.hpp"

using namespace std;

class BigModel : public Module {
		public:
				Linear* linear1;
				ReLU* relu1;
				Linear* linear2;
				ReLU* relu2;
				Linear* linear3;
				Softmax* softmax;

				BigModel(const vector<u64> &activation_shape);
				Tensor* forward(Tensor* input) override;
};

BigModel::BigModel(const vector<u64> &activation_shape) {
		this->activation = new Tensor(activation_shape);

		this->linear1 = new Linear({800, 1}, 784, 800);
		this->linear2 = new Linear({800, 1}, 800, 800);
		this->linear3 = new Linear({10, 1}, 800, 10);

		this->relu1 = new ReLU({800, 1});
		this->relu2 = new ReLU({800, 1});

		this->softmax = new Softmax({10, 1});

		// only tracking sub layers that have parameters, so no relu and no softmax
		this->sub_layers.push_back(this->linear1);
		this->sub_layers.push_back(this->linear2);
		this->sub_layers.push_back(this->linear3);
}
Tensor* BigModel::forward(Tensor* input) {
		Tensor* x;
		x = this->linear1->forward(input);
		x = this->relu1->forward(x);
		x = this->linear2->forward(x);
		x = this->relu2->forward(x);
		x = this->linear3->forward(x);
		x = this->softmax->forward(x);
		
		this->activation = x;
		return x;
}



int main(void) {
		u64 train_sample_length = 50000; // <= 50000
		u64 test_sample_length = 10000; // <= 10000
		
		FILE* train_img_file = fopen("train-images.idx3-ubyte", "rb");
		img_dataset train_img = load_img_dataset(train_img_file, train_sample_length);
		
		FILE* train_lbl_file = fopen("train-labels.idx1-ubyte", "rb");
		lbl_dataset train_lbl = load_lbl_dataset(train_lbl_file, train_sample_length);


		FILE* test_img_file = fopen("test-images.idx3-ubyte", "rb");
		img_dataset test_img = load_img_dataset(test_img_file, test_sample_length);
		
		FILE* test_lbl_file = fopen("test-labels.idx1-ubyte", "rb");
		lbl_dataset test_lbl = load_lbl_dataset(test_lbl_file, test_sample_length);



		BigModel* model = new BigModel({10, 1});
		MSELoss criterion({1});
		Adam optim(model);
		
		Tensor* input;
		Tensor* output;
		Tensor* expected_output;
		Tensor* loss_tensor;
		for (u64 k=0; k<20; k++) {// epochs
				f64 running_loss = 0;
				for (u64 j=0; j<10; j++) {
						optim.zero_grad();
						input = train_img.images[j];
						output = model->forward(input);
						expected_output = train_lbl.labels[j];
						loss_tensor = criterion.forward(output, expected_output);

						/*
						cout << "i=" << k << ", j=" << j <<endl;
						cout << "Parameters : " << endl;
						cout << "Linear 1 weights" << endl;
						for (u64 i=0; i<model->linear1->weight->size; i++)
								cout << model->linear1->weight->data[i] << endl;
						cout << "Linear 1 biases" << endl;
						for (u64 i=0; i<model->linear1->bias->size; i++)
								cout << model->linear1->bias->data[i] << endl;
						cout << "Linear 2 weights" << endl;
						for (u64 i=0; i<model->linear2->weight->size; i++)
								cout << model->linear2->weight->data[i] << endl;
						cout << "Linear 2 biases" << endl;
						for (u64 i=0; i<model->linear2->bias->size; i++)
								cout << model->linear2->bias->data[i] << endl;
						cout << "Linear 3 weights" << endl;
						for (u64 i=0; i<model->linear3->weight->size; i++)
								cout << model->linear3->weight->data[i] << endl;
						cout << "Linear 3 biases" << endl;
						for (u64 i=0; i<model->linear3->bias->size; i++)
								cout << model->linear3->bias->data[i] << endl;

						cout << "Linear 1 activation" << endl;
						for (u64 i=0; i<model->linear1->activation->size; i++)
								cout << model->linear1->activation->data[i] << endl;
						cout << "Relu 1 activation" << endl;
						for (u64 i=0; i<model->relu1->activation->size; i++)
								cout << model->relu1->activation->data[i] << endl;
						cout << "Linear 2 activation" << endl;
						for (u64 i=0; i<model->linear2->activation->size; i++)
								cout << model->linear2->activation->data[i] << endl;
						cout << "Relu 2 activation" << endl;
						for (u64 i=0; i<model->relu2->activation->size; i++)
								cout << model->relu2->activation->data[i] << endl;
						cout << "Linear 3 activation" << endl;
						for (u64 i=0; i<model->linear3->activation->size; i++)
								cout << model->linear3->activation->data[i] << endl;
						cout << "Softmax activation" << endl;
						for (u64 i=0; i<model->softmax->activation->size; i++)
								cout << model->softmax->activation->data[i] << endl;
						cout << "Loss" << endl;
						for (u64 i=0; i<criterion.activation->size; i++)
								cout << criterion.activation->data[i] << endl;
						cout << "expected output" << endl;
						for (u64 i=0; i<10; i++)
								cout << expected_output->data[i] << endl;
						*/

						loss_tensor->backward();
						optim.step();

						running_loss += loss_tensor->data[0];
				}

				cout << running_loss/100. << endl;
		}


		cout << "Finito pipo" << endl;

		return 0;
}



