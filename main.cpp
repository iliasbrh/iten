#include <vector>
#include <iostream>
#include <stdlib.h>

#include "src/types.h" 
#include "src/maths.hpp"
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

				BigModel();
				~BigModel();
				Tensor* forward(Tensor* input) override;
};

BigModel::BigModel() {
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



		BigModel* model = new BigModel();
		MSELoss criterion({1});
		Adam optim(model);
		
		Tensor* input;
		Tensor* output;
		Tensor* expected_output;
		Tensor* loss_tensor;
		for (u64 k=0; k<10; k++) {// epochs
				f64 running_loss = 0;
				for (u64 j=0; j<20; j++) {
						optim.zero_grad();
						input = train_img.images[j];
						output = model->forward(input);
						expected_output = train_lbl.labels[j];
						loss_tensor = criterion.forward(output, expected_output);

						loss_tensor->backward();
						optim.step();

						running_loss += loss_tensor->data[0];
				}

				cout << running_loss/200.0f << endl;
		}


		delete model;
		cout << "Finito pipo" << endl;

		return 0;
}



