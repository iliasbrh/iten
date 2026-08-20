#include <vector>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#include "src/types.hpp" 
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
		this->activation = nullptr;

		this->linear1 = new Linear(784, 16);
		this->linear2 = new Linear(16, 16);
		this->linear3 = new Linear(16, 10);

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



int main(void) {
		u32 train_sample_length = 50000; // <= 50000
		u32 test_sample_length = 10000; // <= 10000
		
		FILE* train_img_file = fopen("train-images.idx3-ubyte", "rb");
		img_dataset train_img = load_img_dataset(train_img_file, train_sample_length);
		
		FILE* train_lbl_file = fopen("train-labels.idx1-ubyte", "rb");
		lbl_dataset train_lbl = load_lbl_dataset(train_lbl_file, train_sample_length);


		FILE* test_img_file = fopen("test-images.idx3-ubyte", "rb");
		img_dataset test_img = load_img_dataset(test_img_file, test_sample_length);
		
		FILE* test_lbl_file = fopen("test-labels.idx1-ubyte", "rb");
		lbl_dataset test_lbl = load_lbl_dataset(test_lbl_file, test_sample_length);



		BigModel* model = new BigModel();
		//model->to(CUDA);
		MSELoss criterion;
		Adam optim(model);
		
		Tensor* input;
		Tensor* output;
		Tensor* expected_output;
		Tensor* loss_tensor;
		f32 pass_loss;
		for (u64 k=0; k<10; k++) {// epochs
				f64 running_loss = 0;
				for (u64 j=0; j<10; j++) {
						optim.zero_grad();

						input = train_img.images[j].get();
						input->shape = {1, 784};
						input->to(CUDA);

						output = model->forward(input);

						expected_output = train_lbl.labels[j].get();
						expected_output->to(CUDA);
						expected_output->shape = {1, 10};
						loss_tensor = criterion.forward(output, expected_output);

						loss_tensor->backward();
						optim.step();

						cudaMemcpy(&pass_loss, &loss_tensor->data[0], sizeof(f32), cudaMemcpyDeviceToHost);

						pass_loss = loss_tensor->data[0];
						running_loss += pass_loss;
				}

				cout << running_loss/10.0f << endl;
		}


		delete model;

		return 0;
}



