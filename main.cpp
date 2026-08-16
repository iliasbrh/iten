#include <vector>
#include <iostream>
#include <stdlib.h>
#include <chrono>

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
		MSELoss criterion;
		Adam optim(model);
		
		Tensor* input;
		Tensor* output;
		Tensor* expected_output;
		Tensor* loss_tensor;
		for (u64 k=0; k<200; k++) {// epochs
				f64 running_loss = 0;
				for (u64 j=0; j<10; j++) {
						/*
						auto start = chrono::high_resolution_clock::now();
						optim.zero_grad();
						auto end = chrono::high_resolution_clock::now();
						chrono::duration<f64> elapsed = end - start;
						cout << elapsed.count() << " : zero_grad" << endl; 

						start = chrono::high_resolution_clock::now();
						input = train_img.images[j];
						end = chrono::high_resolution_clock::now();
						elapsed = end - start;
						cout << elapsed.count() << " : get input" << endl; 

						start = chrono::high_resolution_clock::now();
						output = model->forward(input);
						end = chrono::high_resolution_clock::now();
						elapsed = end - start;
						cout << elapsed.count() << " : forward pass" << endl; 

						start = chrono::high_resolution_clock::now();
						expected_output = train_lbl.labels[j];
						end = chrono::high_resolution_clock::now();
						elapsed = end - start;
						cout << elapsed.count() << " : get label" << endl; 

						start = chrono::high_resolution_clock::now();
						loss_tensor = criterion.forward(output, expected_output);
						end = chrono::high_resolution_clock::now();
						elapsed = end - start;
						cout << elapsed.count() << " : forward criterion" << endl; 


						start = chrono::high_resolution_clock::now();
						loss_tensor->backward();
						end = chrono::high_resolution_clock::now();
						elapsed = end - start;
						cout << elapsed.count() << " : backward pass" << endl; 

						start = chrono::high_resolution_clock::now();
						optim.step();
						end = chrono::high_resolution_clock::now();
						elapsed = end - start;
						cout << elapsed.count() << " : optimizer step" << endl; 
						*/
						optim.zero_grad();
						input = train_img.images[j].get();
						input->shape = {1, 784};
						output = model->forward(input);
						expected_output = train_lbl.labels[j].get();
						expected_output->shape = {1, 10};
						loss_tensor = criterion.forward(output, expected_output);
						loss_tensor->backward();
						optim.step();

						running_loss += loss_tensor->data[0];
				}

				cout << running_loss/10.0f << endl;
		}


		delete model;

		return 0;
}



