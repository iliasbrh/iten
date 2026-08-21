#include <stdlib.h>
#include <iostream>

#include "iten.hpp"
#include "read_ubyte.hpp"
#include "model.hpp"


using namespace std;







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


		f32* dummy_ptr;
		cudaMalloc(&dummy_ptr, 1 << 30);


		BigModel* model = new BigModel();
		model->to(CUDA);
		MSELoss criterion;

		Adam optim(model);
		
		Tensor* input;
		Tensor* output;
		Tensor* expected_output;
		Tensor* loss_tensor;
		f32 pass_loss;
		for (u64 k=0; k<100; k++) {// epochs
				f64 running_loss = 0;
				for (u64 j=0; j<20; j++) {
						optim.zero_grad();

						input = train_img.images[j].get();
						input->shape = {1, 784};
						input->to(CUDA);

						output = model->forward(input);

						expected_output = train_lbl.labels[j].get();
						expected_output->shape = {1, 10};
						expected_output->to(CUDA);
						loss_tensor = criterion.forward(output, expected_output);

						loss_tensor->backward();
						optim.step();

						cudaMemcpy(&pass_loss, &loss_tensor->data[0], sizeof(f32), cudaMemcpyDeviceToHost);

						running_loss += pass_loss;

						input->to(CPU);
						expected_output->to(CPU);
				}
				cout << running_loss/1000.0f << endl;
		}


		delete model;

		return 0;
}



