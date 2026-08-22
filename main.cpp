#include <stdlib.h>
#include <iostream>

#include "iten.hpp"
#include "read_ubyte.hpp"
#include "model.hpp"


using namespace std;

u32 argmax(f32* data) { // for data of length 10
	u32 res = 0;
	for (u32 i=1; i<10; i++) {
		if (data[i] > data[res])
			res = i;
	}
	return res;
}


int main(void) {
		cout << "Finished compiling, now running from entry point" << endl;

		u32 train_sample_length = 32*1200;
		u32 test_sample_length = 10000;
		
		FILE* train_img_file = fopen("mnist_dataset/train-images.idx3-ubyte", "rb");
		FILE* train_lbl_file = fopen("mnist_dataset/train-labels.idx1-ubyte", "rb");
		FILE* test_img_file = fopen("mnist_dataset/test-images.idx3-ubyte", "rb");
		FILE* test_lbl_file = fopen("mnist_dataset/test-labels.idx1-ubyte", "rb");

		
		Tensor* X_train = load_mnist_images(train_img_file, train_sample_length);
		Tensor* Y_train = load_mnist_labels(train_lbl_file, train_sample_length);
		Tensor* X_test = load_mnist_images(test_img_file, test_sample_length);
		Tensor* Y_test = load_mnist_labels(test_lbl_file, test_sample_length);

		cout << "Loaded data" << endl;

		u32 batch_size = 32;
		u32 n_batch = train_sample_length / batch_size;

		device_t device = CPU;

		MNISTDataloader train_dataloader(X_train, Y_train, batch_size, true);
		MNISTDataloader test_dataloader(X_test, Y_test, 1);


		BigModel model;
		model.to(device);
		CrossEntropyLoss criterion;

		Adam optim(&model);
		
		Tensor* input;
		Tensor* output;
		Tensor* expected_output;
		Tensor* loss_tensor;
		f32 pass_loss;

		cout << "Starting training" << endl;

		for (u64 k=0; k<10; k++) {// epochs
				cout << "Epoch " << k+1 << " :" << endl;
				f64 running_loss = 0;
				for (u64 j=0; j<n_batch; j++) {
						optim.zero_grad();

						vector<Tensor*> batch_data = train_dataloader.get_batch();
						input = batch_data[0];
						expected_output = batch_data[1];
						
						input->to(device);
						output = model.forward(input);

						expected_output->to(device);
						loss_tensor = criterion.forward(output, expected_output);

						loss_tensor->backward();
						optim.step();

						if (device == CUDA)
							cudaMemcpy(&pass_loss, &loss_tensor->data[0], sizeof(f32), cudaMemcpyDeviceToHost);
						else if (device == CPU)
							pass_loss = loss_tensor->data[0];

						running_loss += pass_loss;
				}
				cout << "Training loss : " << running_loss/(f32)n_batch << endl;
				
				running_loss = 0;
				f32* output_probabilities = (f32*)malloc(10*sizeof(f32));
				u32 good_predictions = 0;
				u32 prediction;
				for (u64 j=0; j<test_sample_length; j++) {
						vector<Tensor*> batch_data = test_dataloader.get_batch();
						input = batch_data[0];
						expected_output = batch_data[1];

						u32 expected_idx = argmax(expected_output->data);

						input->to(device);
						output = model.forward(input);
						expected_output->to(device);
						loss_tensor = criterion.forward(output, expected_output);

						if (device == CUDA) {
							cudaMemcpy(&pass_loss, &loss_tensor->data[0], sizeof(f32), cudaMemcpyDeviceToHost);
							cudaMemcpy(output_probabilities, &output->data[0], 10*sizeof(f32), cudaMemcpyDeviceToHost);
							prediction = argmax(output_probabilities);
						}
						else if (device == CPU) {
							pass_loss = loss_tensor->data[0];
							prediction = argmax(output->data);
						}

						running_loss += pass_loss;
						good_predictions += (prediction == expected_idx);
				}
				cout << "Test loss : " << running_loss/(f64)test_sample_length << endl;
				cout << "Accuracy : " << 100*(f32)good_predictions/(f32)test_sample_length << "%" << endl;
				cout << "------------------------------" << endl;
		}



		return 0;
}



