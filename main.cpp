#include <stdlib.h>
#include <iostream>

#include "iten.hpp"
#include "read_ubyte.hpp"
#include "model.hpp"


using namespace std;



int main(void) {
		u32 train_sample_length = 60000;
		u32 test_sample_length = 10000;
		
		FILE* train_img_file = fopen("mnist_dataset/train-images.idx3-ubyte", "rb");
		FILE* train_lbl_file = fopen("mnist_dataset/train-labels.idx1-ubyte", "rb");
		FILE* test_img_file = fopen("mnist_dataset/test-images.idx3-ubyte", "rb");
		FILE* test_lbl_file = fopen("mnist_dataset/test-labels.idx1-ubyte", "rb");

		
		Tensor* X_train = load_mnist_images(train_img_file, train_sample_length);
		Tensor* Y_train = load_mnist_labels(train_lbl_file, train_sample_length);

		u32 batch_size = 32;
		u32 n_batch = train_sample_length / batch_size;


		MNISTDataloader train_dataloader(X_train, Y_train, batch_size);


		BigModel model;
		model.to(CUDA);
		MSELoss criterion;

		Adam optim(model);
		
		Tensor* input, output, expected_output, loss_tensor;
		f32 pass_loss;

		for (u64 k=0; k<10; k++) {// epochs
				f64 running_loss = 0;
				for (u64 j=0; j<n_batch; j++) {
						optim.zero_grad();

						vector<Tensor*> batch_data = train_dataloader.get_batch();
						input = batch_data[0];
						expected_output = batch_data[1];
						
						input->to(CUDA);
						output = model.forward(input);

						expected_output->to(CUDA);
						loss_tensor = criterion.forward(output, expected_output);

						loss_tensor->backward();
						optim.step();

						cudaMemcpy(&pass_loss, &loss_tensor->data[0], sizeof(f32), cudaMemcpyDeviceToHost);

						running_loss += pass_loss;
				}
				cout << running_loss/(f32)n_batch << endl;
		}



		return 0;
}



