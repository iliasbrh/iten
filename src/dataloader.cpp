#include "dataloader.hpp"


void shuffle_indexing(u32* indexing, u32 n_samples) {
		for (i32 i=n_samples-1; i>0; i--) {
				u32 rand_idx = randint(0, i);
				u32 tmp = indexing[rand_idx];
				indexing[rand_idx] = indexing[i];
				indexing[i] = tmp;
		}
}


MNISTDataloader::MNISTDataloader(Tensor* input_data, Tensor* expected_output_data, u32 batchsize, b32 random_shuffle) {
		this->n_samples = (input_data->shape[0] / batchsize) * batchsize; // we skip the last elements so that n_samples is a multiple of the batchsize
		this->batch_size = batchsize;
		this->shuffle = random_shuffle;
		this->idx = 0;

		// preparing indexing
		this->indexes = (u32*)malloc(this->n_samples * sizeof(u32));
		for (u32 i=0; i<this->n_samples; i++) this->indexes[i] = i;
		if (this->shuffle) shuffle_indexing(this->indexes, this->n_samples);
		
		// storing data pointers
		this->input = input_data;
		this->expected_output = expected_output_data;

		// computing batch shape and sample size for input and expected output
		std::vector<u32> input_batch_shape = {this->batch_size};
		this->input_sample_size = 1;
		for (u32 i=1; i<this->input->shape.size(); i++) {
				input_batch_shape.push_back(this->input->shape[i]);
				this->input_sample_size *= this->input->shape[i];
		}

		std::vector<u32> expected_output_batch_shape = {this->batch_size};
		this->expected_output_sample_size = 1;
		for (u32 i=1; i<this->expected_output->shape.size(); i++) {
				expected_output_batch_shape.push_back(this->expected_output->shape[i]);
				this->expected_output_sample_size *= this->expected_output->shape[i];
		}

		// allocating batch memory
		this->input_batch = new Tensor(input_batch_shape, CPU, 1, 1);
		this->expected_output_batch = new Tensor(expected_output_batch_shape, CPU, 1, 1);
}


MNISTDataloader::~MNISTDataloader() {
		free(this->indexes);

		delete this->input_batch;
		delete this->expected_output_batch;
		
		// to avoid having to free data from the main entry point
		delete this->input;
		delete this->expected_output;
}

std::vector<Tensor*> MNISTDataloader::get_batch() {
		// shuffling in case we start a new epoch
		if (this->idx >= this->n_samples) {
				if (this->shuffle) shuffle_indexing(this->indexes, this->n_samples);
				this->idx = 0;
		}
		
		// in case batch memory was moved to gpu in the training loop
		this->input_batch->to(CPU);
		this->expected_output_batch->to(CPU);

		for (u32 b=0; b<this->batch_size; b++) {
				memcpy(&this->input_batch->data[b * this->input_sample_size], 
				       &this->input->data[this->indexes[this->idx] * this->input_sample_size],
				       this->input_sample_size * sizeof(f32));
				memcpy(&this->expected_output_batch->data[b * this->expected_output_sample_size], 
				       &this->expected_output->data[this->indexes[this->idx] * this->expected_output_sample_size],
				       this->expected_output_sample_size * sizeof(f32));

				this->idx++;
		}
				
		return {this->input_batch, this->expected_output_batch};

		// batch moved to gpu in the training loop
}












