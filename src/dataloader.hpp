#ifndef DATALOADER_HPP
#define DATALOADER_HPP

#include "types.hpp"
#include "read_ubyte.cpp"
#include "prng.h"

void shuffle_indexing(u32* indexing, u32 n_samples);


class Dataloader {
		public:
				u32 batch_size;
				u32 n_samples;

				u32* indexes;
				u32 idx;

				Dataloader() = default;
				virtual ~Dataloader() = default;

				virtual std::vector<Tensor*> get_batch(); // generator with incrementing idx
};

class MNISTDataloader : public Dataloader {
		public:
				u32 input_sample_size;
				u32 expected_output_sample_size;

				Tensor* input;
				Tensor* expected_output;

				Tensor* input_batch;
				Tensor* expected_output_batch;

				MNISTDataloader(Tensor* input_data, Tensor* expected_output_data, u32 batchsize);
				~MNISTDataloader();

				std::vector<Tensor*> get_batch() override;
}


#endif
