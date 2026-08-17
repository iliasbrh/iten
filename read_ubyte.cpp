#include "src/tensor.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include "read_ubyte.hpp"

std::unique_ptr<Tensor> read_ubyte_image(FILE* input, u32 rows, u32 columns) {
				std::unique_ptr<Tensor> out = std::make_unique<Tensor>(std::vector<u32>{rows, columns}, CPU, true);

                u8 buffer;
                for (u32 row=0; row<rows; row++)
                                for (u32 col=0; col<columns; col++) {
                                                fread(&buffer, 1, 1, input);
                                                out->data[row*columns+col] = (f32)(buffer) / 255.0f;
                                }

                return out;
}

std::unique_ptr<Tensor> read_ubyte_label(FILE* input) {
				std::unique_ptr<Tensor> out = std::make_unique<Tensor>(std::vector<u32>{1, 10}, CPU);
                u8 buffer;
				fread(&buffer, 1, 1, input);
				for (u32 i=0; i<10; i++) {
						out->data[i] = (f32)(buffer == i);
				}

                return out;
}


img_dataset load_img_dataset(FILE* input, u32 size) {
                u8 buffer;
				img_dataset dataset;
				dataset.n_elements = size;

                for (u8 i=0; i<16; ++i) fread(&buffer, 1, 1, input); // reading header

                for (u32 i=0; i<size; ++i) dataset.images.push_back(read_ubyte_image(input, 784, 1));

                return dataset;
}

lbl_dataset load_lbl_dataset(FILE* input, u32 size) {
                u8 buffer;
				lbl_dataset dataset;
				dataset.n_elements = size;

                for (u8 i=0; i<8; ++i) fread(&buffer, 1, 1, input); // reading header

                for (u32 i=0; i<size; ++i) dataset.labels.push_back(read_ubyte_label(input));

                return dataset;
}
