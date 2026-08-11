#include "src/tensor.hpp"

#include <stdio.h>
#include <stdlib.h>
#include "read_ubyte.hpp"

Tensor* read_ubyte_image(FILE* input, u64 rows, u64 columns) {
                Tensor* out = new Tensor({rows, columns}, true);

                u8 buffer;
                for (u64 row=0; row<rows; row++)
                                for (u64 col=0; col<columns; col++) {
                                                fread(&buffer, 1, 1, input);
                                                out->data[row*columns+col] = (f32)(buffer) / 255.0f;
                                }

                return out;
}

Tensor* read_ubyte_label(FILE* input) {
				Tensor* out = new Tensor({10, 1});
                u8 buffer;
				fread(&buffer, 1, 1, input);
				for (u64 i=0; i<10; i++) {
						out->data[i] = (f32)(buffer == i);
				}

                return out;
}


img_dataset load_img_dataset(FILE* input, u64 size) {
                u8 buffer;
				img_dataset dataset;
				dataset.n_elements = size;

                for (u8 i=0; i<16; ++i) fread(&buffer, 1, 1, input); // reading header

                for (u64 i=0; i<size; ++i) dataset.images.push_back(read_ubyte_image(input, 784, 1));

                return dataset;
}

lbl_dataset load_lbl_dataset(FILE* input, u64 size) {
                u8 buffer;
				lbl_dataset dataset;
				dataset.n_elements = size;

                for (u8 i=0; i<8; ++i) fread(&buffer, 1, 1, input); // reading header

                for (u64 i=0; i<size; ++i) dataset.labels.push_back(read_ubyte_label(input));

                return dataset;
}
