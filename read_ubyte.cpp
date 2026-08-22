#include "read_ubyte.hpp"

Tensor* load_mnist_images(FILE* input, u32 n_images) {
				Tensor* out = new Tensor({n_images, 784}, CPU, 0, 0);
				
				u8 buffer;
				for (u32 i=0; i<16; i++) 
						fread(&buffer, 1, 1, input); // reading header
				for (u32 i=0; i<n_images*784; i++) {
						fread(&buffer, 1, 1, input);
						out->data[i] = (f32)(buffer) / 255.0f;
				}

				return out;
}
Tensor* load_mnist_labels(FILE* input, u32 n_images) {
				Tensor* out = new Tensor({n_images, 10}, CPU, 0, 0);

				u8 buffer;
				for (u32 i=0; i<8; i++) fread(&buffer, 1, 1, input); // reading header
				for (u32 i=0; i<n_images; i++) {
						fread(&buffer, 1, 1, input);
						for (u8 j=0; j<10; j++)
								out->data[i*10+j] = (f32)(j == buffer);
				}

				return out;
}
