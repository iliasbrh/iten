#ifndef READ_UBYTE_HPP
#define READ_UBYTE_HPP

#include <stdlib.h>
#include <vector>
#include <stdio.h>

#include "src/tensor.hpp"

typedef struct {
		u64 n_elements;
		std::vector<Tensor*> images;
} img_dataset;
typedef struct {
		u64 n_elements;
		std::vector<Tensor*> labels;
} lbl_dataset;

Tensor* read_ubyte_image(FILE* input, u64 rows, u64 columns);
Tensor* read_ubyte_label(FILE* input);
img_dataset load_img_dataset(FILE* input, u64 size);
lbl_dataset load_lbl_dataset(FILE* input, u64 size);



#endif
