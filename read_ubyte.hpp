#ifndef READ_UBYTE_HPP
#define READ_UBYTE_HPP

#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <memory>

#include "src/tensor.hpp"

typedef struct {
		u64 n_elements;
		std::vector<std::unique_ptr<Tensor>> images;
} img_dataset;
typedef struct {
		u64 n_elements;
		std::vector<std::unique_ptr<Tensor>> labels;
} lbl_dataset;

std::unique_ptr<Tensor> read_ubyte_image(FILE* input, u32 rows, u32 columns);
std::unique_ptr<Tensor> read_ubyte_label(FILE* input);
img_dataset load_img_dataset(FILE* input, u32 size);
lbl_dataset load_lbl_dataset(FILE* input, u32 size);



#endif
