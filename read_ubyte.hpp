#ifndef READ_UBYTE_HPP
#define READ_UBYTE_HPP

#include <stdlib.h>
#include <stdio.h>

#include "src/tensor.hpp"


Tensor* load_mnist_images(FILE* input, u32 n_images);
Tensor* load_mnist_labels(FILE* input, u32 n_images);


#endif
