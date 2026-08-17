#include "cuda_runtime.h"
#include "types.hpp"
#include <cassert>


enum device {
		CPU,
		CUDA
};

f32* _alloc(u32 size, device target);

void _free(f32* ptr, device dev);

f32* _move_to(f32* data, u32 size, device target);

void _check_same_device(std::vector<device> devices);
