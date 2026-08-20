#include "cuda_runtime.h"
#include "types.hpp"
#include <assert.h>


enum device_t {
		CPU,
		CUDA
};

f32* _alloc(u32 size, device_t target);

void _free(f32* ptr, device_t device);

f32* _move_to(f32* data, u32 size, device_t target);

void _check_same_device(std::vector<device_t> devices);
