#include "device_memory.cuh"

f32* _alloc(u32 size, device target) {
		f32* target_ptr;
		if (target == CUDA) {
				cudaMalloc(&target_ptr, size*sizeof(f32));
				cudaMemset(target_ptr, 0.0f, size*sizeof(f32));
		}
		else {
				target_ptr = (f32*)malloc(size*sizeof(f32));
				memset(target_ptr, 0.0f, size*sizeof(f32));
		}

		return target_ptr;
}

void _free(f32* ptr, device dev) {
		if (dev == CUDA)
				cudaFree(ptr);
		else
				free(ptr);
}

f32* _move_to(f32* data, u32 size, device target) {
		f32* target_ptr;
		if (target == CUDA) {
				cudaMalloc(&target_ptr, size*sizeof(f32));
				cudaMemcpy(target_ptr, data, size*sizeof(f32), cudaMemcpyHostToDevice);
				free(data);
		}
		else {
				target_ptr = (f32*)malloc(size*sizeof(f32));
				cudaMemcpy(target_ptr, data, size*sizeof(f32), cudaMemcpyDeviceToHost);
				cudaFree(data);
		}

		return target_ptr;
}

void _check_same_device(std::vector<device> devices) {
		device first_device = devices[0];
		b32 check = true;
		for (u32 i = 1; i < devices.size(); i++) {
				check &= (first_device == devices[i]);
		}

		
		assert(check && "Device mismatch");
}
