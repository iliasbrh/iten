#include "device_memory.cuh"

f32* _alloc(u32 size, device_t target) {
		f32* target_ptr;
		if (target == CUDA) {
				cudaMalloc(&target_ptr, size*sizeof(f32));
				cudaMemset(target_ptr, 0.0f, size*sizeof(f32));
		}
		else if (target == CPU) {
				target_ptr = (f32*)malloc(size*sizeof(f32));
				memset(target_ptr, 0.0f, size*sizeof(f32));
		}
		else std::cout << "Invalid device requested" << std::endl;

		return target_ptr;
}

void _free(f32* ptr, device_t device) {
		if (device == CUDA)
				cudaFree(ptr);
		else
				free(ptr);
}

f32* _move_to(f32* data, u32 size, device_t target) {
		// will crash anyway if the data is on the same device as the target requested
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

void _check_same_device(std::vector<device_t> devices) {
		device_t first_device = devices[0];
		b32 check = true;
		for (u32 i = 1; i < devices.size(); i++) {
				check &= (first_device == devices[i]);
		}

		
		assert(check && "Device mismatch");
}
