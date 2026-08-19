SOURCES = ./main.cpp ./src/maths.cpp ./src/cuda_maths.cu ./src/device_memory.cu ./src/functional.cu ./src/tensor.cpp ./src/modules.cpp ./src/optimizer.cpp ./read_ubyte.cpp;

build_nvcc: $(SOURCES)
	@nvcc -std=c++20 -O3 -lineinfo -DNDEBUG \
    --all-warnings \
    -Xcompiler -O3,-Wall,-Wextra,-flto \
    -arch=native \
    -o cuda_release $(SOURCES)

debug_nvcc: $(SOURCES)
	@nvcc -std=c++20 -g -G -lineinfo \
    --all-warnings -Werror=all-warnings \
    -Xcompiler -Wall,-Wextra,-Wpedantic,-Werror,-Wshadow,-Wconversion,-rdynamic \
    -fdevice-sanitize=memcheck \
    -o cuda_debug $(SOURCES) 

clean:
	rm ./main.exe
