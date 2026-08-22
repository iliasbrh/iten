SOURCES = ./main.cpp \
		  ./src/maths.cpp ./src/cuda_maths.cu \
		  ./src/device_memory.cu ./src/functional.cu \
		  ./src/tensor.cpp \
		  ./src/modules.cpp ./src/optimizer.cpp \
		  ./read_ubyte.cpp ./model.cpp \
		  ./src/prng.c;

build_nvcc: $(SOURCES)
	@nvcc -std=c++20 -O3 -lineinfo -DNDEBUG \
    --all-warnings \
    -Xcompiler -O3,-Wall,-Wextra,-flto \
    -arch=native \
    -o cuda_release $(SOURCES)

debug_nvcc: $(SOURCES)
	@nvcc -std=c++20 -g -G \
    -Werror=all-warnings \
    -Xcompiler -Wall,-Wextra,-Werror,-Wshadow,-Wconversion,-rdynamic \
    -o cuda_debug $(SOURCES) 

clean:
	@rm ./cuda_debug > /dev/null 2>&1
	@rm ./main.exe > /dev/null 2>&1
	@rm ./cuda_release > /dev/null 2>&1
