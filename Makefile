SOURCES = ./main.cpp ./src/maths.cpp ./src/cuda_maths.cu ./src/device_memory.cu ./src/functional.cpp ./src/tensor.cpp ./src/modules.cpp ./src/optimizer.cpp ./read_ubyte.cpp;

build_nvcc: $(SOURCES)
	@nvcc -o main.exe -O3 $(SOURCES)

debug_nvcc: $(SOURCES)
	@nvcc -g -o main.exe $(SOURCES) 

debug : $(SOURCES)
	@g++ -std=c++20 -O0 -g3 -fno-omit-frame-pointer \
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror \
    -fsanitize=address,undefined \
	-o main.exe $(SOURCES) 

clean:
	rm ./main.exe
