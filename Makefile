SOURCES = ./main.cpp ./src/utils.cpp ./src/tensor.cpp ./src/modules.cpp ./src/optimizer.cpp ./read_ubyte.cpp;

build: $(SOURCES)
	@g++ -o main.exe $(SOURCES)

debug: $(SOURCES)
	@g++ -std=c++20 -O0 -g3 -fno-omit-frame-pointer \
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror \
    -fsanitize=address,undefined \
	-o main.exe $(SOURCES) 

clean:
	rm ./main.exe
