SOURCES = ./main.cpp ./src/utils.cpp ./src/tensor.cpp ./src/modules.cpp ./src/optimizer.cpp ./read_ubyte.cpp;


build: $(SOURCES)
	@g++ -o main.exe $(SOURCES) 

clean:
	rm ./main
