#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <string.h>


typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
typedef double f64;

typedef int8_t b8;
typedef int32_t b32;


// importing c++ utils
#include <vector>
#include <memory>
#include <queue>
#include <iostream>
#include <cmath>
#include <chrono>
#include "stdlib.h"

#include "cuda_runtime.h"


// execution time macro

#define TIME_INSTRUCTION(instruction, text) do { \
	auto start = std::chrono::high_resolution_clock::now(); \
	instruction \
	auto end = std::chrono::high_resolution_clock::now(); \
	std::chrono::duration<double, std::milli> elapsed = end - start; \
	std::cout << text << " : " << elapsed.count() << " ms\n"; \
} while(0)


#endif
