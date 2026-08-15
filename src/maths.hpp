#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <cmath>
#include "types.h"


u64 multiply_vector(const std::vector<u64> &vec); // product of vector's elements

void mat_add(f32* U, f32* V, f32* out, u64 M, u64 N);
void mat_mul(f32* A, f32* B, f32* out, u64 M, u64 N, u64 P, b32 zero_out = false);

// for matmul's backward pass
void mat_mul_transpose_A(f32* A, f32* B, f32* out, u64 M, u64 N, u64 P, b32 zero_out = false);
void mat_mul_transpose_B(f32* A, f32* B, f32* out, u64 M, u64 N, u64 P, b32 zero_out = false);

// element-wise operations (matrix shape not needed as parameter)
void mat_exp(f32* A, f32* out, u64 size);
f32 mat_sum(f32* A, u64 size);
void mat_scale(f32* A, f32* out, f32 scaler, u64 size);



#endif
