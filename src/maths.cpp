#include "maths.hpp"

u64 multiply_vector(const std::vector<u64> &vec) {
		u64 result = 1;
		for (const u64 element : vec) {result *= element;}
		return result;
}

void mat_add(f32* U, f32* V, f32* out, u64 M, u64 N) {
		for (u64 row=0; row<M; row++)
				for (u64 col=0; col<N; col++) {
						out[row*N+col] = U[row*N+col] + V[row*N+col];
				}
}

void mat_mul(f32* A, f32* B, f32* out, u64 M, u64 N, u64 P, b32 zero_out) {
		for (u64 row=0; row<M; row++)
				for (u64 col=0; col<P; col++) {
						if (zero_out) out[row*P+col] = 0;
						for (u64 k=0; k<N; k++)
								out[row*P+col] += A[row*N+k] * B[k*P+col];
				}
}

void mat_mul_transpose_A(f32* A, f32* B, f32* out, u64 M, u64 N, u64 P, b32 zero_out) {
		// A of shape (N, M), B of shape (N, P), out of shape (M, P)
		for (u64 row=0; row<M; row++)
				for (u64 col=0; col<P; col++) {
						if (zero_out) out[row*P+col] = 0;
						for (u64 k=0; k<N; k++)
								out[row*P+col] += A[k*M+row] * B[k*P+col];
				}
}

void mat_mul_transpose_B(f32* A, f32* B, f32* out, u64 M, u64 N, u64 P, b32 zero_out) {
		// A of shape (M, N), B of shape (P, N), out of shape (M, P)
		for (u64 row=0; row<M; row++)
				for (u64 col=0; col<P; col++) {
						if (zero_out) out[row*P+col] = 0;
						for (u64 k=0; k<N; k++)
								out[row*P+col] += A[k*M+row] * B[col*N+k];
				}
}

void mat_exp(f32* A, f32* out, u64 size) {
		for (u64 i=0; i<size; i++)
				out[i] = expf(A[i]);
}

f32 mat_sum(f32* A, u64 size) {
		f32 res = 0;
		for (u64 i=0; i<size; i++)
				res += A[i];
		return res;
}

void mat_scale(f32* A, f32* out, f32 scaler, u64 size) {
		for (u64 i=0; i<size; i++)
				out[i] = A[i] * scaler;
}






















