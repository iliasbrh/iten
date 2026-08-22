#include <stdint.h>
#include "types.hpp"

#define MAX_RAND (~ (1 << 31))

uint32_t pcg32(void);
void pcg32_init(uint64_t seed);
int32_t randint(int32_t a, int32_t b); // b not included
float rand_uniform(void);
