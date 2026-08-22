#include <stdint.h>
static uint64_t       state      = 0x4d595df4d0f33173;
static uint64_t const multiplier = 6364136223846793005u;
static uint64_t const increment  = 1442695040888963407u;

#define MAX_RAND (~ (1 << 31))

static uint32_t rotr32(uint32_t x, unsigned r);
uint32_t pcg32(void);
void pcg32_init(uint64_t seed);
int32_t randint(int32_t a, int32_t b); // b not included
float random(void);
