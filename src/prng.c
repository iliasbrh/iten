#include "prng.h"

static uint32_t rotr32(uint32_t x, unsigned r)
{
	return x >> r | x << (-r & 31);
}

uint32_t pcg32(void)
{
	uint64_t x = state;
	unsigned count = (unsigned)(x >> 59);

	state = x * multiplier + increment;
	x ^= x >> 18;
	return rotr32((uint32_t)(x >> 27), count);
}

void pcg32_init(uint64_t seed)
{
	state = seed + increment;
	(void)pcg32();
}

int32_t randint(int32_t a, int32_t b) // b not included
{
		int32_t delta = b - a;
		int32_t rnd = pcg32() % delta;

		return rnd + a;
}

float random(void)
{
		return 0.5f * (float)pcg32() / (float)MAX_RAND;
}
