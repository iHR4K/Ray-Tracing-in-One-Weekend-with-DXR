// CODE FROM: 
// https://github.com/GPSnoopy/RayTracingInVulkan
// Generates a seed for a random number generator from 2 inputs plus a backoff
// https://github.com/nvpro-samples/optix_prime_baking/blob/332a886f1ac46c0b3eea9e89a59593470c755a0e/random.h
// https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/tree/master/ray_tracing_jitter_cam
// https://en.wikipedia.org/wiki/Tiny_Encryption_Algorithm
uint InitRandomSeed(uint val0, uint val1)
{
    uint v0 = val0, v1 = val1, s0 = 0;
    
    [unroll]
    for (uint n = 0; n < 16; n++)
    {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

uint RandomInt(inout uint seed)
{
	// LCG values from Numerical Recipes
    return (seed = 1664525 * seed + 1013904223);
}

float RandomFloat(inout uint seed)
{
	//// Float version using bitmask from Numerical Recipes
	//const uint one = 0x3f800000;
	//const uint msk = 0x007fffff;
	//return uintBitsToFloat(one | (msk & (RandomInt(seed) >> 9))) - 1;

	// Faster version from NVIDIA examples; quality good enough for our use case.
    return (float(RandomInt(seed) & 0x00FFFFFF) / float(0x01000000));
}

float3 SampleSquare(uint seed)
{
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    return float3(RandomFloat(seed) - 0.5, RandomFloat(seed) - 0.5, 0);
}

float2 RandomInUnitDisk(inout uint seed)
{
    for (;;)
    {
        const float2 p = 2 * float2(RandomFloat(seed), RandomFloat(seed)) - 1;
        if (dot(p, p) < 1)
        {
            return p;
        }
    }
}

float3 RandomInUnitSphere(inout uint seed)
{
    for (;;)
    {
        const float3 p = normalize(2 * float3(RandomFloat(seed), RandomFloat(seed), RandomFloat(seed)) - 1);
        if (dot(p, p) < 1)
        {
            return p;
        }
    }
}