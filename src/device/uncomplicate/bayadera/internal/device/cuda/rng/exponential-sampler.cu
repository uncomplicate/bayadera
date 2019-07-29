extern "C" {
    
#include "Random123/philox.h"
#include <stdint.h>
    
#ifndef R123_0x1p_23f
#define R123_0x1p_23f 1.1920928955078125E-7f
#endif

// Sampling from the uniform distribution
    inline float u01fpt_oo_32_24(uint32_t i) {
        return (0.5f + (i >> 9)) * R123_0x1p_23f;
    }

    __global__ void sample (const uint32_t n, const REAL* params, const uint32_t seed,
                            float4* x, const uint32_t offset_x) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid * 4 < n) {
            // Generate uniform(0,1) floats
            philox4x32_key_t key;
            uint32_t* key_v = key.v;
            key_v[0] = seed;
            key_v[1] = 0xdecafaaa;
            philox4x32_ctr_t cnt;
            uint32_t* cnt_v = cnt.v;
            cnt_v[0] = gid;
            cnt_v[1] = 0xf00dcafe;
            cnt_v[2] = 0xdeadbeef;
            cnt_v[3] = 0xbeeff00d;

            const REAL lambda = params[0];

            uint32_t* rand_uni = philox4x32(cnt, key).v;
            float4 result;
            result.x = -1.0f / lambda * log(1.0f - u01fpt_oo_32_24(rand_uni[0]));
            result.y = -1.0f / lambda * log(1.0f - u01fpt_oo_32_24(rand_uni[1]));
            result.z = -1.0f / lambda * log(1.0f - u01fpt_oo_32_24(rand_uni[2]));
            result.w = -1.0f / lambda * log(1.0f - u01fpt_oo_32_24(rand_uni[3]));
            x[offset_x + gid] = result;
        }
    }
}
