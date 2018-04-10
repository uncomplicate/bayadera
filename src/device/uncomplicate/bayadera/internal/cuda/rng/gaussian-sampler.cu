extern "C" {

#include "Random123/philox.h"
#include <stdint.h>
    
#ifndef M_2PI_F
#define M_2PI_F 6.2831855f
#endif

#ifndef R123_0x1p_23f
#define R123_0x1p_23f 1.1920928955078125E-7f
#endif

    inline float u01fpt_oo_32_24(uint32_t i) {
        return (0.5f + (i >> 9)) * R123_0x1p_23f;
    }

//Sampling from the Gaussian distribution
    inline float4 box_muller(float4 uniform) {
        float4 result;
        result.x = sin(M_2PI_F * uniform.x) * sqrt(-2.0f * log(uniform.y));
        result.y = cos(M_2PI_F * uniform.x) * sqrt(-2.0f * log(uniform.y));
        result.z = sin(M_2PI_F * uniform.z) * sqrt(-2.0f * log(uniform.w));
        result.w = cos(M_2PI_F * uniform.z) * sqrt(-2.0f * log(uniform.w));
        return result;
    }

    __global__ void sample (const uint32_t n, const REAL* params, const uint32_t seed, float4* x) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            // Generate uniform(0,1) floats
            philox4x32_key_t key;
            uint32_t* key_v = key.v;
            key_v[0] = seed;
            key_v[1] = 0xdecafaaa;
            key_v[2] = 0xfacebead;
            key_v[3] = 0x12345678;
            philox4x32_ctr_t cnt;
            uint32_t* cnt_v = cnt.v;
            cnt_v[0] = gid;
            cnt_v[1] = 0xf00dcafe;
            cnt_v[2] = 0xdeadbeef;
            cnt_v[3] = 0xbeeff00d;

            const REAL mu = params[0];
            const REAL sigma = params[1];

            uint32_t* rand_uni = philox4x32(cnt, key).v;
            float4 result;
            result.x = u01fpt_oo_32_24(rand_uni[0]);
            result.y = u01fpt_oo_32_24(rand_uni[1]);
            result.z = u01fpt_oo_32_24(rand_uni[2]);
            result.w = u01fpt_oo_32_24(rand_uni[3]);
            result = box_muller(result);
            result.x = result.x * sigma + mu;
            result.y = result.y * sigma + mu;
            result.z = result.z * sigma + mu;
            result.w = result.w * sigma + mu;
            x[gid] = result;
        }
    }
}
