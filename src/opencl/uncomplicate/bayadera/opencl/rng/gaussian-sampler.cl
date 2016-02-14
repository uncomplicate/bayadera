#include "Random123/philox.h"

#ifndef WGS
#define WGS 256
#endif

#ifndef M_2PI_F
#define M_2PI_F 6.2831855f
#endif

#ifndef R123_0x1p_23f
#define R123_0x1p_23f 1.1920928955078125E-7f
#endif

inline float4 u01fpt_oo_4x32_24(uint4 i) {
    return (float4)((0.5f + (i.x >> 9)) * R123_0x1p_23f,
                    (0.5f + (i.y >> 9)) * R123_0x1p_23f,
                    (0.5f + (i.z >> 9)) * R123_0x1p_23f,
                    (0.5f + (i.w >> 9)) * R123_0x1p_23f);
}

//Sampling from the Gaussian distribution
inline float4 box_muller(float4 uniform) {
    return (float4)(native_sin(M_2PI_F * uniform.x)
                    * sqrt(-2.0f * native_log(uniform.y)),
                    native_cos(M_2PI_F * uniform.x)
                    * sqrt(-2.0f * native_log(uniform.y)),
                    native_sin(M_2PI_F * uniform.z)
                    * sqrt(-2.0f * native_log(uniform.w)),
                    native_cos(M_2PI_F * uniform.z)
                    * sqrt(-2.0f * native_log(uniform.w)));
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sample( __constant const float* params
                      __attribute__ ((max_constant_size(2))),
                      const uint seed, __global float4* x) {

    uint gid = get_global_id(0);
    // Generate uniform(0,1) floats
    philox4x32_key_t key = {{seed, 0xdecafaaa, 0xfacebead, 0x12345678}};
    philox4x32_ctr_t cnt = {{gid, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};

    x[gid] = box_muller(u01fpt_oo_4x32_24(((uint4*)philox4x32(cnt, key).v)[0]))
        * params[1] + params[0];

}
