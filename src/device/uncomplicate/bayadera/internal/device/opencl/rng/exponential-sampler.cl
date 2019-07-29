#include "Random123/philox.h"

#ifndef R123_0x1p_23f
#define R123_0x1p_23f 1.1920928955078125E-7f
#endif

inline float4 u01fpt_oo_4x32_24(uint4 i) {
    return (float4)((0.5f + (i.x >> 9)) * R123_0x1p_23f,
                    (0.5f + (i.y >> 9)) * R123_0x1p_23f,
                    (0.5f + (i.z >> 9)) * R123_0x1p_23f,
                    (0.5f + (i.w >> 9)) * R123_0x1p_23f);

}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sample(__global const float* params, const uint seed,
                     __global float4* x, const uint offset_x) {

    const uint gid = get_global_id(0);
    // Generate uniform(0,1) floats
    const philox4x32_key_t key = {{seed, 0xdecafaaa}};
    const philox4x32_ctr_t cnt = {{gid, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};

    x[offset_x + gid] = -1.0f / params[0]
        * native_log(1.0f - u01fpt_oo_4x32_24(((uint4*)philox4x32(cnt, key).v)[0]));

}
