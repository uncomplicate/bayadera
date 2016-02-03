#include "Random123/philox.h"

#ifndef WGS
#define WGS 256
#endif

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void pdf(__constant const float* params
                  __attribute__ ((max_constant_size(2))),
                  __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = gaussian_pdf(params[0], params[1], x[gid]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void logpdf(__constant const float* params
                     __attribute__ ((max_constant_size(2))),
                     __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = gaussian_logpdf(params, x[gid]);
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
