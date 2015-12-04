#include "Random123/philox.h"

#ifndef WGS
#define WGS 256
#endif

inline float beta_mcmc(__constant float* params, float x) {
    return beta_log(params[0], params[1], x);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void pdf(__constant const float* params
                  __attribute__ ((max_constant_size(3))),
                  __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = native_exp(beta_log(params[0], params[1], x[gid]) - params[2]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void logpdf(__constant const float* params
                  __attribute__ ((max_constant_size(3))),
                  __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = beta_log(params[0], params[1], x[gid]) - params[2];

}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sample( __constant const float* params
                      __attribute__ ((max_constant_size(2))),
                      const uint seed, __global float4* x) {
    //TODO
}
