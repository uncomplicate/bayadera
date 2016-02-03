#ifndef WGS
#define WGS 256
#endif


__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void pdf(__constant const float* params
                  __attribute__ ((max_constant_size(3))),
                  __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = native_exp(binomial_logpdf(params, x[gid]) + params[3]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void logpdf(__constant const float* params
                     __attribute__ ((max_constant_size(3))),
                     __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = binomial_logpdf(params, x[gid]) + params[3];
}
