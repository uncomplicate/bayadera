#ifndef WGS
#define WGS 256
#endif

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void pdf(__constant const float* params
                  __attribute__ ((max_constant_size(2))),
                  __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = uniform_pdf(params[0], params[1], x[gid]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void logpdf(__constant const float* params
                  __attribute__ ((max_constant_size(2))),
                  __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = uniform_log(params[0], params[1], x[gid]);
}
