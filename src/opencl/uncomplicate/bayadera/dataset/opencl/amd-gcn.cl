#ifndef WGS
#define WGS 256
#endif

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void variance_reduce(__global float* acc,
                              __global const float* x, const float mu) {
    float xi = x[get_global_id(0)];
    work_group_reduction_sum(acc, (double)((xi - mu) * (xi - mu)));
}
