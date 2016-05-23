#ifndef WGS
#define WGS 256
#endif

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void loglik(__constant const float* params
                     __attribute__ ((max_constant_size(PARAMS_SIZE))),
                     __global const float* x, __global float* res) {

    uint gid = DIM * get_global_id(0);
    float px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[gid + i];
    }
    res[gid] = LOGLIK(params, px);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void lik(__constant const float* params
                  __attribute__ ((max_constant_size(PARAMS_SIZE))),
                  __global const float* x, __global float* res) {

    uint gid = DIM * get_global_id(0);
    float px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[gid + i];
    }
    res[gid] = native_exp(LOGLIK(params, px));
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_reduction (__global double* acc) {
    double sum = work_group_reduction_sum(acc[get_global_id(0)]);
    if (get_local_id(0) == 0) {
        acc[get_group_id(0)] = sum;
    }
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void evidence_reduce(__global double* x_acc,
                              __constant const float* params
                              __attribute__ ((max_constant_size(PARAMS_SIZE))),
                              __global const float* x) {

    uint gid = DIM * get_global_id(0);
    float px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[gid + i];
    }
    double sum = work_group_reduction_sum(native_exp(LOGLIK(params, px)));
    if (get_local_id(0) == 0) {
        x_acc[get_group_id(0)] = sum;
    }

}
