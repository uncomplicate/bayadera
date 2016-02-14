#ifndef WGS
#define WGS 256
#endif

#ifndef PARAMS_SIZE
#define PARAMS_SIZE 2
#endif

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void loglik(__constant const float* params
                     __attribute__ ((max_constant_size(PARAMS_SIZE))),
                     __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = LOGLIK(params, x[gid]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void lik(__constant const float* params
                  __attribute__ ((max_constant_size(PARAMS_SIZE))),
                  __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = native_exp(LOGLIK(params, x[gid]));
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_reduction (__global double* acc) {
    double sum = work_group_reduction_sum(acc[get_global_id(0)]);
    if (get_local_id(0) == 0) {
        acc[get_group_id(0)] = sum;
    }
}


__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void evidence_reduce(__constant const float* params
                              __attribute__ ((max_constant_size(PARAMS_SIZE))),
                              __global double* x_acc, __global const float* x) {

    float xi = x[get_global_id(0)];
    double sum = work_group_reduction_sum(1.0f / native_exp(LOGLIK(params, xi)));
    if (get_local_id(0) == 0) {
        x_acc[get_group_id(0)] = sum;
    }

}
