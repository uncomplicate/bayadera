#ifndef WGS
#define WGS 256
#endif

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void loglik(__constant const REAL* params
                     __attribute__ ((max_constant_size(PARAMS_SIZE))),
                     __global const REAL* x, __global REAL* res) {

    uint start = DIM * get_global_id(0);
    REAL px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[start + i];
    }
    res[get_global_id(0)] = LOGLIK(params, px);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void lik(__constant const REAL* params
                  __attribute__ ((max_constant_size(PARAMS_SIZE))),
                  __global const REAL* x, __global REAL* res) {

    uint start = DIM * get_global_id(0);
    REAL px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[start + i];
    }
    res[get_global_id(0)] = native_exp(LOGLIK(params, px));
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_reduction (__global ACCUMULATOR* acc) {
    ACCUMULATOR sum = work_group_reduction_sum(acc[get_global_id(0)]);
    if (get_local_id(0) == 0) {
        acc[get_group_id(0)] = sum;
    }
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void evidence_reduce(__global ACCUMULATOR* x_acc,
                              __constant const REAL* params
                              __attribute__ ((max_constant_size(PARAMS_SIZE))),
                              __global const REAL* x) {

    uint start = DIM * get_global_id(0);
    REAL px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[start + i];
    }
    ACCUMULATOR sum = work_group_reduction_sum(native_exp(LOGLIK(params, px)));
    if (get_local_id(0) == 0) {
        x_acc[get_group_id(0)] = sum;
    }

}
