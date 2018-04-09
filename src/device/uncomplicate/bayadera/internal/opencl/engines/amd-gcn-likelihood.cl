__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void loglik(const uint dim, __global const REAL* params,
                     __global const REAL* x, __global REAL* res) {

    const uint start = dim * get_global_id(0);
    res[get_global_id(0)] = LOGLIK(params, x + start);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void lik(const uint dim, __global const REAL* params,
                  __global const REAL* x, __global REAL* res) {

    const uint start = dim * get_global_id(0);
    res[get_global_id(0)] = native_exp(LOGLIK(params, x + start));
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void evidence_reduce(const uint dim, __global ACCUMULATOR* x_acc,
                              __global const REAL* params, __global const REAL* x) {

    const uint start = dim * get_global_id(0);
    ACCUMULATOR sum = work_group_reduction_sum(native_exp(LOGLIK(params, x + start)));
    if (get_local_id(0) == 0) {
        x_acc[get_group_id(0)] = sum;
    }

}
