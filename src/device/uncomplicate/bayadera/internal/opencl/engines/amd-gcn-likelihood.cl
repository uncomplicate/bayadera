__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void loglik(const uint params_len, __global const REAL* params,
                     const uint dim, __global const REAL* x, __global REAL* res) {

    const uint start = dim * get_global_id(0);
    res[get_global_id(0)] = LOGLIK(params_len, params, dim, x + start);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void lik(const uint params_len, __global const REAL* params,
                  const uint dim, __global const REAL* x, __global REAL* res) {

    const uint start = dim * get_global_id(0);
    res[get_global_id(0)] = native_exp(LOGLIK(params_len, params, dim, x + start));
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void evidence_reduce(__global ACCUMULATOR* x_acc,
                              const uint params_len, __global const REAL* params,
                              const uint dim, __global const REAL* x) {

    const uint start = dim * get_global_id(0);
    const ACCUMULATOR sum = work_group_reduction_sum(native_exp(LOGLIK(params_len, params, dim, x + start)));
    if (get_local_id(0) == 0) {
        x_acc[get_group_id(0)] = sum;
    }

}
