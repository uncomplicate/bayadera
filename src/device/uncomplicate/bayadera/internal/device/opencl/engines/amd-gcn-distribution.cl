__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void logpdf(const uint data_len, const uint params_len, __global const REAL* params,
                     const uint dim, __global const REAL* x, __global REAL* res) {
    const uint start = dim * get_global_id(0);
    res[get_global_id(0)] = LOGPDF(data_len, params_len, params, dim, x + start);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void pdf(const uint data_len, const uint params_len, __global const REAL* params,
                  const uint dim, __global const REAL* x, __global REAL* res) {

    const uint start = dim * get_global_id(0);
    res[get_global_id(0)] = native_exp(LOGPDF(data_len, params_len, params, dim, x + start));
}
