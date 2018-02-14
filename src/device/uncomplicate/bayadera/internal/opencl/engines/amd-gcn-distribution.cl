__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void logpdf(__global const REAL* params,
                     __global const REAL* x, __global REAL* res) {
    const uint start = DIM * get_global_id(0);
    REAL px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[start + i];
    }
    res[get_global_id(0)] = LOGPDF(params, px);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void pdf(__global const REAL* params,
                  __global const REAL* x, __global REAL* res) {

    const uint start = DIM * get_global_id(0);
    REAL px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[start + i];
    }
    res[get_global_id(0)] = native_exp(LOGPDF(params, px));
}
