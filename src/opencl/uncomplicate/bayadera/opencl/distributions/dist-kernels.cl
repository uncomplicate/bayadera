#ifndef WGS
#define WGS 256
#endif

#ifndef PARAMS_SIZE
#define PARAMS_SIZE 2
#endif

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void logpdf(__constant const REAL* params
                     __attribute__ ((max_constant_size(PARAMS_SIZE))),
                     __global const REAL* x, __global REAL* res) {
    uint start = DIM * get_global_id(0);
    REAL px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[start + i];
    }
    res[get_global_id(0)] = LOGPDF(params, px);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void pdf(__constant const REAL* params
                  __attribute__ ((max_constant_size(PARAMS_SIZE))),
                  __global const REAL* x, __global REAL* res) {

    uint start = DIM * get_global_id(0);
    REAL px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[start + i];
    }
    res[get_global_id(0)] = native_exp(LOGPDF(params, px));
}
