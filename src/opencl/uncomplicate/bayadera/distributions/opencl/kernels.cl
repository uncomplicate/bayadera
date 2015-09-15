#ifndef WGS
#define WGS 256
#endif

#ifndef PARAMS_SIZE
#define PARAMS_SIZE 2
#endif

#ifndef DIST_PDF
#define DIST_PDF unknown
#endif


#ifndef DIST_LOGPDF
#define DIST_LOGPDF unknown
#endif

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void logpdf_kernel(__constant const float* params
                            __attribute__ ((max_constant_size(PARAMS_SIZE))),
                            __global const float* x, __global float* res) {
    uint gid = get_global_id(0);
    res[gid] = DIST_LOGPDF(params, x[gid]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void pdf_kernel(__constant const float *params
                         __attribute__ ((max_constant_size(PARAMS_SIZE))),
                         __global const float* x, __global float* res) {

    uint gid = get_global_id(0);
    res[gid] = DIST_PDF(params, x[gid]);
}
