extern "C" {

#include <stdint.h>
    
    __global__ void logpdf (const uint32_t n, 
                            const uint32_t params_len, const REAL* params,
                            const uint32_t dim, const REAL* x, REAL* res) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = dim * gid;
            res[gid] = LOGPDF(params_len, params, dim, x + start);
        }
    }

    __global__ void pdf (const uint32_t n,
                         const uint32_t params_len, const REAL* params,
                         const uint32_t dim, const REAL* x, REAL* res) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = dim * gid;
            res[gid] = exp(LOGPDF(params_len, params, dim, x + start));
        }
    }
}
