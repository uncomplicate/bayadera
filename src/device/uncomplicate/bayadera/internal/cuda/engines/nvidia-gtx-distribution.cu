extern "C" {

#include <stdint.h>
    
    __global__ void logpdf (const uint32_t n, const REAL* params, const REAL* x, REAL* res) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = DIM * gid;
            res[gid] = LOGPDF(params, x + start);
        }
    }

    __global__ void pdf (const uint32_t n, const REAL* params, const REAL* x, REAL* res) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = DIM * gid;
            res[gid] = exp(LOGPDF(params, x + start));
        }
    }
}
