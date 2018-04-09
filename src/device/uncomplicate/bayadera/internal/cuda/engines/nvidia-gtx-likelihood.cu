extern "C" {

    #include <stdint.h>
    
    __global__ void loglik(const uint32_t n, const uint32_t dim,
                           const REAL* params, const REAL* x, REAL* res) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = dim * gid;
            res[gid] = LOGLIK(params, x + start);
        }
    }

    __global__ void lik(const uint32_t n, const uint32_t dim,
                        const REAL* params, const REAL* x, REAL* res) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = dim * gid;
            res[gid] = exp(LOGLIK(params, x + start));
        }
    }
    
    __global__ void evidence_reduce(const uint32_t n, const uint32_t dim,
                                    ACCUMULATOR* x_acc, const REAL* params, const REAL* x) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t start = dim * gid;
        const ACCUMULATOR sum = block_reduction_sum((gid < n) ? exp(LOGLIK(params, x + start)) : 0.0f);
        if (threadIdx.x == 0) {
            x_acc[blockIdx.x] = sum;
        }
        
    }
}
