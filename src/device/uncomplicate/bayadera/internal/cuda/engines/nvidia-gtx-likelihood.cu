extern "C" {

    #include <stdint.h>
    
    __global__ void loglik(const uint32_t n,
                           const uint32_t params_len, const REAL* params,
                           const uint32_t dim, const REAL* x, REAL* res) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = dim * gid;
            res[gid] = LOGLIK(params_len, params, dim, x + start);
        }
    }

    __global__ void lik(const uint32_t n,
                        const uint32_t params_len, const REAL* params,
                        const uint32_t dim, const REAL* x, REAL* res) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = dim * gid;
            res[gid] = exp(LOGLIK(params_len, params, dim, x + start));
        }
    }
    
    __global__ void evidence_reduce(const uint32_t n,
                                    ACCUMULATOR* x_acc,
                                    const uint32_t params_len, const REAL* params,
                                    const uint32_t dim, const REAL* x) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t start = dim * gid;
        const ACCUMULATOR sum =
            block_reduction_sum((gid < n) ? exp(LOGLIK(params_len, params, dim, x + start)) : 0.0f);
        if (threadIdx.x == 0) {
            x_acc[blockIdx.x] = sum;
        }
        
    }
}
