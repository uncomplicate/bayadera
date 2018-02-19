extern "C" {

    #include <stdint.h>
    
    __global__ void loglik(const uint32_t n,  const REAL* params, const REAL* x, REAL* res) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = DIM * gid;
            res[gid] = LOGLIK(params, x + start);
        }
    }

    __global__ void lik(const uint32_t n, const REAL* params, const REAL* x, REAL* res) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = DIM * gid;
            res[gid] = exp(LOGLIK(params, x + start));
        }
    }
    
    __global__ void evidence_reduce(const uint32_t n, ACCUMULATOR* x_acc, const REAL* params, const REAL* x) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t start = DIM * gid;
        const ACCUMULATOR sum = block_reduction_sum((gid < n) ? exp(LOGLIK(params, x + start)) : 0.0f);
        if (threadIdx.x == 0) {
            x_acc[blockIdx.x] = sum;
        }
        
    }
}
