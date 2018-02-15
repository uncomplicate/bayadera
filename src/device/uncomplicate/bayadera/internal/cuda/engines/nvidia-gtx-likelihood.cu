extern "C" {

    __global__ void loglik(const int n,  const REAL* params, const REAL* x, REAL* res) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int start = DIM * gid;
            REAL px[DIM];
            for (int i = 0; i < DIM; i++) {
                px[i] = x[start + i];
            }
            res[gid] = LOGLIK(params, px);
        }
    }

    __global__ void lik(const int n, const REAL* params, const REAL* x, REAL* res) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int start = DIM * gid;
            REAL px[DIM];
            for (int i = 0; i < DIM; i++) {
                px[i] = x[start + i];
            }
            res[gid] = exp(LOGLIK(params, px));
        }
    }

    __global__ void evidence_reduce(const int n, ACCUMULATOR* x_acc, const REAL* params, const REAL* x) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        const int start = DIM * gid;
        if (gid < n) {
            REAL px[DIM];
            for (int i = 0; i < DIM; i++) {
                px[i] = x[start + i];
            }
            const ACCUMULATOR sum = block_reduction_sum(exp(LOGLIK(params, px)));
            if (threadIdx.x == 0) {
                x_acc[blockIdx.x] = sum;
            }
        }
    }
}
