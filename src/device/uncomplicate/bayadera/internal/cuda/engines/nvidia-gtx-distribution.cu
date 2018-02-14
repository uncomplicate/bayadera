extern "C" {
    
    __global__ void logpdf (const int n, const REAL* params, const REAL* x, REAL* res) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int start = DIM * gid;
            REAL px[DIM];
            for (int i = 0; i < DIM; i++) {
                px[i] = x[start + i];
            }
            res[gid] = LOGPDF(params, px);
        }
    }

    __global__ void pdf (const int n, const REAL* params, const REAL* x, REAL* res) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int start = DIM * gid;
            REAL px[DIM];
            for (int i = 0; i < DIM; i++) {
                px[i] = x[start + i];
            }
            res[gid] = exp(LOGPDF(params, px));
        }
    }
}
