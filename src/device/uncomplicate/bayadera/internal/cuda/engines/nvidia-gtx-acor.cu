extern "C" {

#include <stdint.h>

// ======================== Acor =====================================

    __global__ void sum_pairwise(const uint32_t n,
                                 const uint32_t stride,
                                 const uint32_t offset,
                                 REAL* x){

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            x[offset + 2 * gid * stride] += x[offset + (2 * gid + 1) * stride];
        }
            
    };

    __global__ void acor_1d (const uint32_t n,
                                       const uint32_t stride,
                                       const uint32_t dim_id,
                                       const uint32_t lag,
                                       REAL* c0acc,
                                       REAL* dacc,
                                       REAL* means) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t local_size = blockDim.x;
        const uint32_t local_id = threadIdx.x;

        const bool load_lag = (threadIdx.x < lag) && (blockIdx.x + 1 < gridDim.x);
        const bool compute = gid + lag < n;

        const REAL x = compute ? means[gid * stride + dim_id] : 0.0f;
        const REAL x_load = load_lag ? means[(gid + local_size) * stride + dim_id] : 0.0f;
        
        __shared__ REAL local_means[2 * WGS];
        
        local_means[local_id] = x;
        local_means[local_id + local_size] = x_load;

        __syncthreads();

        REAL xacc = 0.0f;
        for (uint32_t s = 0; s < lag; s++) {
            xacc += local_means[local_id + s + 1];
        }
        xacc = x * (x + 2.0f * xacc);

        __syncthreads();

        REAL* lc0 = local_means;
        REAL* ld = local_means + local_size;
        REAL pc0 = x * x;
        lc0[local_id] = pc0;
        ld[local_id] = xacc;

        __syncthreads();

        uint32_t i = local_size;
        while (i > 0) {
            const bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
            i >>= 1;
            if (include_odd) {
                pc0 += lc0[local_id + i + 1];
                xacc += ld[local_id + i + 1];
            }
            if (local_id < i) {
                pc0 += lc0[local_id + i];
                lc0[local_id] = pc0;
                xacc += ld[local_id + i];
                ld[local_id] = xacc;
            }
            __syncthreads();
        }

        if (local_id == 0) {
            atomicAdd(&c0acc[dim_id], pc0);
            atomicAdd(&dacc[dim_id], xacc);
        }

    }
    
    __global__ void acor_2d (const uint32_t n,
                                       const uint32_t dim,
                                       const uint32_t lag,
                                       REAL* c0acc,
                                       REAL* dacc,
                                       REAL* means) {

        const uint32_t dim_id = blockIdx.y * blockDim.y + threadIdx.y;
        acor_1d(n, dim, dim_id, lag, c0acc, dacc, means);
    }

    __global__ void acor (const uint32_t n,
                                    const uint32_t dim,
                                    const uint32_t lag,
                                    const uint32_t min_lag,
                                    const uint32_t win_mult,
                                    REAL* c0acc,
                                    REAL* dacc,
                                    REAL* means) {

        uint32_t dim_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (dim_id == 0) {
            const uint32_t block_dim = (WGS < n) ? WGS : n;
            const uint32_t grid_dim = (n - 1) / block_dim + 1;
            dim3 blocks(block_dim, 1, 1); 
            dim3 grids(grid_dim, dim, 1);
            acor_2d<<<grids, blocks>>>(n, dim, lag, c0acc, dacc, means);
            cudaDeviceSynchronize();
        }
        __syncthreads();

        while (dim_id < dim) {
            const REAL c0 = c0acc[dim_id];
            REAL tau = dacc[dim_id] / c0;
            cudaStream_t hstream;
            cudaStreamCreateWithFlags(&hstream, cudaStreamNonBlocking);
            uint32_t lag2 = lag;
            uint32_t n2 = n;
            uint32_t stride = 1;
            while ((min_lag < lag2) && (lag2 < tau * win_mult)) {
                n2 /= 2;
                const uint32_t block_dim = (WGS < n2) ? WGS : n2;
                const uint32_t grid_dim = (n2 - 1) / block_dim + 1;
                lag2 = ((lag * win_mult) < n2) ? lag : (n2 / win_mult);
                const REAL c0 = c0acc[dim_id];
                c0acc[dim_id] = 0.0;
                dacc[dim_id] = 0.0;
                sum_pairwise<<<grid_dim, block_dim, 0, hstream>>>(n2, stride * dim, dim_id, means);
                stride *= 2;
                acor_1d<<<grid_dim, block_dim, 0, hstream>>>
                    (n2, stride * dim, dim_id, lag2, c0acc, dacc, means);
                cudaDeviceSynchronize();
                tau = dacc[dim_id] / c0;
            }
            cudaStreamDestroy(hstream);
            const REAL scale = stride * (REAL)(n2 - lag2);
            c0acc[dim_id] = dacc[dim_id] * (n - lag) / (scale * c0);                
            dacc[dim_id] = sqrt(dacc[dim_id] / (scale * n));

            dim_id += blockDim.x;
        }
                
    }

}
