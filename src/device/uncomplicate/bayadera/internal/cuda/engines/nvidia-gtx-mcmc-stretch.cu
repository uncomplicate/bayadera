extern "C" {

#include "Random123/philox.h"
#include <float.h>

// =========================== Stretch move ====================================

    int block_reduction_sum_int (const int value) {

        const int local_id = threadIdx.x;

        __shared__ int lacc[WGS];
        lacc[local_id] = value;

        __syncthreads();

        int pacc = value;
        int i = blockDim.x;
        while (i > 0) {
            bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
            i >>= 1;
            if (include_odd) {
                pacc += lacc[local_id + i + 1];
            }
            if (local_id < i) {
                pacc += lacc[local_id + i];
                lacc[local_id] = pacc;
            }
            __syncthreads();
        }

        return lacc[0];
    }

    bool stretch_move(const int K,
                      const int seed,
                      const REAL* params,
                      const REAL* Scompl,
                      REAL* X,
                      REAL* logfn_X,
                      const REAL a,
                      const REAL beta,
                      const int step_counter,
                      const int odd_or_even) {

        // Get the index of this walker Xk
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Generate uniform(0,1) floats
        philox4x32_key_t key;
            uint32_t* key_v = key.v;
            key_v[0] = seed;
            key_v[1] = 0xdecafaaa;
            key_v[2] = 0xfacebead;
            key_v[3] = 0x12345678;
            philox4x32_ctr_t cnt;
            uint32_t* cnt_v = cnt.v;
            cnt_v[0] = k;
            cnt_v[1] = step_counter;
            cnt_v[2] = odd_or_even;
            cnt_v[3] = 0xbeeff00d;
            uint32_t* rand_uni = philox4x32(cnt, key).v;
            
            float4 u;
            u.x = u01fpt_oo_32_24(rand_uni[0]);
            u.y = u01fpt_oo_32_24(rand_uni[1]);
            u.z = u01fpt_oo_32_24(rand_uni[2]);
            u.w = u01fpt_oo_32_24(rand_uni[3]);

        // Draw a sample from g(z) using the formula from [Christen 2007]
        const REAL z = (a - 2.0f + 1.0f / a) * u.y * u.y
            + (2.0f * (1.0f - 1.0f / a)) * u.y + 1.0f / a;

        // Draw a walker Xj's index at random from the complementary ensemble S(~i)(t)
        const int j0 = (int)(u.x * K * DIM);
        const int k0 = k * DIM;

        REAL Y[DIM];

        for (int i = 0; i < DIM; i++) {
            const REAL Xji = Scompl[j0 + i];
            Y[i] = Xji + z * (X[k0 + i] - Xji);
        }
        
        const REAL logfn_y = LOGFN(params, Y);
        const REAL q = (isfinite(logfn_y)) ?
            powf(z, DIM - 1) * exp(beta * (logfn_y - logfn_X[k])) : 0.0f;

        const bool accepted = u.z <= q;

        if (accepted) {
            for (int i = 0; i < DIM; i++) {
                X[k0 + i] = Y[i];
            }
            logfn_X[k] = logfn_y;
            return 1;
        } else {
            return 0;
        }
    }

    __global__ void stretch_move_accu(const int n,
                                      const int seed,
                                      const int odd_or_even,
                                      const REAL* params,
                                      const REAL* Scompl,
                                      REAL* X,
                                      REAL* logfn_X,
                                      int* accept,
                                      REAL* means,
                                      const REAL a,
                                      const int step_counter) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        const bool accepted = (gid < n) &&
            stretch_move(n, seed, params, Scompl, X, logfn_X, a, 1.0f, step_counter, odd_or_even);

        const int accepted_sum = block_reduction_sum_int(accepted ? 1 : 0);
        if (threadIdx.x == 0) {
            accept[blockIdx.x] += accepted_sum;
        }
        
        const int k0 = gid * DIM;
        const int id = n * step_counter * DIM;
        for (int i = 0; i < DIM; i++) {
            const REAL mean_sum = block_reduction_sum((gid < n) ? X[k0 + i] : 0.0f);
            if (threadIdx.x == 0) {
                means[id + i * gridDim.x] += mean_sum;
            }
        }
        
    }

    __global__ void stretch_move_bare(const int n,
                                      const int seed,
                                      const int odd_or_even,
                                      const REAL* params,
                                      const REAL* Scompl,
                                      REAL* X,
                                      REAL* logfn_X,
                                      const REAL a,
                                      const REAL beta,
                                      const int step_counter) {

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            stretch_move(n, seed, params, Scompl, X, logfn_X, a, beta, step_counter, odd_or_even);
        }
    }

// ====================== Walkers initialization ===============================
    __global__ void init_walkers(const int n, const int seed, const REAL2* limits, REAL* xs){

        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int i = gid * 4;
            const REAL2 limits_m0 = limits[i % DIM];
            const REAL2 limits_m1 = limits[(i + 1) % DIM];
            const REAL2 limits_m2 = limits[(i + 2) % DIM];
            const REAL2 limits_m3 = limits[(i + 3) % DIM];

            // Generate uniform(0,1) floats
                        // Generate uniform(0,1) floats
            philox4x32_key_t key;
            uint32_t* key_v = key.v;
            key_v[0] = seed;
            key_v[1] = 0xdecafaaa;
            key_v[2] = 0xfacebead;
            key_v[3] = 0x12345678;
            philox4x32_ctr_t cnt;
            uint32_t* cnt_v = cnt.v;
            cnt_v[0] = gid;
            cnt_v[1] = 0xf00dcafe;
            cnt_v[2] = 0xdeadbeef;
            cnt_v[3] = 0xbeeff00d;
            uint32_t* rand_uni = philox4x32(cnt, key).v;
            
            float4 u;
            u.x = u01fpt_oo_32_24(rand_uni[0]);
            u.y = u01fpt_oo_32_24(rand_uni[1]);
            u.z = u01fpt_oo_32_24(rand_uni[2]);
            u.w = u01fpt_oo_32_24(rand_uni[3]);
            
            xs[i] = u.x * limits_m0.y + (1.0f - u.x) * limits_m0.x;
            xs[i + 1] = u.y * limits_m1.y + (1.0f - u.y) * limits_m1.x;
            xs[i + 2] = u.z * limits_m2.y + (1.0f - u.z) * limits_m2.x;
            xs[i + 3] = u.w * limits_m3.y + (1.0f - u.w) * limits_m3.x;
        }
    }

    __global__ void logfn(const int n, const REAL* params, const REAL* x, REAL* res) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int start = DIM * gid;
            REAL px[DIM];
            for (int i = 0; i < DIM; i++) {
                px[i] = x[start + i];
            }
            res[gid] = LOGFN(params, px);
        }
    }

// ======================== Acceptance =========================================

    void block_reduction_sum_ulong (uint64_t* acc, const uint64_t value) {

        const int local_size = blockDim.x;
        const int local_id = threadIdx.x;

        __shared__ uint64_t lacc[WGS];
        lacc[local_id] = value;

        __syncthreads();

        uint64_t pacc = value;
        int i = local_size;
        while (i > 0) {
            bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
            i >>= 1;
            if (include_odd) {
                pacc += lacc[local_id + i + 1];
            }
            if (local_id < i) {
                pacc += lacc[local_id + i];
                lacc[local_id] = pacc;
            }
            __syncthreads();
        }

        if(local_id == 0) {
            acc[blockIdx.x] = pacc;
        }
    }

    __global__ void sum_accept_reduction (const int n, uint64_t* acc) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            block_reduction_sum_ulong(acc, acc[gid]);
        }
    }

    __global__ void sum_accept_reduce (const int n, uint64_t* acc, const int* data) {
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            block_reduction_sum_ulong(acc, (uint64_t)data[gid]);
        }
    }

    __global__ void sum_means_vertical (const int m, const int n, REAL* acc, const REAL* data) {
        const int gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const int i = n * gid_1 + gid_0;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        const REAL sum = block_reduction_sum_2((valid) ? data[i] : 0.0);
        const bool write = valid && (threadIdx.y == 0);
        if (write) {
            acc[m * blockIdx.y + gid_0] = sum;
        }
    }

    __global__ void subtract_mean (const int dim, const int n, REAL* means, const REAL* mean) {
        const int dim_id = blockIdx.x * blockDim.x + threadIdx.x;
        const int n_id = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (dim_id < dim) && (n_id < n);
        if (valid) {
            means[dim * n_id + dim_id] -= mean[dim_id];
        }
    }

// ======================== Autocovariance =====================================

    REAL2 block_reduction_autocovariance (REAL* c0acc, REAL* dacc, const REAL x2, const REAL xacc) {

        const int local_size = blockDim.x;
        const int local_id = threadIdx.x;

        __shared__ REAL lc0[WGS];
        lc0[local_id] = x2;

        __shared__ REAL ld[WGS];
        ld[local_id] = xacc;

        __syncthreads();

        REAL pc0 = x2;
        REAL pd = xacc;

        int i = local_size;
        while (i > 0) {
            i >>= 1;
            if (local_id < i) {
                pc0 += lc0[local_id + i];
                lc0[local_id] = pc0;
                pd += ld[local_id + i];
                ld[local_id] = pd;
            }
            __syncthreads();
        }

        REAL2 result;
        result.x = lc0[0];
        result.y = ld[0];
        return result;

    }

    __global__ void autocovariance (const int n,
                                    const int lag,
                                    REAL* c0acc,
                                    REAL* dacc,
                                    const REAL* means,
                                    const int imax) {
        
        const int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const int lid = threadIdx.x;
            const int local_size = blockDim.x;
            const int group_id = blockIdx.x;

            __shared__ REAL local_means[2 * WGS];

            const bool load_lag = (lid < lag) && (gid + local_size < n);
            const bool compute = gid < imax;
            REAL xacc = 0.0f;

            for (int i = 0; i < DIM; i++) {
                const REAL x = means[gid * DIM + i];
                local_means[lid] = x;
                local_means[lid + local_size] = load_lag ? means[gid + local_size] : 0.0f;
                __syncthreads();
                xacc = 0.0f;
                for (int s = 0; s < lag; s++) {
                    xacc += x * local_means[lid + s + 1];
                }
                xacc = compute ? x * x + 2 * xacc : 0.0f;
                const REAL2 sums = block_reduction_autocovariance(c0acc, dacc, compute? x*x : 0.0f, xacc);
                if (lid == 0) {
                    c0acc[group_id * DIM + i] = sums.x;
                    dacc[group_id * DIM + i] = sums.y;
                }

            }
        }
    }
}
