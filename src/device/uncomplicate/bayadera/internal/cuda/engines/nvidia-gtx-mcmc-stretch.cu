extern "C" {

#include "Random123/philox.h"
#include <stdint.h>

// =========================== Stretch move ====================================

    uint32_t block_reduction_sum_uint (const uint32_t value) {

        const uint32_t local_id = threadIdx.x;

        __shared__ uint32_t lacc[WGS];
        lacc[local_id] = value;

        __syncthreads();

        uint32_t pacc = value;
        uint32_t i = blockDim.x;
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

    bool stretch_move(const uint32_t K,
                      const uint32_t seed,
                      const uint32_t params_len,
                      const REAL* params,
                      const REAL* Scompl,
                      REAL* X,
                      REAL* logfn_X,
                      const REAL a,
                      const REAL beta,
                      const uint32_t step_counter,
                      const uint32_t odd_or_even) {

        // Get the index of this walker Xk
        const uint32_t k = blockIdx.x * blockDim.x + threadIdx.x;
        
        // Generate uniform(0,1) floats
        philox4x32_key_t key;
        uint32_t* key_v = key.v;
        key_v[0] = seed;
        key_v[1] = 0xdecafbad;
        key_v[2] = 0xfacebead;
        key_v[3] = 0x12345678;
        philox4x32_ctr_t cnt;
        uint32_t* cnt_v = cnt.v;
        cnt_v[0] = k;
        cnt_v[1] = step_counter;
        cnt_v[2] = odd_or_even;
        cnt_v[3] = 0xbeeff00d;
        uint32_t* rand_uni = (uint32_t*)philox4x32(cnt, key).v;
            
        float4 u;
        u.x = u01fpt_oo_32_24(rand_uni[0]);
        u.y = u01fpt_oo_32_24(rand_uni[1]);
        u.z = u01fpt_oo_32_24(rand_uni[2]);
        u.w = u01fpt_oo_32_24(rand_uni[3]);

        // Draw a sample from g(z) using the formula from [Christen 2007]
        const REAL z = (a - 2.0f + 1.0f / a) * u.y * u.y
            + (2.0f * (1.0f - 1.0f / a)) * u.y + 1.0f / a;

        // Draw a walker Xj's index at random from the complementary ensemble S(~i)(t)
        const uint32_t j0 = (uint32_t)(u.x * K * DIM);
        const uint32_t k0 = k * DIM;

        REAL Y[DIM];

        for (uint32_t i = 0; i < DIM; i++) {
            const REAL Xji = Scompl[j0 + i];
            Y[i] = Xji + z * (X[k0 + i] - Xji);
        }
        
        const REAL logfn_y = LOGFN(params_len, params, DIM, Y);
        const REAL q = (isfinite(logfn_y)) ?
            powf(z, DIM - 1) * exp(beta * (logfn_y - logfn_X[k])) : 0.0f;

        const bool accepted = u.z <= q;

        if (accepted) {
            for (uint32_t i = 0; i < DIM; i++) {
                X[k0 + i] = Y[i];
            }
            logfn_X[k] = logfn_y;
        }

        return accepted;
    }

    __global__ void stretch_move_accu(const uint32_t n,
                                      const uint32_t seed,
                                      const uint32_t odd_or_even,
                                      const uint32_t params_len,
                                      const REAL* params,
                                      const REAL* Scompl,
                                      REAL* X,
                                      REAL* logfn_X,
                                      uint32_t* accept,
                                      REAL* means,
                                      const REAL a,
                                      const uint32_t step_counter) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        const bool accepted = (gid < n) &&
            stretch_move(n, seed, params_len, params, Scompl, X, logfn_X, a, 1.0f, step_counter, odd_or_even);

        const uint32_t accepted_sum = block_reduction_sum_uint(accepted ? 1 : 0);
        if (threadIdx.x == 0) {
            accept[blockIdx.x] += accepted_sum;
        }
        
        const uint32_t k0 = gid * DIM;
        const uint32_t id = blockIdx.x + gridDim.x * step_counter * DIM;
        for (uint32_t i = 0; i < DIM; i++) {
            const REAL mean_sum = block_reduction_sum((gid < n) ? X[k0 + i] : 0.0f);
            if (threadIdx.x == 0) {
                means[id + i * gridDim.x] += mean_sum;
            }
        }
        
    }

    __global__ void stretch_move_bare(const uint32_t n,
                                      const uint32_t seed,
                                      const uint32_t odd_or_even,
                                      const uint32_t params_len,
                                      const REAL* params,
                                      const REAL* Scompl,
                                      REAL* X,
                                      REAL* logfn_X,
                                      const REAL a,
                                      const REAL beta,
                                      const uint32_t step_counter) {

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            stretch_move(n, seed, params_len, params, Scompl, X, logfn_X, a, beta, step_counter, odd_or_even);
        }
    }

// ====================== Walkers initialization ===============================

    __global__ void init_walkers(const uint32_t n, const uint32_t seed, const REAL2* limits, REAL* xs){

        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t i = gid * 4;
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

    __global__ void logfn(const uint32_t n,
                          const uint32_t params_len, const REAL* params,
                          const REAL* x, REAL* res) {
        
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t start = DIM * gid;
            res[gid] = LOGFN(params_len, params, DIM, x + start);
        }
    }

// ======================== Acceptance =========================================

    void block_reduction_sum_ulong (uint64_t* acc, const uint64_t value) {

        const uint32_t local_size = blockDim.x;
        const uint32_t local_id = threadIdx.x;

        __shared__ uint64_t lacc[WGS];
        lacc[local_id] = value;

        __syncthreads();

        uint64_t pacc = value;
        uint32_t i = local_size;
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

    __global__ void sum_accept_reduction (const uint32_t n, uint64_t* acc) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        block_reduction_sum_ulong(acc, (gid < n) ? acc[gid] : 0);
    }

    __global__ void sum_accept_reduce (const uint32_t n, uint64_t* acc, const uint32_t* data) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        block_reduction_sum_ulong(acc, (gid < n) ? (uint64_t)data[gid] : 0);        
    }

    __global__ void sum_means_vertical (const uint32_t m, const uint32_t n, REAL* acc, const REAL* data) {
        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t i = n * gid_0 + gid_1;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        const REAL sum = block_reduction_sum_2((valid) ? data[i] : 0.0);
        const bool write = valid && (threadIdx.y == 0);
        if (write) {
            acc[m * blockIdx.y + gid_0] = sum;
        }
    }
}
