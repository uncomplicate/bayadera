#include "Random123/philox.h"


#ifndef M_SQRT_2PI_F
#define M_SQRT_2PI_F 0.9189385332046727f
#endif

#ifndef PARAMS_SIZE
    #define PARAMS_SIZE 2
#endif

#ifndef WGS
    #define WGS 256
#endif


// We'll test this with gaussian
inline float logpdf(__constant float *params, float x) {
    float mu = params[0];
    float sigma = params[1];
    return (x - mu) * (x - mu) / (-2.0f * sigma * sigma)
        - native_log(sigma) - M_SQRT_2PI_F;
}

// =============================================================================

inline void work_group_reduction_accumulate (__global uint* accept,
                                             const uint accepted,
                                             __global float* acc,
                                             const float value,
                                             const uint step_counter) {

    uint local_size = get_local_size(0);
    uint local_id = get_local_id(0);

    __local uint laccept[WGS];
    laccept[local_id] = accepted;

    __local float lacc[WGS];
    lacc[local_id] = value;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint paccept = accepted;
    float pacc = value;
    uint i = local_size;
    while (i > 0) {
        i >>= 1;
        if (local_id < i) {
            paccept += laccept[local_id + i];
            laccept[local_id] = paccept;
            pacc += lacc[local_id + i];
            lacc[local_id] = pacc;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint group_id = get_group_id(0);
    if(local_id == 0) {
        accept[group_id] += paccept;
        acc[group_id + get_num_groups(0) * step_counter] += pacc;
    }
}

inline void work_group_reduction_sum_ulong (__global ulong* acc,
                                            const ulong value) {

    uint local_size = get_local_size(0);
    uint local_id = get_local_id(0);

    __local ulong lacc[WGS];
    lacc[local_id] = value;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    ulong pacc = value;
    uint i = local_size;
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
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id == 0) {
        acc[get_group_id(0)] = pacc;
    }
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_accept_reduction (__global ulong* acc) {
    work_group_reduction_sum_ulong(acc, acc[get_global_id(0)]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_accept_reduce (__global ulong* acc, __global const uint* data) {
    work_group_reduction_sum_ulong(acc, (ulong)data[get_global_id(0)]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_means (__global float* acc, __global const float* data,
                         const uint n) {
    uint gid = get_global_id(0);
    uint start = gid * n;
    float pacc = 0.0;
    for(uint i = 0; i < n; i++) {
        pacc += data[start + i];
    }
    acc[gid] = pacc / (n * WGS * 2);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void subtract_mean (__global float* means, const float mean) {
    uint gid = get_global_id(0);
    means[gid] -= mean;
}

inline void work_group_reduction_autocovariance (__global float* c0acc,
                                                 __global float* dacc,
                                                 const float x2,
                                                 const float xacc) {

    uint local_size = get_local_size(0);
    uint local_id = get_local_id(0);

    __local float lc0[WGS];
    lc0[local_id] = x2;

    __local float ld[WGS];
    ld[local_id] = xacc;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    float pc0 = x2;
    float pd = xacc;

    uint i = local_size;
    while (i > 0) {
        i >>= 1;
        if (local_id < i) {
            pc0 += lc0[local_id + i];
            lc0[local_id] = pc0;
            pd += ld[local_id + i];
            ld[local_id] = pd;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    uint group_id = get_group_id(0);
    if(local_id == 0) {
        c0acc[group_id] = pc0;
        dacc[group_id] = pd;
    }
}

#define TAUMAX 16
#define WINMULT 16
#define MAXLAG TAUMAX * WINMULT

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void autocovariance (const uint lag,
                              __global float* c0acc, __global float* dacc,
                              __global float* const means,
                              const uint imax) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint local_size = get_local_size(0);

    __local float local_means[WGS + MAXLAG];

    float x = means[gid];
    local_means[lid] = x;

    bool load_lag = (lid < lag) && (gid + local_size < get_global_size(0));

    local_means[lid + local_size] = load_lag ? means[gid + local_size] : 0.0f;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    float xacc = 0.0f;

    for (uint s = 0; s < lag; s++) {
        xacc += x * local_means[lid + s + 1];
    }

    bool compute = gid < imax;
    xacc = compute ? x * x + 2 * xacc : 0.0f;

    work_group_reduction_autocovariance(c0acc, dacc, compute? x*x : 0.0f, xacc);

}

inline float stretch_move1(const uint seed, __constant const float* params,
                           const float* Scompl, float* X, const float Xk,
                           float* logpdf_X,
                           const uint step_counter, const float a) {

    // Get the index of this walker Xk
    uint k = get_global_id(0);
    uint K2 = get_global_size(0);

    // Generate uniform(0,1) floats
    philox4x32_key_t key = {{seed, 0xdecafbad, 0xfacebead, 0x12345678}};
    philox4x32_ctr_t cnt = {{k, step_counter, 0xdeadbeef, 0xbeeff00d}};
    float4 u = u01fpt_oo_4x32_24(((uint4*)philox4x32(cnt, key).v)[0]);

    // Draw a walker Xj at random from the complementary ensemble S(~i)(t)
    float Xj = Scompl[(uint)(u.s0 * K2)];

    // Compute the proposal Y. Since pow(z, n-1) == 1, z is directly computed
    // Draw a sample from g(z) using the formula from [Christen 2007]
    float Y = Xj + ((a - 2.0f + 1.0f / a) * u.s1 * u.s1 +
                    (2.0f * (1.0f - 1.0f / a)) * u.s1 +
                    1.0f / a) * (Xk - Xj);

    float logpdf_y = logpdf(params, Y);
    float q = (isfinite(logpdf_y)) ?
        native_exp(logpdf(params, Y) - logpdf_X[k]) : 0.0f;

    bool accepted = u.s2 <= q;

    if (accepted) {
        X[k] = Y;
        logpdf_X[k] = logpdf_y;
    }

    return accepted ? Y : Xk;
}

// =============================================================================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void stretch_move1_accu(const uint seed,
                                 __constant const float* params
                                 __attribute__ ((max_constant_size(PARAMS_SIZE))),
                                 __global const float* Scompl, __global float *X,
                                 __global float* logpdf_X, __global uint *accept,
                                 __global float *means,
                                 const float a, const uint step_counter) {

    float Xk = X[get_global_id(0)];
    float res = stretch_move1(seed, params, Scompl, X, Xk, logpdf_X,
                              step_counter, a);

    work_group_reduction_accumulate(accept, (res != Xk) ? 1 : 0, means,
                                    res, step_counter);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void stretch_move1_bare(const uint seed,
                                 __constant const float *params
                                 __attribute__ ((max_constant_size(PARAMS_SIZE))),
                                 __global const float *Scompl, __global float *X,
                                 __global float* logpdf_X,
                                 const float a, const uint step_counter) {
    float Xk = X[get_global_id(0)];
    stretch_move1(seed, params, Scompl, X, Xk, logpdf_X, step_counter, a);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void log_pdf(__constant const float* params
                      __attribute__ ((max_constant_size(PARAMS_SIZE))),
                      __global const float* X, __global float* logpdf_X) {
    uint gid = get_global_id(0);
    logpdf_X[gid] = logpdf(params, X[gid]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void pdf(__constant const float *params
                  __attribute__ ((max_constant_size(PARAMS_SIZE))),
                  __global const float *X,
                  __global float *pdf_X) {

    uint gid = get_global_id(0);
    pdf_X[gid] = native_exp(logpdf(params, X[gid]));
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void init_walkers(uint const seed, __global float4 *xs){
    uint gid = get_global_id(0);
    // Generate uniform(0,1) floats
    philox4x32_key_t key = {{seed, 0xdecafaaa, 0xfacebead, 0x12345678}};
    philox4x32_ctr_t cnt = {{gid, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
    xs[gid] = (float4)u01fpt_oo_4x32_24(((uint4*)philox4x32(cnt, key).v)[0]);
}
