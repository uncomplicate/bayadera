#include "Random123/philox.h"


// =========================== Stretch move ====================================

uint work_group_reduction_sum_uint (const uint value) {

    uint local_id = get_local_id(0);

    __local uint lacc[WGS];
    lacc[local_id] = value;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint pacc = value;
    uint i = get_local_size(0);
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

    return lacc[0];
}

bool stretch_move(const uint seed,
                  __constant const REAL *params,
                  const REAL *Scompl,
                  REAL *X,
                  REAL* logfn_X,
                  const REAL a,
                  const REAL beta,
                  const uint step_counter,
                  const uint odd_or_even) {

    // Get the index of this walker Xk
    const uint k = get_global_id(0);
    const uint K = get_global_size(0);

    // Generate uniform(0,1) floats
    const philox4x32_key_t key = {{seed, 0xdecafbad, 0xfacebead, 0x12345678}};
    const philox4x32_ctr_t cnt = {{k, step_counter, odd_or_even, 0xbeeff00d}};
    const float4 u = u01fpt_oo_4x32_24(((uint4*)philox4x32(cnt, key).v)[0]);

    // Draw a sample from g(z) using the formula from [Christen 2007]
    const REAL z = (a - 2.0f + 1.0f / a) * u.s1 * u.s1
        + (2.0f * (1.0f - 1.0f / a)) * u.s1 + 1.0f / a;

    // Draw a walker Xj's index at random from the complementary ensemble S(~i)(t)
    const uint j0 = (uint)(u.s0 * K * DIM);
    const uint k0 = k * DIM;

    REAL Y[DIM];

    for (uint i = 0; i < DIM; i++) {
        const REAL Xji = Scompl[j0 + i];
        Y[i] = Xji + z * (X[k0 + i] - Xji);
    }

    const REAL logfn_y = LOGFN(params, Y);
    const REAL q = (isfinite(logfn_y)) ?
        pown(z, DIM - 1) * native_exp(beta * (logfn_y - logfn_X[k])) : 0.0f;

    const bool accepted = u.s2 <= q;

    if (accepted) {
        for (uint i = 0; i < DIM; i++) {
            X[k0 + i] = Y[i];
        }
        logfn_X[k] = logfn_y;
    }

    return accepted;
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void stretch_move_accu(const uint seed,
                                const uint odd_or_even,
                                __constant const REAL* params
                                __attribute__ ((max_constant_size(PARAMS_SIZE))),
                                __global const REAL* Scompl,
                                __global REAL* X,
                                __global REAL* logfn_X,
                                __global uint* accept,
                                __global REAL* means,
                                const REAL a,
                                const uint step_counter) {

    const bool accepted = stretch_move(seed, params, Scompl, X, logfn_X, a,
                                       1.0f, step_counter, odd_or_even);

    const uint accepted_sum = work_group_reduction_sum_uint(accepted ? 1 : 0);
    if (get_local_id(0) == 0) {
        accept[get_group_id(0)] += accepted_sum;
    }

    const uint k0 = get_global_id(0) * DIM;
    const uint id = get_group_id(0) + get_num_groups(0) * step_counter * DIM;
    for (uint i = 0; i < DIM; i++) {
        const REAL mean_sum = work_group_reduction_sum(X[k0 + i]);
        if (get_local_id(0) == 0) {
            means[id + i * get_num_groups(0)] += mean_sum;
        }
    }

}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void stretch_move_bare(const uint seed,
                                const uint odd_or_even,
                                __constant const REAL* params
                                __attribute__ ((max_constant_size(PARAMS_SIZE))),
                                __global const REAL* Scompl,
                                __global REAL* X,
                                __global REAL* logfn_X,
                                const REAL a,
                                const REAL beta,
                                const uint step_counter) {

    stretch_move(seed, params, Scompl, X, logfn_X, a, beta, step_counter, odd_or_even);

}

// ====================== Walkers initialization ===============================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void init_walkers(const uint seed,
                           __constant const REAL2* limits,
                           __global REAL* xs){

    const uint i = get_global_id(0) * 4;
    const REAL2 limits_m0 = limits[i % DIM];
    const REAL2 limits_m1 = limits[(i + 1) % DIM];
    const REAL2 limits_m2 = limits[(i + 2) % DIM];
    const REAL2 limits_m3 = limits[(i + 3) % DIM];

    // Generate uniform(0,1) floats
    const philox4x32_key_t key = {{seed, 0xdecafaaa, 0xfacebead, 0x12345678}};
    const philox4x32_ctr_t cnt = {{get_global_id(0), 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
    const float4 u = u01fpt_oo_4x32_24(((uint4*)philox4x32(cnt, key).v)[0]);

    xs[i] = u.s0 * limits_m0.s1 + (1.0f - u.s0) * limits_m0.s0;
    xs[i + 1] = u.s1 * limits_m1.s1 + (1.0f - u.s1) * limits_m1.s0;
    xs[i + 2] = u.s2 * limits_m2.s1 + (1.0f - u.s2) * limits_m2.s0;
    xs[i + 3] = u.s3 * limits_m3.s1 + (1.0f - u.s3) * limits_m3.s0;

}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void logfn(__constant const REAL* params
                    __attribute__ ((max_constant_size(PARAMS_SIZE))),
                    __global const REAL* x, __global REAL* res) {
    uint start = DIM * get_global_id(0);
    REAL px[DIM];
    for (uint i = 0; i < DIM; i++) {
        px[i] = x[start + i];
    }
    res[get_global_id(0)] = LOGFN(params, px);
}

// ======================== Acceptance =========================================

void work_group_reduction_sum_ulong (__global ulong* acc,
                                     const ulong value) {

    const uint local_size = get_local_size(0);
    const uint local_id = get_local_id(0);

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
__kernel void sum_accept_reduce (__global ulong* acc,
                                 __global const uint* data) {
    work_group_reduction_sum_ulong(acc, (ulong)data[get_global_id(0)]);
}

__kernel void sum_means_vertical (__global REAL* acc,
                                  __global const REAL* data) {
    size_t i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    size_t iacc = get_global_size(1) * get_group_id(0) + get_global_id(1);
    REAL sum = work_group_reduction_sum_2(0, data[i]);
    if (get_local_id(0) == 0) {
        acc[iacc] = sum;
    }
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void subtract_mean (__global REAL* means,
                             __global const REAL* mean) {
    const uint dim_id = get_global_id(0);
    const uint dim_size = get_global_size(0);
    const uint n_id = get_global_id(1);

    means[dim_size * n_id + dim_id] -= mean[dim_id];
}

// ======================== Autocovariance =====================================

REAL2 work_group_reduction_autocovariance (__global REAL* c0acc,
                                           __global REAL* dacc,
                                           const REAL x2,
                                           const REAL xacc) {

    const uint local_size = get_local_size(0);
    const uint local_id = get_local_id(0);

    __local REAL lc0[WGS];
    lc0[local_id] = x2;

    __local REAL ld[WGS];
    ld[local_id] = xacc;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    REAL pc0 = x2;
    REAL pd = xacc;

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

    return (REAL2){lc0[0], ld[0]};

}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void autocovariance (const uint lag,
                              __global REAL* c0acc,
                              __global REAL* dacc,
                              __global const REAL* means,
                              const uint imax) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint group_id = get_group_id(0);

    __local REAL local_means[2 * WGS];

    const bool load_lag = (lid < lag) && (gid + local_size < get_global_size(0));
    const bool compute = gid < imax;
    REAL xacc = 0.0f;

    for (uint i = 0; i < DIM; i++) {
        const REAL x = means[gid * DIM + i];
        local_means[lid] = x;
        local_means[lid + local_size] = load_lag ? means[gid + local_size] : 0.0f;
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
        xacc = 0.0f;
        for (uint s = 0; s < lag; s++) {
            xacc += x * local_means[lid + s + 1];
        }
        xacc = compute ? x * x + 2 * xacc : 0.0f;
        const REAL2 sums =
            work_group_reduction_autocovariance(c0acc, dacc, compute? x*x : 0.0f, xacc);
        if (lid == 0) {
            c0acc[group_id * DIM + i] = sums.x;
            dacc[group_id * DIM + i] = sums.y;
        }

    }

}
