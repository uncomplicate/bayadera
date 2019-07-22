#include "Random123/philox.h"

// =========================== Stretch move ====================================

uint work_group_reduction_sum_uint (uint* lacc, const uint value) {

    const uint local_id = get_local_id(0);

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
                  const uint data_len,
                  const uint params_len,
                  const REAL* params,
                  const REAL* Scompl,
                  REAL* X,
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

    const REAL logfn_y = LOGFN(data_len, params_len, params, DIM, Y);
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
                                const uint data_len,
                                const uint params_len,
                                __global const REAL* params,
                                __global const REAL* Scompl,
                                __global REAL* X,
                                __global REAL* logfn_X,
                                __global uint* accept,
                                __global REAL* means,
                                const REAL a,
                                const uint step_counter) {

    const bool accepted = stretch_move(seed, data_len, params_len, params, Scompl, X, logfn_X, a,
                                       1.0f, step_counter, odd_or_even);

    __local uint uint_lacc[WGS];
    const uint accepted_sum = work_group_reduction_sum_uint(uint_lacc, accepted ? 1 : 0);
    if (get_local_id(0) == 0) {
        accept[get_group_id(0)] += accepted_sum;
    }

    const uint k0 = get_global_id(0) * DIM;
    const uint offset = get_num_groups(0) * DIM * step_counter;
    __local ACCUMULATOR lacc[WGS];
    for (uint i = 0; i < DIM; i++) {
        const REAL mean_sum = work_group_reduction_sum(lacc, X[k0 + i]);
        if (get_local_id(0) == 0) {
            means[offset + i * get_num_groups(0) + get_group_id(0)] += mean_sum;
        }
    }

}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void stretch_move_bare(const uint seed,
                                const uint odd_or_even,
                                const uint data_len,
                                const uint params_len,
                                __global const REAL* params,
                                __global const REAL* Scompl,
                                __global REAL* X,
                                __global REAL* logfn_X,
                                const REAL a,
                                const REAL beta,
                                const uint step_counter) {

    stretch_move(seed, data_len, params_len, params, Scompl, X, logfn_X, a, beta,
                 step_counter, odd_or_even);

}

// ====================== Walkers initialization ===============================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void init_walkers(const uint seed,
                           __global const REAL2* limits,
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
__kernel void logfn(const uint data_len, const uint params_len, __global const REAL* params,
                    __global const REAL* x, __global REAL* res) {

    const uint start = DIM * get_global_id(0);
    res[get_global_id(0)] = LOGFN(data_len, params_len, params, DIM, x + start);
}

// ======================== Acceptance =========================================

void work_group_reduction_sum_ulong (__global ulong* acc, ulong* lacc, const ulong value) {

    const uint local_size = get_local_size(0);
    const uint local_id = get_local_id(0);

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
    __local ulong lacc[WGS];
    work_group_reduction_sum_ulong(acc, lacc, acc[get_global_id(0)]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_accept_reduce (__global ulong* acc, __global const uint* data) {
    __local ulong lacc[WGS];
    work_group_reduction_sum_ulong(acc, lacc, (ulong)data[get_global_id(0)]);
}

__kernel void sum_means_vertical (__global REAL* acc, __global const REAL* data) {
    const uint i = get_global_size(1) * get_global_id(0) + get_global_id(1);
    const uint iacc = get_global_size(0) * get_group_id(1) + get_global_id(0);
    __local ACCUMULATOR lacc[WGS];
    const REAL sum = work_group_reduction_sum_2(lacc, data[i]);
    if (get_local_id(1) == 0) {
        acc[iacc] = sum;
    }
}
