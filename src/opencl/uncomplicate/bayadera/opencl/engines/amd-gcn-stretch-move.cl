#include "Random123/philox.h"

inline void work_group_reduction_accumulate (__global uint* accept,
                                             const uint accepted,
                                             __global REAL* acc,
                                             REAL* pacc,
                                             const uint step_counter) {

    const uint local_id = get_local_id(0);

    __local uint laccept[WGS];
    laccept[local_id] = accepted;

    __local REAL lacc[DIM][WGS];
    for (uint j = 0; j < DIM; j++){
        lacc[j][local_id] = pacc[j];
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint i = get_local_size(0);
    while (i > 0) {
        i >>= 1;
        if (local_id < i) {
            laccept[local_id] += laccept[local_id + i];
            for (uint j = 0; j < DIM; j++) {
                pacc[j] += lacc[j][local_id + i];
                lacc[j][local_id] = pacc[j];
            }

        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id == 0) {
        uint id = get_group_id(0);
        accept[id] += laccept[0];
        id += get_num_groups(0) * step_counter * DIM;
        for (uint j = 0; j < DIM; j++) {
            id += j * get_num_groups(0);
            acc[id] += pacc[j];
        }
    }
}

// =============================================================================

inline bool stretch_move(const uint seed,
                         __constant const REAL *params,
                         const REAL *Scompl,
                         REAL *X,
                         REAL* logpdf_X,
                         const REAL a,
                         const uint step_counter,
                         REAL* Y) {

    // Get the index of this walker Xk
    const uint k = get_global_id(0);
    const uint K = get_global_size(0);

    // Generate uniform(0,1) floats
    const philox4x32_key_t key = {{seed, 0xdecafbad, 0xfacebead, 0x12345678}};
    const philox4x32_ctr_t cnt = {{k, step_counter, 0xdeadbeef, 0xbeeff00d}};
    const float4 u = u01fpt_oo_4x32_24(((uint4*)philox4x32(cnt, key).v)[0]);

    // Draw a sample from g(z) using the formula from [Christen 2007]
    const REAL z = (a - 2.0f + 1.0f / a) * u.s1 * u.s1
        + (2.0f * (1.0f - 1.0f / a)) * u.s1 + 1.0f / a;

    // Draw a walker Xj's index at random from the complementary ensemble S(~i)(t)
    const uint j0 = (uint)(u.s0 * K * DIM);
    const uint k0 = k * DIM;

    for (uint i = 0; i < DIM; i++) {
        const REAL Xji = Scompl[j0 + i];
        Y[i] = Xji + z * (X[k0 + i] - Xji);
    }

    const REAL logpdf_y = LOGPDF(params, Y);
    const REAL q = (isfinite(logpdf_y)) ?
        pown(z, DIM - 1) * native_exp(logpdf_y - logpdf_X[k]) : 0.0f;

    const bool accepted = u.s2 <= q;

    if (accepted) {
        for (uint i = 0; i < DIM; i++) {
            X[k0 + i] = Y[i];
        }
        logpdf_X[k] = logpdf_y;
    }

    return accepted;
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void stretch_move_accu(const uint seed,
                                __constant const REAL* params
                                __attribute__ ((max_constant_size(PARAMS_SIZE))),
                                __global const REAL* Scompl,
                                __global REAL* X,
                                __global REAL* logpdf_X,
                                __global uint* accept,
                                __global REAL* means,
                                const REAL a,
                                const uint step_counter) {

    REAL Y[DIM];
    bool accepted = stretch_move(seed, params, Scompl, X, logpdf_X, a, step_counter, Y);

    if (!accepted){
        const uint k0 = get_global_id(0) * DIM;
        for (uint i = 0; i < DIM; i++) {
            Y[i] = X[k0 + i]; // Reuse private memory Y that is no longer needed.
        }
    }
    work_group_reduction_accumulate(accept, accepted ? 1 : 0, means, Y, step_counter);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void stretch_move_bare(const uint seed,
                                __constant const REAL* params
                                __attribute__ ((max_constant_size(PARAMS_SIZE))),
                                __global const REAL* Scompl,
                                __global REAL* X,
                                __global REAL* logpdf_X,
                                const REAL a,
                                const uint step_counter) {

    REAL Y[DIM];
    bool accepted = stretch_move(seed, params, Scompl, X, logpdf_X, a, step_counter, Y);
}

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

    xs[i] = u.s0 * (limits_m0.s1 - limits_m0.s0) + limits_m0.s0;
    xs[i + 1] = u.s1 * (limits_m1.s1 - limits_m1.s0) + limits_m1.s0;
    xs[i + 2] = u.s2 * (limits_m2.s1 - limits_m2.s0) + limits_m2.s0;
    xs[i + 3] = u.s3 * (limits_m3.s1 - limits_m3.s0) + limits_m3.s0;
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
