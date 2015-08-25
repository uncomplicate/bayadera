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

#ifndef A0
    #define A0 0.6666667f
#endif

#ifndef A1
    #define A1 1.3333334f
#endif

#ifndef A2
    #define A2 1.3333334f
#endif

// We'll test this with gaussian
inline float logpdf(__constant float *params, float x) {
    float mu = params[0];
    float sigma = params[1];
    return (x - mu) * (x - mu) / (-2.0f * sigma * sigma)
        - native_log(sigma) - M_SQRT_2PI_F;
}

// =============================================================================

inline void work_group_reduction_accumulate (__global uint* acc,
                                             const uint value) {

    uint local_size = get_local_size(0);
    uint local_id = get_local_id(0);

    __local uint lacc[WGS];
    lacc[local_id] = value;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint pacc = value;
    uint i = local_size;
    while (i > 0) {
        i >>= 1;
        if (local_id < i) {
            pacc += lacc[local_id + i];
            lacc[local_id] = pacc;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(local_id == 0) {
        acc[get_group_id(0)] += pacc;
    }
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_reduction (__global ulong* acc) {
    work_group_reduction_sum(acc, acc[get_global_id(0)]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void sum_accept_reduce (__global ulong* acc, __global uint* data) {
    work_group_reduction_sum(acc, (ulong)data[get_global_id(0)]);
}

// =============================================================================

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void stretch_move1(uint const seed,
                            __constant const float *params
                            __attribute__ ((max_constant_size(PARAMS_SIZE))),
                            __global const float *Scompl, __global float *X,
                            __global uint *accept) {

    // Get the index of this walker Xk
    uint k = get_global_id(0);
    uint K2 = get_global_size(0);

    // Generate uniform(0,1) floats
    philox4x32_key_t key = {{seed, 0xdecafbad, 0xfacebead, 0x12345678}};
    philox4x32_ctr_t cnt = {{k, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
    float4 u = u01fpt_oo_4x32_24(((uint4*)philox4x32(cnt, key).v)[0]);

    // Draw a walker Xj at random from the complementary ensemble S(~i)(t)
    float Xj = Scompl[(uint)(u.s0 * K2)];
    float Xk = X[k];

    // Compute the proposal Y. Since pow(z, n-1) == 1, z is directly computed
    // Draw a sample from g(z) using the formula from [Christen 2007]
    float Y = Xj + (A2 * u.s1 * u.s1 + A1 * u.s1 - A2) * (Xk - Xj);
    float q = native_exp(logpdf(params, Y) - logpdf(params, Xk));

    bool accepted = u.s2 <= q;
    work_group_reduction_accumulate(accept, accepted ? 1 : 0);

    if (accepted) {
        X[k] = Y;
    }
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void pdf_kernel(__constant const float *params
                         __attribute__ ((max_constant_size(PARAMS_SIZE))),
                         __global const float *X,
                         __global float *pdf) {

    uint gid = get_global_id(0);
    pdf[gid] = native_exp(logpdf(params, X[gid]));
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void logpdf_kernel(__constant const float *params
                            __attribute__ ((max_constant_size(PARAMS_SIZE))),
                            __global const float *X,
                            __global float *pdf) {

    uint gid = get_global_id(0);
    pdf[gid] = logpdf(params, X[gid]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void init_walkers(uint const seed, __global float4 *xs){
    uint gid = get_global_id(0);
    // Generate uniform(0,1) floats
    philox4x32_key_t key = {{seed, 0xdecafbad, 0xfacebead, 0x12345678}};
    philox4x32_ctr_t cnt = {{gid, 0xf00dcafe, 0xdeadbeef, 0xbeeff00d}};
    xs[gid] = (float4)u01fpt_oo_4x32_24(((uint4*)philox4x32(cnt, key).v)[0]);
}
