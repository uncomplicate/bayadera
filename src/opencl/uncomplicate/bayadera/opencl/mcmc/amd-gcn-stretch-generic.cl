#ifndef WGS
#define WGS 256
#endif

inline void work_group_reduction_sum_ulong (__global ulong* acc,
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
                                  __global REAL* data) {
    size_t i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    size_t iacc = get_global_size(1) * get_group_id(0) + get_global_id(1);
    REAL sum = work_group_reduction_sum_2(0, data[i]);
    if (get_local_id(0) == 0) {
        acc[iacc] = sum;
    }
}

__kernel void sum_reduction_horizontal (__global REAL* acc) {
    uint i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    uint iacc = get_global_size(0) * get_group_id(1) + get_global_id(0);
    REAL sum = work_group_reduction_sum_2(1, acc[i]);
    if (get_local_id(1) == 0) {
        acc[iacc] = sum;
    }
}

__kernel void sum_reduce_horizontal (__global REAL* acc,
                                     __global REAL* data) {
    uint i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    uint iacc = get_global_size(0) * get_group_id(1) + get_global_id(0);
    REAL sum = work_group_reduction_sum_2(1, data[i]);
    if (get_local_id(1) == 0) {
        acc[iacc] = sum;
    }
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void scal (const REAL alpha, __global REAL* x) {
    x[get_global_id(0)] *= alpha;
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void subtract_mean (__global REAL* means,
                             __global const REAL* mean) {
    const uint dim_id = get_global_id(0);
    const uint dim_size = get_global_size(0);
    const uint n_id = get_global_id(1);

    means[dim_size * n_id + dim_id] -= mean[dim_id];
}

inline float2 work_group_reduction_autocovariance (__global float* c0acc,
                                                   __global float* dacc,
                                                   const float x2,
                                                   const float xacc) {

    const uint local_size = get_local_size(0);
    const uint local_id = get_local_id(0);

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

    return (float2){lc0[0], ld[0]};

}
