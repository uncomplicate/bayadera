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

inline REAL2 work_group_reduction_autocovariance (__global REAL* c0acc,
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
