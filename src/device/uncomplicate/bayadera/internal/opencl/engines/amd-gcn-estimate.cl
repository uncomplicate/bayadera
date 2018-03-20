__attribute__((reqd_work_group_size(1, WGS, 1)))
__kernel void histogram(__global const REAL* limits, __global const REAL* data, __global uint* res) {

    const uint dim = get_global_size(0);
    const uint dim_id = get_global_id(0);
    const uint lid = get_local_id(1);
    const uint gid = get_global_id(1);
    const REAL lower = limits[dim_id * 2];
    const REAL upper = limits[dim_id * 2 + 1];

    __local uint hist[WGS];
    hist[lid] = 0;
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    const REAL x = data[dim * gid + dim_id];
    uint bin = (uint) ((x - lower) / (upper - lower) * WGS);
    bin = (bin < WGS) ? bin : WGS - 1;

    atomic_inc(&hist[bin]);

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    atomic_add(&res[WGS * dim_id + lid], hist[lid]);
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void uint_to_real(const REAL alpha, __global const REAL* limits,
                           __global const uint* data, __global REAL* res) {
    const uint bin_id = get_global_id(0);
    const uint dim_id = get_global_id(1);
    const uint data_id = get_global_size(0) * dim_id + bin_id;
    res[data_id] = alpha / (limits[2 * dim_id + 1] - limits[2 * dim_id]) * (REAL)data[data_id];
}

// ================ Max reduction ==============================================

REAL2 work_group_reduction_min_max_2 (const REAL2 val) {

    const uint local_row = get_local_id(0);
    const uint local_col = get_local_id(1);
    const uint local_m = get_local_size(0);
    const uint id = local_row + local_col * local_m;

    REAL pmin = val.x;
    REAL pmax = val.y;
    __local REAL lmin[WGS];
    lmin[id] = pmin;
    __local REAL lmax[WGS];
    lmax[id] = pmax;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint i = get_local_size(1);
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (local_col == ((i >> 1) - 1));
        i >>= 1;
        uint other_id = local_row + (local_col + i) * local_m;
        if (include_odd) {
            pmax = max(pmax, lmax[other_id + local_m]);
            pmin = min(pmin, lmin[other_id + local_m]);
        }
        if (local_col < i) {
            pmax = max(pmax, lmax[other_id]);
            pmin = min(pmin, lmin[other_id]);
        }
        lmax[id] = pmax;
        lmin[id] = pmin;
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    return (REAL2)(lmin[local_row], lmax[local_row]);

}

__kernel void min_max_reduction (__global REAL2* acc) {
    const uint ia = get_global_size(0) * get_global_id(1) + get_global_id(0);
    const REAL2 min_max = work_group_reduction_min_max_2(acc[ia]);
    if (get_local_id(1) == 0) {
        acc[get_global_size(0) * get_group_id(1) + get_global_id(0)] = min_max;
    }
}

__kernel void min_max_reduce (__global REAL2* acc, __global const REAL* a) {
    const REAL val = a[get_global_size(0) * get_global_id(1) + get_global_id(0)];
    const REAL2 min_max = work_group_reduction_min_max_2((REAL2)(val, val));
    if (get_local_id(1) == 0) {
        acc[get_global_size(0) * get_group_id(1) + get_global_id(0)] = min_max;
    }
}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void bitonic_local(__global const REAL* in, __global REAL* out) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);

    REAL2 value = (REAL2)((REAL)lid, in[gid]);
    __local REAL2 aux[WGS];
    aux[lid] = value;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    for (uint length = 1; length < WGS; length <<= 1) {
        const bool direction = ((lid & (length << 1)) != 0);
        for (int inc = length; inc > 0; inc >>= 1) {
            const int j = lid ^ inc;
            const REAL2 other_value = aux[j];
            const bool smaller = (value.y < other_value.y)
                || (other_value.y == value.y && j < lid);
            const bool swap = smaller ^ (j < lid) ^ direction;
            value = (swap) ? other_value : value;
            work_group_barrier(CLK_LOCAL_MEM_FENCE);
            aux[lid] = value;
            work_group_barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    out[gid] = value.x;

}

__kernel void sum_reduce_horizontal (__global REAL* acc, __global REAL* data) {
    const uint i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    const uint iacc = get_global_size(0) * get_group_id(1) + get_global_id(0);
    const REAL sum = work_group_reduction_sum_2(data[i]);
    if (get_local_id(1) == 0) {
        acc[iacc] = sum;
    }
}

__kernel void mean_reduce(__global ACCUMULATOR* acc, __global const REAL* x) {
    const uint i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    const uint iacc = get_global_size(0) * get_group_id(1) + get_global_id(0);
    const ACCUMULATOR sum = work_group_reduction_sum_2(x[i]);
    if (get_local_id(1) == 0) {
        acc[iacc] = sum;
    }
}

__kernel void variance_reduce(__global ACCUMULATOR* acc,
                              __global const REAL* x, __global const REAL* mu) {
    const uint i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    const uint iacc = get_global_size(0) * get_group_id(1) + get_global_id(0);
    const REAL diff = x[i] - mu[get_global_id(0)];
    const ACCUMULATOR sum = work_group_reduction_sum_2(diff * diff);
    if (get_local_id(1) == 0) {
        acc[iacc] = sum;
    }
}

__kernel void subtract_mean (__global REAL* means, __global const REAL* mean) {
    const uint dim_id = get_global_id(0);
    const uint dim_size = get_global_size(0);
    const uint n_id = get_global_id(1);

    means[dim_size * n_id + dim_id] -= mean[dim_id];
}

// ======================== Autocovariance =====================================

REAL2 work_group_reduction_autocovariance (const REAL x2, const REAL xacc) {

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
        const bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            pc0 += lc0[local_id + i + 1];
            pd += ld[local_id + i + 1];
        }
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
                              __global const REAL* means) {

    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint group_id = get_group_id(0);

    const uint dim_id = get_global_id(1);
    const uint dim = get_global_size(1);

    __local REAL local_means[2 * WGS];

    const bool load_lag = (lid < lag) && (group_id + 1 < get_num_groups(0));
    const bool compute = gid + lag < get_global_size(0);

    const REAL x = compute ? means[gid * dim + dim_id] : 0.0f;
    REAL xacc = 0.0f;
    local_means[lid] = x;
    local_means[lid + local_size] = load_lag ? means[(gid + local_size) * dim + dim_id] : 0.0f;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (compute) {
        for (uint s = 0; s < lag; s++) {
            xacc += local_means[lid + s + 1];
        }
        xacc = x * (x + 2.0f * xacc);
    }
    const REAL2 sums = work_group_reduction_autocovariance(x*x, xacc);

    if (lid == 0) {
        c0acc[group_id * dim + dim_id] = sums.x;
        dacc[group_id * dim + dim_id] = sums.y;
    }



}
