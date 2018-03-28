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

__kernel void sum_pairwise (const uint stride, const uint offset, __global REAL* x){
    const uint gid = get_global_id(0);
    x[offset + 2 * gid * stride] += x[offset + (2 * gid + 1) * stride];
};

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
__kernel void autocovariance_1d (const uint stride,
                                 const uint dim_id,
                                 const uint lag,
                                 __global REAL* c0acc,
                                 __global REAL* dacc,
                                 __global const REAL* means) {

    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint group_id = get_group_id(0);

    __local REAL local_means[2 * WGS];

    const bool load_lag = (lid < lag) && (group_id + 1 < get_num_groups(0));
    const bool compute = gid + lag < get_global_size(0);

    const REAL x = compute ? means[gid * stride + dim_id] : 0.0f;
    REAL xacc = 0.0f;
    local_means[lid] = x;
    local_means[lid + local_size] = load_lag ? means[(gid + local_size) * stride + dim_id] : 0.0f;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    if (compute) {
        for (uint s = 0; s < lag; s++) {
            xacc += local_means[lid + s + 1];
        }
        xacc = x * (x + 2.0f * xacc);
    }
    const REAL2 sums = work_group_reduction_autocovariance(x*x, xacc);

    if (lid == 0) {
        c0acc[group_id * stride + dim_id] = sums.x;
        dacc[group_id * stride + dim_id] = sums.y;
    }
}

__kernel void sum_reduction_autocovariance (const uint acor_count,
                                            const uint stride,
                                            const uint n1,
                                            const uint dim,
                                            const uint dim_id,
                                            const uint lag,
                                            const uint min_lag,
                                            const uint win_mult,
                                            const uint orig_n,
                                            const REAL orig_c0,
                                            const REAL prev_c0,
                                            __global REAL* c0acc,
                                            __global REAL* dacc,
                                            __global REAL* means) {

    uint gid = get_global_id(0);
    uint local_size = get_local_size(0);

    REAL c0 = 0.0f;
    REAL d = 0.0f;
    const uint iterations = (acor_count - 1)/get_local_size(0) + 1;
    for (uint i = 0; i < iterations; i++) {
        const uint cnt = i * local_size;
        const uint id = (cnt + gid) * stride * dim + dim_id;
        const bool valid = (cnt + gid) < acor_count;
        const REAL2 sums = work_group_reduction_autocovariance
            (valid ? c0acc[id] : 0.0f, valid ? dacc[id] : 0.0f);
        c0 += sums.x;
        d += sums.y;
    }

    if (gid == 0) {
        const REAL kk_c0 = (stride == 1) ? c0 : orig_c0;
        const REAL c01 = (stride == 1) ? c0 : prev_c0;
        const REAL tau = d / c01;//TODO
        const uint lag2 = ((lag * win_mult) < n1) ? lag : (n1 / win_mult);
        if ((tau * win_mult < lag2) || (lag2 < min_lag)) {
            const REAL scale = (REAL)stride * (n1 - lag2);
            c0acc[dim_id] = d * (orig_n - lag) / (scale * kk_c0);
            dacc[dim_id] = sqrt(d / (scale * orig_n));
        } else {
            queue_t queue = get_default_queue();
            clk_event_t sum_pairwise_event;
            clk_event_t autocovariance_event;
            clk_event_t reduction_event;
            clk_event_t marker;
            const uint n2 = n1 / 2;
            const uint wgs = (WGS < n2) ? WGS : n2;
            enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n2, wgs),
                           0, NULL, &sum_pairwise_event,
                           ^{sum_pairwise(stride * dim, dim_id, means);});
            enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n2, wgs),
                           1, &sum_pairwise_event, &autocovariance_event,
                           ^{autocovariance_1d(2 * stride * dim, dim_id, lag2, c0acc, dacc, means);});
            const uint acor_count = (n2-1)/wgs + 1;
            const uint red_count = (WGS < acor_count) ? WGS : acor_count;
            enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(red_count, red_count),
                           1, &autocovariance_event, &reduction_event,
                           ^{sum_reduction_autocovariance(acor_count, stride * 2, n2, dim, dim_id,
                                                          lag, min_lag, win_mult,
                                                          orig_n, kk_c0, c0,
                                                          c0acc, dacc, means);});
            enqueue_marker(queue, 1, &reduction_event, &marker);
            release_event(sum_pairwise_event);
            release_event(autocovariance_event);
            release_event(reduction_event);
            release_event(marker);
        }
    }

}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void autocovariance_2d (const uint lag,
                                 __global REAL* c0acc,
                                 __global REAL* dacc,
                                 __global const REAL* means) {

    autocovariance_1d(get_global_size(1), get_global_id(1), lag, c0acc, dacc, means);

}

__kernel void autocovariance_refine (const uint n,
                                     const uint lag,
                                     const uint min_lag,
                                     const uint win_mult,
                                     __global REAL* c0acc,
                                     __global REAL* dacc,
                                     __global REAL* means) {

    const uint dim_id = get_global_id(0);
    const uint dim = get_global_size(0);

    queue_t queue = get_default_queue();
    clk_event_t reduction_event;
    clk_event_t marker;
    const uint acor_count = (n-1)/WGS + 1;
    const uint red_count = (WGS < acor_count) ? WGS : acor_count;
    enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(red_count, red_count),
                   0, NULL, &reduction_event,
                   ^{sum_reduction_autocovariance(acor_count, 1, n, dim, dim_id,
                                                  lag, min_lag, win_mult,
                                                  n, 0.0f, 0.0f,
                                                  c0acc, dacc, means);});
    enqueue_marker(queue, 1, &reduction_event, &marker);
    release_event(reduction_event);
    release_event(marker);
}
