__kernel void sum_reduce_horizontal (__global REAL* acc, __global REAL* data) {
    const uint i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    const uint iacc = get_global_size(0) * get_group_id(1) + get_global_id(0);
    __local ACCUMULATOR lacc[WGS];
    const REAL sum = work_group_reduction_sum_2(lacc, data[i]);
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

// ======================== Acor =====================================

__kernel void sum_pairwise (const uint stride, const uint offset, __global REAL* x){
    const uint gid = get_global_id(0);
    x[offset + 2 * gid * stride] += x[offset + (2 * gid + 1) * stride];
};

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void acor_1d (const uint stride, const uint dim_id, const uint lag,
                       __global REAL* c0acc, __global REAL* dacc, __global const REAL* means,
                       __local REAL* local_means) {

    const uint gid = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint local_size = get_local_size(0);
    const uint group_id = get_group_id(0);

    const bool load_lag = (local_id < lag) && (group_id + 1 < get_num_groups(0));
    const bool compute = gid + lag < get_global_size(0);

    const REAL x = compute ? means[gid * stride + dim_id] : 0.0f;
    const REAL x_load = load_lag ? means[(gid + local_size) * stride + dim_id] : 0.0f;

    local_means[local_id] = x;
    local_means[local_id + local_size] = x_load;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    REAL xacc = 0.0f;

    for (uint s = 0; s < lag; s++) {
        xacc += local_means[local_id + s + 1];
    }
    xacc = x * (x + 2.0f * xacc);

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    REAL* lc0 = local_means;
    REAL* ld = local_means + local_size;
    REAL pc0 = x * x;
    lc0[local_id] = pc0;
    ld[local_id] = xacc;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint i = local_size;
    while (i > 0) {
        const bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            pc0 += lc0[local_id + i + 1];
            xacc += ld[local_id + i + 1];
        }
        if (local_id < i) {
            pc0 += lc0[local_id + i];
            lc0[local_id] = pc0;
            xacc += ld[local_id + i];
            ld[local_id] = xacc;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        c0acc[group_id * stride + dim_id] = pc0;
        dacc[group_id * stride + dim_id] = xacc;
    }

}

__kernel void sum_reduction_acor (const uint acor_dim, const uint stride,
                                  const uint dim, const uint dim_id,
                                  const uint lag, const uint min_lag, const uint win_mult,
                                  const uint orig_n, const REAL orig_c0,
                                  const uint prev_n, const REAL prev_c0,
                                  __global REAL* c0acc, __global REAL* dacc, __global REAL* means,
                                  __local REAL* lacc) {

    const uint gid = get_global_id(0);
    const uint local_id = get_local_id(0);
    const uint local_size = get_local_size(0);

    REAL* lc0 = lacc;
    REAL* ld = lacc + local_size;

    REAL c0 = 0.0f;
    REAL d = 0.0f;
    const uint iterations = (acor_dim - 1) / local_size + 1;
    for (uint iter = 0; iter < iterations; iter++) {
        const uint cnt = iter * local_size;
        const uint id = (cnt + gid) * stride * dim + dim_id;
        const bool valid = (cnt + gid) < acor_dim;

        if (valid) {
            c0 += c0acc[id];
            d += dacc[id];
        }
    }

    lc0[local_id] = c0;
    ld[local_id] = d;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    uint i = local_size;
    while (i > 0) {
        const bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            c0 += lc0[local_id + i + 1];
            d += ld[local_id + i + 1];
        }
        if (local_id < i) {
            c0 += lc0[local_id + i];
            lc0[local_id] = c0;
            d += ld[local_id + i];
            ld[local_id] = d;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid == 0) {
        const REAL start_c0 = (stride == 1) ? c0 : orig_c0;
        const REAL tau = d / ((stride == 1) ? c0 : prev_c0);
        const uint lag2 = ((lag * win_mult) < prev_n) ? lag : (prev_n / win_mult);
        if ((lag2 < tau * win_mult) && (min_lag < lag2)) {
            queue_t queue = get_default_queue();
            clk_event_t sum_pairwise_event;
            clk_event_t acor_event;
            clk_event_t reduction_event;
            clk_event_t marker;
            const uint n2 = prev_n / 2;
            const uint wgs = (WGS < n2) ? WGS : n2;
            enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n2, wgs),
                           0, NULL, &sum_pairwise_event,
                           ^{sum_pairwise(stride * dim, dim_id, means);});
            enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(n2, wgs),
                           1, &sum_pairwise_event, &acor_event,
                           ^(local void *acc)
                           {acor_1d(2 * stride * dim, dim_id, lag2, c0acc, dacc, means, acc);},
                           4*2*WGS);
            const uint acor_dim = (n2-1)/wgs + 1;
            const uint reduction_dim = (WGS < acor_dim) ? WGS : acor_dim;
            enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(reduction_dim, reduction_dim),
                           1, &acor_event, &reduction_event,
                           ^(local void *acc)
                           {sum_reduction_acor(acor_dim, stride * 2, dim, dim_id,
                                               lag, min_lag, win_mult,
                                               orig_n, start_c0, n2, c0,
                                               c0acc, dacc, means, acc);},
                           4*2*WGS);
            enqueue_marker(queue, 1, &reduction_event, &marker);
            release_event(sum_pairwise_event);
            release_event(acor_event);
            release_event(reduction_event);
            release_event(marker);
        } else {
            const REAL scale = (REAL)stride * (prev_n - lag2);
            c0acc[dim_id] = d * (orig_n - lag) / (scale * start_c0);
            dacc[dim_id] = sqrt(d / (scale * orig_n));
        }
    }

}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void acor_2d (const uint lag,
                       __global REAL* c0acc, __global REAL* dacc, __global const REAL* means) {

    __local REAL local_means[2*WGS];
    acor_1d(get_global_size(1), get_global_id(1), lag, c0acc, dacc, means, local_means);

}

__kernel void acor (const uint n, const uint lag, const uint min_lag, const uint win_mult,
                    __global REAL* c0acc, __global REAL* dacc, __global REAL* means) {

    const uint dim_id = get_global_id(0);
    const uint dim = get_global_size(0);

    queue_t queue = get_default_queue();
    clk_event_t reduction_event;
    clk_event_t marker;
    const uint acor_dim = (n-1) / WGS + 1;
    const uint reduction_dim = (WGS < acor_dim) ? WGS : acor_dim;
    enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(reduction_dim, reduction_dim),
                   0, NULL, &reduction_event,
                   ^(local void *acc)
                   {sum_reduction_acor(acor_dim, 1, dim, dim_id,
                                       lag, min_lag, win_mult,
                                       n, 0.0f, n, 0.0f,
                                       c0acc, dacc, means, acc);},
                   4*2*WGS);
    enqueue_marker(queue, 1, &reduction_event, &marker);
    release_event(reduction_event);
    release_event(marker);
}
