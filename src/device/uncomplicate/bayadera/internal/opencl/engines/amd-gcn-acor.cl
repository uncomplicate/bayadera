// ======================== Acor =====================================

__kernel void sum_pairwise (const uint stride, const uint offset, __global REAL* x){
    const uint gid = get_global_id(0);
    x[offset + 2 * gid * stride] += x[offset + (2 * gid + 1) * stride];
};

REAL2 work_group_reduction_acor (const REAL x2, const REAL xacc) {

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
__kernel void acor_1d (const uint stride, const uint dim_id, const uint lag,
                       __global REAL* c0acc, __global REAL* dacc, __global const REAL* means) {

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
    const REAL2 sums = work_group_reduction_acor(x*x, xacc);

    if (lid == 0) {
        c0acc[group_id * stride + dim_id] = sums.x;
        dacc[group_id * stride + dim_id] = sums.y;
    }
}

__kernel void sum_reduction_acor (const uint acor_dim, const uint stride,
                                  const uint dim, const uint dim_id,
                                  const uint lag, const uint min_lag, const uint win_mult,
                                  const uint orig_n, const REAL orig_c0,
                                  const uint prev_n, const REAL prev_c0,
                                  __global REAL* c0acc, __global REAL* dacc, __global REAL* means) {

    uint gid = get_global_id(0);
    uint local_size = get_local_size(0);

    REAL c0 = 0.0f;
    REAL d = 0.0f;
    const uint iterations = (acor_dim - 1) / get_local_size(0) + 1;
    for (uint i = 0; i < iterations; i++) {
        const uint cnt = i * local_size;
        const uint id = (cnt + gid) * stride * dim + dim_id;
        const bool valid = (cnt + gid) < acor_dim;
        const REAL2 sums = work_group_reduction_acor
            (valid ? c0acc[id] : 0.0f, valid ? dacc[id] : 0.0f);
        c0 += sums.x;
        d += sums.y;
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
                           ^{acor_1d(2 * stride * dim, dim_id, lag2, c0acc, dacc, means);});
            const uint acor_dim = (n2-1)/wgs + 1;
            const uint reduction_dim = (WGS < acor_dim) ? WGS : acor_dim;
            enqueue_kernel(queue, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange_1D(reduction_dim, reduction_dim),
                           1, &acor_event, &reduction_event,
                           ^{sum_reduction_acor(acor_dim, stride * 2, dim, dim_id,
                                                lag, min_lag, win_mult,
                                                orig_n, start_c0, n2, c0,
                                                c0acc, dacc, means);});
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

    acor_1d(get_global_size(1), get_global_id(1), lag, c0acc, dacc, means);

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
                   ^{sum_reduction_acor(acor_dim, 1, dim, dim_id,
                                        lag, min_lag, win_mult,
                                        n, 0.0f, n, 0.0f,
                                        c0acc, dacc, means);});
    enqueue_marker(queue, 1, &reduction_event, &marker);
    release_event(reduction_event);
    release_event(marker);
}
