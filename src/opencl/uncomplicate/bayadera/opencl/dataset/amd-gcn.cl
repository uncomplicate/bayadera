__kernel void sum_reduction_horizontal(__global ACCUMULATOR* acc) {
    const uint i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    const uint iacc = get_global_size(0) * get_group_id(1) + get_global_id(0);
    const ACCUMULATOR sum = work_group_reduction_sum_2(1, acc[i]);
    if (get_local_id(1) == 0) {
        acc[iacc] = sum;
    }
}

__kernel void mean_reduce(__global ACCUMULATOR* acc, __global const REAL* x) {
    const uint i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    const uint iacc = get_global_size(0) * get_group_id(1) + get_global_id(0);
    const ACCUMULATOR sum = work_group_reduction_sum_2(1, x[i]);
    if (get_local_id(1) == 0) {
        acc[iacc] = sum;
    }
}

__kernel void variance_reduce(__global ACCUMULATOR* acc,
                              __global const REAL* x,
                              __constant const REAL* mu) {
    const uint i = get_global_size(0) * get_global_id(1) + get_global_id(0);
    const uint iacc = get_global_size(0) * get_group_id(1) + get_global_id(0);
    const REAL diff = x[i] - mu[get_global_id(0)];
    const ACCUMULATOR sum = work_group_reduction_sum_2(1, diff * diff);
    if (get_local_id(1) == 0) {
        acc[iacc] = sum;
    }
}
