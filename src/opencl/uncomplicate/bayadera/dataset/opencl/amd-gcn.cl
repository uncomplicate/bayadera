#ifndef ACCUMULATOR
#define ACCUMULATOR double
#endif

#ifndef WGS
#define WGS 256
#endif

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void mean_variance_reduce(__global ACCUMULATOR* x_acc,
                                   __global ACCUMULATOR* m2n_acc,
                                   __global const float* x) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint group_id = get_group_id(0);
    uint lsize = get_local_size(0);

    ACCUMULATOR xi = x[gid];

    work_group_reduction_sum(x_acc, xi);

    if (lid == 0) {
        //lmu = sum / n
        x_acc[group_id] /= lsize;
    }
    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    ACCUMULATOR lmu = x_acc[group_id];

    work_group_reduction_sum(m2n_acc, (xi - lmu) * (xi - lmu));

    /*if (lid == 0) {

        m2n_acc[group_id] = sum;
    }*/

}

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void mean_variance_reduction(__global ACCUMULATOR* x_acc,
                                      __global ACCUMULATOR* m2n_acc,
                                       const uint tail_rem) {

    uint local_id = get_local_id(0);
    uint global_id = get_global_id(0);

    __local ACCUMULATOR lx[WGS];
    ACCUMULATOR xa = x_acc[global_id];
    lx[local_id] = xa;
    __local ACCUMULATOR lm2n[WGS];
    ACCUMULATOR m2a = m2n_acc[global_id];
    lm2n[local_id] = m2a;
    __local uint ln[WGS];
    uint na = (global_id < (get_global_size(0) - 1)) ? WGS : WGS - tail_rem;
    ln[local_id] = na;

    work_group_barrier(CLK_LOCAL_MEM_FENCE);

    ACCUMULATOR xb, m2b;
    uint nb;
    uint i = get_local_size(0);
    while (i > 0) {
        bool include_odd = (i > ((i >> 1) << 1)) && (local_id == ((i >> 1) - 1));
        i >>= 1;
        if (include_odd) {
            nb = ln[local_id + i + 1];
            xb = lx[local_id + i + 1];
            xa = (na * xa + nb * xb) / (na + nb);
            m2a = m2a + lm2n[local_id + i + 1] +
                ((xb - xa) * (xb - xa) * na * nb / (na + nb));
            na += nb;
        }
        if (local_id < i) {
            nb = ln[local_id + i];
            xb = lx[local_id + i];
            xa = (na * xa + nb * xb) / (na + nb);
            lx[local_id] = xa;
            m2a = m2a + lm2n[local_id + i] +
                ((xb - xa) * (xb - xa) * na * nb / (na + nb));
            lm2n[local_id] = m2a;
            ln[local_id] = na + nb;
        }
        work_group_barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        x_acc[get_group_id(0)] = xa;
        m2n_acc[get_group_id(0)] = m2a;
    }

}
