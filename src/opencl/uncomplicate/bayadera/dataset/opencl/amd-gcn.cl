#ifndef WGS
#define WGS 256
#endif

__attribute__((reqd_work_group_size(WGS, 1, 1)))
__kernel void variance_reduce(__global double* x_acc, __global const float* x,
                              float mu) {

    double xi = x[get_global_id(0)];
    double sum = work_group_reduction_sum(pown(xi - mu, 2));
    if (get_local_id(0) == 0) {
        x_acc[get_group_id(0)] = sum;
    }

}
