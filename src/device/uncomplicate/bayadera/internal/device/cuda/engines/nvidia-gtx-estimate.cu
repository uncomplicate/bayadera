extern "C" {

    #include <stdint.h>

    __global__ void histogram(const uint32_t dim, const uint32_t n,
                              const REAL* limits,
                              const REAL* data, const uint32_t offset, const uint32_t ld,
                              uint32_t* res) {

        const uint32_t dim_id = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t lid = threadIdx.y;
        const uint32_t gid = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (dim_id < dim) && (gid < n);

        __shared__ uint32_t hist[WGS];
        hist[lid] = 0;
        __syncthreads();

        if (valid) {
            const REAL lower = limits[dim_id * 2];
            const REAL upper = limits[dim_id * 2 + 1];
            const REAL x = data[offset + ld * gid + dim_id];
            uint32_t bin = (uint32_t) ((x - lower) / (upper - lower) * WGS);
            bin = (bin < WGS) ? bin : WGS - 1;

            atomicAdd(&hist[bin], 1);
                        
            __syncthreads();

            atomicAdd(&res[WGS * dim_id + lid], hist[lid]);
        }
    }

    __global__ void uint_to_real(const uint32_t wgs, const uint32_t m,
                                 const REAL alpha, const REAL* limits,
                                 const uint32_t* data, REAL* res) {
        const uint32_t bin_id = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t dim_id = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t data_id = gridDim.x * blockDim.x * dim_id + bin_id;
        const bool valid = (bin_id < wgs) && (dim_id < m);
        if (valid) {
            res[data_id] = alpha / (limits[2 * dim_id + 1] - limits[2 * dim_id]) * (REAL)data[data_id];
        }
    }

    // ================ Max reduction ==============================================

    REAL2 block_reduction_min_max_2 (const REAL val_x, const REAL val_y) {

        const uint32_t local_row = threadIdx.x;
        const uint32_t local_col = threadIdx.y;
        const uint32_t local_m = blockDim.x;
        const uint32_t id = local_row + local_col * local_m;

        REAL pmin = val_x;
        REAL pmax = val_y;
        __shared__ REAL lmin[WGS];
        lmin[id] = pmin;
        __shared__ REAL lmax[WGS];
        lmax[id] = pmax;

        __syncthreads();

        uint32_t i = blockDim.y;
        while (i > 0) {
            bool include_odd = (i > ((i >> 1) << 1)) && (local_col == ((i >> 1) - 1));
            i >>= 1;
            uint32_t other_id = local_row + (local_col + i) * local_m;
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
            __syncthreads();
        }

        REAL2 result;
        result.x = lmin[local_row];
        result.y = lmax[local_row];
        return result;

    }

    __global__ void min_max_reduction (const uint32_t m, const uint32_t n, REAL2* acc) {
        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t ia = m * gid_1 + gid_0;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        REAL2 val;
        if (valid)
            val = acc[ia];
        else {val.x = 0.0; val.y = 0.0;}
        REAL2 min_max = block_reduction_min_max_2(val.x, val.y);
        if (threadIdx.y == 0) {
            acc[m * blockIdx.y + gid_0] = min_max;
        }
    }

    __global__ void min_max_reduce (const uint32_t m, const uint32_t n, REAL2* acc,
                                    const REAL* a, const uint32_t offset, const uint32_t ld) {
        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        const REAL val = (valid) ? a[offset + ld * gid_1 + gid_0] : 0.0;
        const REAL2 min_max = block_reduction_min_max_2(val, val);
        const bool write = valid && (threadIdx.y == 0);
        if (write) {
            acc[m * blockIdx.y + gid_0] = min_max;
        }
    }

    __global__ void bitonic_local(const uint32_t n, const REAL* in, REAL* out) {
        const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid < n) {
            const uint32_t lid = threadIdx.x;

            REAL2 value;
            value.x = (REAL)lid;
            value.y = in[gid];
            __shared__ REAL2 aux[WGS];
            aux[lid] = value;

            __syncthreads();

            for (uint32_t length = 1; length < WGS; length <<= 1) {
                const bool direction = ((lid & (length << 1)) != 0);
                for (uint32_t inc = length; inc > 0; inc >>= 1) {
                    const uint32_t j = lid ^ inc;
                    const REAL2 other_value = aux[j];
                    const bool smaller = (value.y < other_value.y)
                        || (other_value.y == value.y && j < lid);
                    const bool swap = smaller ^ (j < lid) ^ direction;
                    value = (swap) ? other_value : value;
                    __syncthreads();
                    aux[lid] = value;
                    __syncthreads();
                }
            }

            out[gid] = value.x;
        }
    }
    
    __global__ void sum_reduce_horizontal (const uint32_t m, const uint32_t n, REAL* acc, const REAL* data) {
        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t i = m * gid_1 + gid_0;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        const REAL sum = block_reduction_sum_2( (valid) ? data[i] : 0.0f);
        const bool write = valid && (threadIdx.y == 0);
        if (write) {
            acc[m * blockIdx.y + gid_0] = sum;
        }
    }

    __global__ void mean_reduce (const uint32_t m, const uint32_t n, ACCUMULATOR* acc,
                                 const REAL* a, const uint32_t offset_x, const uint32_t ld_x) {
        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t i = offset_x + ld_x * gid_1 + gid_0;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        const ACCUMULATOR sum = block_reduction_sum_2( (valid) ? a[i] : 0.0);
        const bool write = valid && (threadIdx.y == 0);
        if (write) {
            acc[m * blockIdx.y + gid_0] = sum;
        }
    }

    __global__ void variance_reduce (const uint32_t m, const uint32_t n, ACCUMULATOR* acc,
                                     const REAL* x, const uint32_t offset_x, const uint32_t ld_x,
                                     const REAL* mu) {
        const uint32_t gid_0 = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t gid_1 = blockIdx.y * blockDim.y + threadIdx.y;
        const uint32_t i = offset_x + ld_x * gid_1 + gid_0;
        const bool valid = (gid_0 < m) && (gid_1 < n);
        const REAL diff = (valid) ? x[i] - mu[gid_0] : 0.0;
        const ACCUMULATOR sum = block_reduction_sum_2(diff * diff);
        const bool write = valid && (threadIdx.y == 0);
        if (write) {
            acc[m * blockIdx.y + gid_0] = sum;
        }
    }

}

