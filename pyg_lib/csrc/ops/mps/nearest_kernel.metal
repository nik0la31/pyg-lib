#include <metal_stdlib>

using namespace metal;

#define NEAREST_THREADS 1024

#define NEAREST_OP_ARGUMENTS(t) \
    device const t* x [[buffer(0)]], \
    device const t* y [[buffer(1)]], \
    device const long* ptr_x [[buffer(2)]], \
    device const long* ptr_y [[buffer(3)]], \
    device long* out [[buffer(4)]], \
    constant long& batch_size, \
    constant long& dim, \
    uint3 threadgroup_pos [[threadgroup_position_in_grid]], \
    uint3 threads_per_threadgroup [[threads_per_threadgroup]], \
    uint  thread_idx [[thread_index_in_threadgroup]]
    
template <typename T>
kernel void nearest_op(NEAREST_OP_ARGUMENTS(T)) {
    const uint n_x = threadgroup_pos.x;

    long batch_idx = 0;
    for (long b = 0; b < batch_size; ++b) {
        if (n_x >= ptr_x[b] && n_x < ptr_x[b + 1]) {
            batch_idx = b;
            break;
        }
    }

    const long y_start = ptr_y[batch_idx];
    const long y_end = ptr_y[batch_idx + 1];

    T best = T(1e38);
    long best_idx = y_start;
    for (long n_y = y_start + thread_idx; n_y < y_end; n_y += threads_per_threadgroup.x) {
        T dist = T(0);
        for (int d = 0; d < dim; ++d) {
            T diff = x[n_x * dim + d] - y[n_y * dim + d];
            dist += diff * diff;
        }
        if (dist < best) {
            best = dist;
            best_idx = n_y;
        }
    }

    threadgroup T best_dist[NEAREST_THREADS]; 
    threadgroup long best_dist_idx[NEAREST_THREADS];

    best_dist[thread_idx] = best;
    best_dist_idx[thread_idx] = best_idx;

    for (uint offset = threads_per_threadgroup.x / 2; offset > 0; offset /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (thread_idx < offset) {
            if (best_dist[thread_idx + offset] < best_dist[thread_idx]) {
                best_dist[thread_idx] = best_dist[thread_idx + offset];
                best_dist_idx[thread_idx] = best_dist_idx[thread_idx + offset];
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_idx == 0) {
        out[n_x] = best_dist_idx[0];
    }
}
    
#define INSTANTIATE_NEAREST_OP(T, NAME) \
    template [[host_name("nearest_op_" #NAME)]] \
    kernel void nearest_op<T>(NEAREST_OP_ARGUMENTS(T));

INSTANTIATE_NEAREST_OP(float, float)
INSTANTIATE_NEAREST_OP(half, half)
INSTANTIATE_NEAREST_OP(bfloat, bfloat)
INSTANTIATE_NEAREST_OP(long, long)
INSTANTIATE_NEAREST_OP(int, int)
INSTANTIATE_NEAREST_OP(short, short)
INSTANTIATE_NEAREST_OP(char, char)
INSTANTIATE_NEAREST_OP(uchar, uchar)

#undef INSTANTIATE_NEAREST_OP
#undef NEAREST_OP_ARGUMENTS