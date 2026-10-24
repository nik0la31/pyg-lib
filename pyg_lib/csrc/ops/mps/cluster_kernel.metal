#include <metal_stdlib>

using namespace metal;

#define GRID_CLUSTER_OP_ARGUMENTS(T) \
    device const T* pos [[buffer(0)]], \
    device const T* size [[buffer(1)]], \
    device const T* start [[buffer(2)]], \
    device const T* end [[buffer(3)]], \
    device long* out [[buffer(4)]], \
    constant long& D [[buffer(5)]], \
    uint thread_idx [[thread_position_in_grid]]

template <typename T>
kernel void grid_cluster_op(GRID_CLUSTER_OP_ARGUMENTS(T)) {

    long c = 0, k = 1;
    for (long d = 0; d < D; ++d) {
        T p = pos[ thread_idx * D + d] - start[d];
        c += (long)(p / size[d]) * k;
        k *= (long)((end[d] - start[d]) / size[d]) + 1;
    }
    out[thread_idx] = c;
}

#define INSTANTIATE_GRID_CLUSTER_OP(T, NAME) \
	template [[host_name("grid_cluster_op_" #NAME)]] \
	kernel void grid_cluster_op<T>(GRID_CLUSTER_OP_ARGUMENTS(T));

INSTANTIATE_GRID_CLUSTER_OP(float, float)
INSTANTIATE_GRID_CLUSTER_OP(half, half)
INSTANTIATE_GRID_CLUSTER_OP(bfloat, bfloat)
INSTANTIATE_GRID_CLUSTER_OP(long, long)
INSTANTIATE_GRID_CLUSTER_OP(int, int)
INSTANTIATE_GRID_CLUSTER_OP(short, short)
INSTANTIATE_GRID_CLUSTER_OP(char, char)
INSTANTIATE_GRID_CLUSTER_OP(uchar, uchar)

#undef INSTANTIATE_GRID_CLUSTER_OP
#undef GRID_CLUSTER_OP_ARGUMENTS
