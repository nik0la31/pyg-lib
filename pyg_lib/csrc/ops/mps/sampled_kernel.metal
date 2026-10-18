#include <metal_stdlib>

using namespace metal;

#define SAMPLED_OP_ARGUMENTS(T) \
	device const T* left [[buffer(0)]], \
	device const T* right [[buffer(1)]], \
	device T* out [[buffer(2)]], \
	device const long* left_index [[buffer(3)]], \
	device const long* right_index [[buffer(4)]], \
	constant uint& fn_type [[buffer(5)]], \
	constant bool& has_left_index [[buffer(6)]], \
	constant bool& has_right_index [[buffer(7)]], \
	constant long& num_feats [[buffer(8)]], \
	constant long& numel [[buffer(9)]], \
	uint tid [[thread_position_in_grid]]

template <typename T>
kernel void sampled_op_kernel(SAMPLED_OP_ARGUMENTS(T)) {

    long i = tid / num_feats;
    if (has_left_index) {
        i = left_index[i];
    }

    long j = tid / num_feats;
    if (has_right_index) {
        j = right_index[j];
    }

    long k = tid % num_feats;

    T a = left[i * num_feats + k];
    T b = right[j * num_feats + k];

    T c = T(0);

    if (fn_type == 0) {
        c = a + b;
    } else if (fn_type == 1) {
        c = a - b;
    } else if (fn_type == 2) {
        c = a * b;
    } else if (fn_type == 3) {
        c = a / b;
    }

    out[tid] = c;

}

#define INSTANTIATE_SAMPLED_OP(T, NAME) \
	template [[host_name("sampled_op_" #NAME)]] \
	kernel void sampled_op_kernel<T>(SAMPLED_OP_ARGUMENTS(T));

INSTANTIATE_SAMPLED_OP(float, float)
INSTANTIATE_SAMPLED_OP(half, half)
INSTANTIATE_SAMPLED_OP(bfloat, bfloat)
INSTANTIATE_SAMPLED_OP(long, long)
INSTANTIATE_SAMPLED_OP(int, int)
INSTANTIATE_SAMPLED_OP(short, short)
INSTANTIATE_SAMPLED_OP(char, char)
INSTANTIATE_SAMPLED_OP(uchar, uchar)

#undef INSTANTIATE_SAMPLED_OP
#undef SAMPLED_OP_ARGUMENTS