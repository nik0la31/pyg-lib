#include "../nearest.h"

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/TensorFactory.h>
#include <torch/library.h>

#include "utils.h"

namespace pyg {
namespace ops {

#define NEAREST_THREADS 1024

at::Tensor nearest(const at::Tensor& x,
                    const at::Tensor& y,
                    const std::optional<at::Tensor>& ptr_x,
                    const std::optional<at::Tensor>& ptr_y) {

    TORCH_CHECK(x.is_mps() && y.is_mps(), "Input must be on MPS tensors");
    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(), "Input tensors must be contiguous");

    const auto tname = at::native::mps::scalarToMetalTypeString(x);

    std::optional<at::Tensor> ptr_x_v = ptr_x;
    std::optional<at::Tensor> ptr_y_v = ptr_y;

    if (!ptr_x_v.has_value()) {
        ptr_x_v = at::arange(0, x.size(0) + 1, x.size(0), x.options().dtype(at::kLong));
    }
    if (!ptr_y_v.has_value()) {
        ptr_y_v = at::arange(0, y.size(0) + 1, y.size(0), y.options().dtype(at::kLong));
    }

    auto out = at::empty({x.size(0)}, ptr_x_v.value().options());
    auto stream = at::mps::getCurrentMPSStream();

    AT_DISPATCH_MPS_TYPES(x.scalar_type(), "nearest_mps", [&] {
        auto& lib = getMPSLibrary();
        const auto pipeline = lib.getPipelineStateForFunc("nearest_op_" + tname);

        at::mps::dispatch_sync_with_rethrow(stream->commandQueue(), ^() {
            @autoreleasepool {
                auto encoder = stream->commandEncoder();
                [encoder setComputePipelineState:pipeline];
                at::native::mps::mtl_setArgs<0>(
                    x, y, ptr_x_v.value(), ptr_y_v.value(), out, ptr_x_v.value().size(0) - 1, x.size(1));
                
                MTLSize threadsPerGrid = MTLSizeMake(x.size(0) * NEAREST_THREADS, 1, 1);
                MTLSize threadsPerThreadgroup = MTLSizeMake(NEAREST_THREADS, 1, 1);
                [encoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            }
        });
    });

    return out;
}

} // namespace ops
} // namespace pyg