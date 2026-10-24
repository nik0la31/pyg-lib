#include "../cluster.h"

#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/TensorFactory.h>
#include <torch/library.h>

#include "utils.h"

namespace pyg {
namespace ops {

at::Tensor grid_cluster_mps(const at::Tensor& pos,
                             const at::Tensor& size,
                             const std::optional<at::Tensor>& optional_start,
                             const std::optional<at::Tensor>& optional_end) {
  TORCH_CHECK(pos.is_mps(), "pos must be an MPS tensor");
  TORCH_CHECK(pos.is_contiguous(), "pos must be contiguous");

  const auto tname = at::native::mps::scalarToMetalTypeString(pos);

  auto N = pos.size(0);
  auto D = pos.size(1);

  at::Tensor start;
  if (optional_start.has_value()) {
    start = optional_start.value().contiguous();
  }
  else {
    start = std::get<0>(pos.min(0));
  }

  at::Tensor end;
  if (optional_end.has_value()) {
    end = optional_end.value().contiguous();
  }
  else {
    end = std::get<0>(pos.max(0));
  }

  auto out = at::empty({N}, pos.options().dtype(at::kLong));
  auto stream = at::mps::getCurrentMPSStream();

  AT_DISPATCH_MPS_TYPES(pos.scalar_type(), "grid_cluster_mps", [&]() {
    auto& lib = getPygMetalLibrary();
    const auto pipeline = lib.getPipelineStateForFunc("grid_cluster_op_" + tname);

    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];
        at::native::mps::mtl_setArgs<0>(
            encoder, pos, size, start, end, out, D);
        at::native::mps::mtl_dispatch1DJob(
            encoder, pipeline, static_cast<NSUInteger>(N));
      }
    });
  });

  return out;
}

TORCH_LIBRARY_IMPL(pyg, MPS, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::grid_cluster"), TORCH_FN(grid_cluster_mps));
}

}  // namespace ops
}  // namespace pyg
