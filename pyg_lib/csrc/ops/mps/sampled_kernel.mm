#include <ATen/ATen.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/TensorFactory.h>
#include <torch/library.h>

#include "utils.h"

#include <cstdint>
#include <map>
#include <string>

namespace pyg {
namespace ops {

namespace {

enum FnType : uint32_t { ADD = 0, SUB = 1, MUL = 2, DIV = 3 };
const std::map<std::string, FnType> to_fn_type = {
  {"add", ADD},
  {"sub", SUB},
  {"mul", MUL},
  {"div", DIV},
};

at::Tensor sampled_op_mps(const at::Tensor& left,
                          const at::Tensor& right,
                          const at::optional<at::Tensor> left_index,
                          const at::optional<at::Tensor> right_index,
                          const std::string fn) {
  const auto tname = at::native::mps::scalarToMetalTypeString(left);

  auto dim_size = left.size(0);
  if (left_index.has_value() && !right_index.has_value()) {
    dim_size = right.size(0);
  } else if (left_index.has_value() && right_index.has_value()) {
    dim_size = left_index.value().size(0);
  }
  const auto num_feats = left.size(1);
  const auto numel = dim_size * num_feats;

  const auto out = left.new_empty({dim_size, num_feats});
  auto stream = at::mps::getCurrentMPSStream();
  const auto fn_type = static_cast<uint32_t>(to_fn_type.at(fn));

  AT_DISPATCH_MPS_TYPES(left.scalar_type(), "sampled_op_mps", [&]() {
    auto& lib = getPygMetalLibrary();
    const auto pipeline = lib.getPipelineStateForFunc("sampled_op_" + tname);
    
    at::mps::dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        auto encoder = stream->commandEncoder();
        [encoder setComputePipelineState:pipeline];
        at::native::mps::mtl_setArgs<0>(
            encoder, left, right, out, left_index, right_index, fn_type,
            left_index.has_value(), right_index.has_value(), num_feats);
        at::native::mps::mtl_dispatch1DJob(
            encoder, pipeline, static_cast<NSUInteger>(numel));
      }
    });
  });

  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(pyg, MPS, m) {
  m.impl(TORCH_SELECTIVE_NAME("pyg::sampled_op"), TORCH_FN(sampled_op_mps));
}

}  // namespace ops
}  // namespace pyg