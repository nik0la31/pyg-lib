#pragma once

#include <ATen/native/mps/MetalShaderLibrary.h>

namespace pyg {
namespace ops {

// Returns the shared `pyg.metallib` library, loaded once on first use.
at::native::mps::PrecompiledMetalShaderLibrary& getPygMetalLibrary();

}  // namespace ops
}  // namespace pyg
