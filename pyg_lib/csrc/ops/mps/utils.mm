#include "utils.h"

#include <dlfcn.h>
#include <filesystem>
#include <string>

namespace pyg {
namespace ops {

at::native::mps::PrecompiledMetalShaderLibrary& getPygMetalLibrary() {
  static at::native::mps::PrecompiledMetalShaderLibrary library([] {
    static const int module_anchor = 0;
    Dl_info image_info{};
    TORCH_CHECK(dladdr(&module_anchor, &image_info) != 0 &&
                    image_info.dli_fname,
                "Unable to locate the pyg-lib extension image");

    return (std::filesystem::path(image_info.dli_fname).parent_path() /
            "pyg.metallib")
        .string();
  }());
  return library;
}

}  // namespace ops
}  // namespace pyg
