#pragma once

#include <vector>

#include "gemm/internal.hpp"

namespace gemm {

std::vector<KernelDefinition> make_default_kernel_definitions();

}  // namespace gemm
