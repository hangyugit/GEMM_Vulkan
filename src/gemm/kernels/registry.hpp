#pragma once

#include <vector>

#include "gemm/internal.hpp"

namespace gemm {

const std::vector<KernelDefinition>& kernel_registry();
const KernelDefinition& get_kernel_definition(int kernel_id);

}  // namespace gemm
