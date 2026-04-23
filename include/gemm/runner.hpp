#pragma once

#include <vector>

#include "gemm/config.hpp"
#include "gemm/result.hpp"

namespace gemm {

std::vector<KernelInfo> list_kernels();
RunResult run(const RunOptions& options);

}  // namespace gemm
