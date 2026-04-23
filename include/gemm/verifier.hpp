#pragma once

#include <vector>

#include "gemm/result.hpp"

namespace gemm {

VerificationResult verify_results(const std::vector<float>& reference,
                                  const std::vector<float>& candidate,
                                  std::uint32_t rows, std::uint32_t cols,
                                  float abs_tolerance = 1.0e-3F,
                                  float rel_tolerance = 1.0e-3F);

}  // namespace gemm
