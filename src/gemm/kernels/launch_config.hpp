#pragma once

#include <cstdint>

#include "gemm/config.hpp"

namespace gemm {

struct DispatchShape {
  std::uint32_t x = 1;
  std::uint32_t y = 1;
  std::uint32_t z = 1;
};

DispatchShape compute_dispatch_shape(const ProblemSize& problem,
                                     const KernelConfig& config);

}  // namespace gemm
