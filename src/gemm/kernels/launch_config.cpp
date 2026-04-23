#include "gemm/kernels/launch_config.hpp"

namespace gemm {

namespace {

std::uint32_t ceil_div(std::uint32_t value, std::uint32_t divisor) {
  return (value + divisor - 1) / divisor;
}

}  // namespace

DispatchShape compute_dispatch_shape(const ProblemSize& problem,
                                     const KernelConfig& config) {
  DispatchShape dispatch{};
  const std::uint32_t row_groups = ceil_div(problem.m, config.block_m);
  const std::uint32_t column_groups = ceil_div(problem.n, config.block_n);

  if (config.dispatch_order == DispatchOrder::kColumnsThenRows) {
    dispatch.x = column_groups;
    dispatch.y = row_groups;
  } else {
    dispatch.x = row_groups;
    dispatch.y = column_groups;
  }
  dispatch.z = 1;
  return dispatch;
}

}  // namespace gemm
