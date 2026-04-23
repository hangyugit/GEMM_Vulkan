#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>

#include "gemm/types.hpp"

namespace gemm {

enum class DispatchOrder {
  kRowsThenColumns,
  kColumnsThenRows,
};

struct KernelConfig {
  std::uint32_t workgroup_x = 32;
  std::uint32_t workgroup_y = 32;
  std::uint32_t workgroup_z = 1;
  std::uint32_t block_m = 32;
  std::uint32_t block_n = 32;
  std::uint32_t block_k = 1;
  std::uint32_t thread_m = 1;
  std::uint32_t thread_n = 1;
  std::uint32_t vector_width = 1;
  bool use_shared_memory = false;
  bool use_subgroup = false;
  DispatchOrder dispatch_order = DispatchOrder::kRowsThenColumns;
  VendorTag vendor = VendorTag::kAny;
};

struct BenchmarkOptions {
  std::uint32_t warmup_iterations = 1;
  std::uint32_t timed_iterations = 10;
  bool verify = true;
  bool profile = true;
  std::uint32_t random_seed = 7;
  std::filesystem::path csv_output;
};

struct AutotuneOptions {
  std::uint32_t local_size_x = 256;
  std::uint32_t block_m = 128;
  std::uint32_t block_n = 256;
  std::uint32_t block_k = 16;
  std::uint32_t warp_m = 128;
  std::uint32_t warp_n = 32;
  std::uint32_t warp_n_iter = 1;
  std::uint32_t thread_m = 8;
  std::uint32_t thread_n = 8;
  std::uint32_t stages = 2;
  DispatchOrder dispatch_order = DispatchOrder::kColumnsThenRows;
};

struct RunOptions {
  int kernel_id = 0;
  ProblemSize problem{};
  Scalars scalars{};
  BenchmarkOptions benchmark{};
  std::optional<AutotuneOptions> autotune;
  std::uint32_t split_count = 4;
  std::uint32_t scheduler_workgroups = 256;
};

}  // namespace gemm
