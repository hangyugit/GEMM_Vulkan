#pragma once

#include <cstdint>
#include <string>

#include "gemm/config.hpp"

namespace gemm {

struct VerificationResult {
  bool passed = true;
  float max_abs_error = 0.0F;
  float max_rel_error = 0.0F;
  std::uint32_t row = 0;
  std::uint32_t col = 0;
};

struct SampleStats {
  double average = 0.0;
  double minimum = 0.0;
  double maximum = 0.0;
  double stddev = 0.0;
};

struct TimingResult {
  SampleStats kernel_ms;
  SampleStats wall_ms;
};

struct RunResult {
  KernelInfo kernel;
  KernelConfig config;
  VerificationResult verification;
  TimingResult timing;
  double gflops = 0.0;
  double bytes_moved = 0.0;
  std::string notes;
};

}  // namespace gemm
