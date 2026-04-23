#pragma once

#include <filesystem>
#include <vector>

#include "gemm/config.hpp"
#include "gemm/result.hpp"
#include "gemm/verifier.hpp"

namespace gemm {

struct KernelDefinition {
  KernelInfo info;
  KernelConfig config;
  std::filesystem::path shader_path;
  bool implemented = false;
};

struct HostMatrices {
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> c_initial;
  std::vector<float> c_output;
  std::vector<float> c_reference;
};

struct BenchmarkSamples {
  std::vector<double> kernel_ms;
  std::vector<double> wall_ms;
};

HostMatrices make_host_matrices(const ProblemSize& problem, std::uint32_t seed);
void reference_sgemm(const ProblemSize& problem, const Scalars& scalars,
                     const std::vector<float>& a, const std::vector<float>& b,
                     const std::vector<float>& c_initial,
                     std::vector<float>* c_out);
double compute_gflops(const ProblemSize& problem, double kernel_ms);
double compute_bytes_moved(const ProblemSize& problem);
void append_result_csv(const RunOptions& options, const RunResult& result);

}  // namespace gemm
