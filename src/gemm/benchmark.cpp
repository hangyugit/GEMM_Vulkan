#include <filesystem>
#include <fstream>

#include "gemm/internal.hpp"

namespace gemm {

double compute_gflops(const ProblemSize& problem, double kernel_ms) {
  if (kernel_ms <= 0.0) {
    return 0.0;
  }

  const double flops = 2.0 * static_cast<double>(problem.m) *
                       static_cast<double>(problem.n) *
                       static_cast<double>(problem.k);
  return flops / (kernel_ms * 1.0e6);
}

double compute_bytes_moved(const ProblemSize& problem) {
  const double bytes_per_float = sizeof(float);
  const double a_bytes =
      bytes_per_float * static_cast<double>(problem.m) * problem.k;
  const double b_bytes =
      bytes_per_float * static_cast<double>(problem.k) * problem.n;
  const double c_bytes =
      bytes_per_float * static_cast<double>(problem.m) * problem.n * 2.0;
  return a_bytes + b_bytes + c_bytes;
}

void append_result_csv(const RunOptions& options, const RunResult& result) {
  if (options.benchmark.csv_output.empty()) {
    return;
  }

  const std::filesystem::path path = options.benchmark.csv_output;
  std::filesystem::create_directories(path.parent_path());

  const bool needs_header = !std::filesystem::exists(path);
  std::ofstream stream(path, std::ios::app);
  if (!stream) {
    throw std::runtime_error("Failed to open CSV output: " + path.string());
  }

  if (needs_header) {
    stream << "kernel_id,kernel_name,m,n,k,alpha,beta,warmup_iters,timed_iters,"
              "kernel_avg_ms,kernel_min_ms,kernel_max_ms,kernel_stddev_ms,"
              "wall_avg_ms,wall_min_ms,wall_max_ms,wall_stddev_ms,gflops,"
              "bytes_moved,verify_passed,max_abs_error,max_rel_error,"
              "workgroup_x,workgroup_y,workgroup_z,block_m,block_n,block_k,"
              "thread_m,thread_n,vector_width,use_shared,use_subgroup\n";
  }

  stream << options.kernel_id << ',' << '"' << result.kernel.name << '"' << ','
         << options.problem.m << ',' << options.problem.n << ','
         << options.problem.k << ',' << options.scalars.alpha << ','
         << options.scalars.beta << ',' << options.benchmark.warmup_iterations
         << ',' << options.benchmark.timed_iterations << ','
         << result.timing.kernel_ms.average << ','
         << result.timing.kernel_ms.minimum << ','
         << result.timing.kernel_ms.maximum << ','
         << result.timing.kernel_ms.stddev << ','
         << result.timing.wall_ms.average << ','
         << result.timing.wall_ms.minimum << ','
         << result.timing.wall_ms.maximum << ',' << result.timing.wall_ms.stddev
         << ',' << result.gflops << ',' << result.bytes_moved << ','
         << (result.verification.passed ? 1 : 0) << ','
         << result.verification.max_abs_error << ','
         << result.verification.max_rel_error << ','
         << result.config.workgroup_x << ',' << result.config.workgroup_y << ','
         << result.config.workgroup_z << ',' << result.config.block_m << ','
         << result.config.block_n << ',' << result.config.block_k << ','
         << result.config.thread_m << ',' << result.config.thread_n << ','
         << result.config.vector_width << ','
         << (result.config.use_shared_memory ? 1 : 0) << ','
         << (result.config.use_subgroup ? 1 : 0) << '\n';
}

}  // namespace gemm
