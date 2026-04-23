#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "gemm/runner.hpp"

namespace {

void print_usage() {
  std::cout << "Usage: gemm_vulkan [options]\n"
            << "  --list-kernels\n"
            << "  --kernel <id>\n"
            << "  --m <rows>\n"
            << "  --n <cols>\n"
            << "  --k <depth>\n"
            << "  --alpha <value>\n"
            << "  --beta <value>\n"
            << "  --warmup <iters>\n"
            << "  --iters <iters>\n"
            << "  --seed <value>\n"
            << "  --no-verify\n"
            << "  --no-profile\n"
            << "  --output <csv path>\n"
            << "  --spec-local-size-x <threads>\n"
            << "  --spec-bm <rows>\n"
            << "  --spec-bn <cols>\n"
            << "  --spec-bk <depth>\n"
            << "  --spec-wm <rows>\n"
            << "  --spec-wn <cols>\n"
            << "  --spec-wniter <iters>\n"
            << "  --spec-tm <rows>\n"
            << "  --spec-tn <cols>\n"
            << "  --spec-stages <stages>\n"
            << "  --dispatch-order <columns_then_rows|rows_then_columns>\n"
            << "  --splits <count>\n"
            << "  --scheduler-workgroups <count>\n";
}

std::string require_value(int argc, char** argv, int* index,
                          std::string_view flag) {
  if (*index + 1 >= argc) {
    throw std::runtime_error("Missing value for flag " + std::string(flag));
  }
  *index += 1;
  return argv[*index];
}

std::uint32_t parse_u32(int argc, char** argv, int* index,
                        std::string_view flag) {
  return static_cast<std::uint32_t>(
      std::stoul(require_value(argc, argv, index, flag)));
}

gemm::AutotuneOptions& ensure_autotune_options(gemm::RunOptions* options) {
  if (!options->autotune.has_value()) {
    options->autotune = gemm::AutotuneOptions{};
  }
  return *options->autotune;
}

gemm::DispatchOrder parse_dispatch_order(const std::string& value) {
  if (value == "columns_then_rows") {
    return gemm::DispatchOrder::kColumnsThenRows;
  }
  if (value == "rows_then_columns") {
    return gemm::DispatchOrder::kRowsThenColumns;
  }
  throw std::runtime_error(
      "dispatch order must be columns_then_rows or rows_then_columns");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    gemm::RunOptions options{};
    bool list_only = false;

    for (int index = 1; index < argc; ++index) {
      const std::string arg = argv[index];
      if (arg == "--help" || arg == "-h") {
        print_usage();
        return EXIT_SUCCESS;
      }
      if (arg == "--list-kernels") {
        list_only = true;
        continue;
      }
      if (arg == "--kernel") {
        options.kernel_id = std::stoi(require_value(argc, argv, &index, arg));
        continue;
      }
      if (arg == "--m") {
        options.problem.m = parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--n") {
        options.problem.n = parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--k") {
        options.problem.k = parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--alpha") {
        options.scalars.alpha =
            std::stof(require_value(argc, argv, &index, arg));
        continue;
      }
      if (arg == "--beta") {
        options.scalars.beta =
            std::stof(require_value(argc, argv, &index, arg));
        continue;
      }
      if (arg == "--warmup") {
        options.benchmark.warmup_iterations = parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--iters") {
        options.benchmark.timed_iterations = parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--seed") {
        options.benchmark.random_seed = parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--no-verify") {
        options.benchmark.verify = false;
        continue;
      }
      if (arg == "--no-profile") {
        options.benchmark.profile = false;
        continue;
      }
      if (arg == "--output") {
        options.benchmark.csv_output = require_value(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--spec-local-size-x") {
        ensure_autotune_options(&options).local_size_x =
            parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--spec-bm") {
        ensure_autotune_options(&options).block_m =
            parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--spec-bn") {
        ensure_autotune_options(&options).block_n =
            parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--spec-bk") {
        ensure_autotune_options(&options).block_k =
            parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--spec-wm") {
        ensure_autotune_options(&options).warp_m =
            parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--spec-wn") {
        ensure_autotune_options(&options).warp_n =
            parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--spec-wniter") {
        ensure_autotune_options(&options).warp_n_iter =
            parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--spec-tm") {
        ensure_autotune_options(&options).thread_m =
            parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--spec-tn") {
        ensure_autotune_options(&options).thread_n =
            parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--spec-stages") {
        ensure_autotune_options(&options).stages =
            parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--dispatch-order") {
        ensure_autotune_options(&options).dispatch_order =
            parse_dispatch_order(require_value(argc, argv, &index, arg));
        continue;
      }
      if (arg == "--splits") {
        options.split_count = parse_u32(argc, argv, &index, arg);
        continue;
      }
      if (arg == "--scheduler-workgroups") {
        options.scheduler_workgroups = parse_u32(argc, argv, &index, arg);
        continue;
      }
      throw std::runtime_error("Unknown argument: " + arg);
    }

    if (list_only) {
      for (const gemm::KernelInfo& kernel : gemm::list_kernels()) {
        std::cout << kernel.id << ": " << kernel.name << " - "
                  << kernel.description << '\n';
      }
      return EXIT_SUCCESS;
    }

    const gemm::RunResult result = gemm::run(options);
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "kernel: " << result.kernel.name << '\n';
    std::cout << "problem: M=" << options.problem.m
              << " N=" << options.problem.n << " K=" << options.problem.k
              << '\n';
    std::cout << "kernel_ms: avg=" << result.timing.kernel_ms.average
              << " min=" << result.timing.kernel_ms.minimum
              << " max=" << result.timing.kernel_ms.maximum
              << " stddev=" << result.timing.kernel_ms.stddev << '\n';
    std::cout << "wall_ms: avg=" << result.timing.wall_ms.average
              << " min=" << result.timing.wall_ms.minimum
              << " max=" << result.timing.wall_ms.maximum
              << " stddev=" << result.timing.wall_ms.stddev << '\n';
    std::cout << "gflops: " << result.gflops << '\n';
    std::cout << "bytes_moved: " << result.bytes_moved << '\n';
    if (options.benchmark.verify) {
      std::cout << "verify: " << (result.verification.passed ? "PASS" : "FAIL")
                << '\n';
      std::cout << "max_abs_error: " << result.verification.max_abs_error
                << '\n';
      std::cout << "max_rel_error: " << result.verification.max_rel_error
                << '\n';
    } else {
      std::cout << "verify: SKIPPED\n";
    }
    std::cout << "workgroup: " << result.config.workgroup_x << "x"
              << result.config.workgroup_y << "x" << result.config.workgroup_z
              << '\n';
    std::cout << "tile: BM=" << result.config.block_m
              << " BN=" << result.config.block_n
              << " BK=" << result.config.block_k << '\n';
    if (!result.notes.empty()) {
      std::cout << "notes: " << result.notes << '\n';
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
