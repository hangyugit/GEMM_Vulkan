#pragma once

#include <cstdint>
#include <string>

namespace gemm {

enum class DataLayout : std::uint32_t {
  kRowMajor = 0,
};

enum class VendorTag : std::uint32_t {
  kAny = 0,
  kNvidia = 1,
  kAmd = 2,
};

struct ProblemSize {
  std::uint32_t m = 1024;
  std::uint32_t n = 1024;
  std::uint32_t k = 1024;
};

struct Scalars {
  float alpha = 1.0F;
  float beta = 0.0F;
};

struct KernelInfo {
  int id = 0;
  std::string name;
  std::string shader_name;
  std::string description;
};

}  // namespace gemm
