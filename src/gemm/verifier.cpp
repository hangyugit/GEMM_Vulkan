#include "gemm/verifier.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gemm {

VerificationResult verify_results(const std::vector<float>& reference,
                                  const std::vector<float>& candidate,
                                  std::uint32_t rows, std::uint32_t cols,
                                  float abs_tolerance, float rel_tolerance) {
  if (reference.size() != candidate.size()) {
    throw std::runtime_error("Verification inputs have different sizes");
  }

  VerificationResult result{};
  for (std::size_t index = 0; index < reference.size(); ++index) {
    const float ref = reference[index];
    const float got = candidate[index];
    const float abs_error = std::fabs(ref - got);
    const float rel_error = abs_error / std::max(1.0F, std::fabs(ref));

    if (abs_error > result.max_abs_error) {
      result.max_abs_error = abs_error;
      result.max_rel_error = rel_error;
      result.row = static_cast<std::uint32_t>(index / cols);
      result.col = static_cast<std::uint32_t>(index % cols);
    }

    if (abs_error > abs_tolerance && rel_error > rel_tolerance) {
      result.passed = false;
    }
  }

  if (rows == 0 || cols == 0) {
    result.row = 0;
    result.col = 0;
  }

  return result;
}

}  // namespace gemm
