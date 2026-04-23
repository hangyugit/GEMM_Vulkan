#include <random>

#include "gemm/internal.hpp"

namespace gemm {

HostMatrices make_host_matrices(const ProblemSize& problem,
                                std::uint32_t seed) {
  const std::size_t size_a =
      static_cast<std::size_t>(problem.m) * static_cast<std::size_t>(problem.k);
  const std::size_t size_b =
      static_cast<std::size_t>(problem.k) * static_cast<std::size_t>(problem.n);
  const std::size_t size_c =
      static_cast<std::size_t>(problem.m) * static_cast<std::size_t>(problem.n);

  HostMatrices matrices{};
  matrices.a.resize(size_a);
  matrices.b.resize(size_b);
  matrices.c_initial.resize(size_c);
  matrices.c_output.resize(size_c, 0.0F);
  matrices.c_reference.resize(size_c, 0.0F);

  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

  for (float& value : matrices.a) {
    value = distribution(generator);
  }
  for (float& value : matrices.b) {
    value = distribution(generator);
  }
  for (float& value : matrices.c_initial) {
    value = distribution(generator);
  }

  return matrices;
}

void reference_sgemm(const ProblemSize& problem, const Scalars& scalars,
                     const std::vector<float>& a, const std::vector<float>& b,
                     const std::vector<float>& c_initial,
                     std::vector<float>* c_out) {
  c_out->assign(c_initial.begin(), c_initial.end());

  for (std::uint32_t row = 0; row < problem.m; ++row) {
    for (std::uint32_t col = 0; col < problem.n; ++col) {
      float acc = 0.0F;
      for (std::uint32_t kk = 0; kk < problem.k; ++kk) {
        acc += a[static_cast<std::size_t>(row) * problem.k + kk] *
               b[static_cast<std::size_t>(kk) * problem.n + col];
      }
      (*c_out)[static_cast<std::size_t>(row) * problem.n + col] =
          scalars.alpha * acc +
          scalars.beta *
              c_initial[static_cast<std::size_t>(row) * problem.n + col];
    }
  }
}

}  // namespace gemm
