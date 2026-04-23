#include "gemm/kernels/registry.hpp"

#include "gemm/configs/default_configs.hpp"

namespace gemm {

const std::vector<KernelDefinition>& kernel_registry() {
  static const std::vector<KernelDefinition> registry =
      make_default_kernel_definitions();
  return registry;
}

const KernelDefinition& get_kernel_definition(int kernel_id) {
  const auto& registry = kernel_registry();
  for (const KernelDefinition& definition : registry) {
    if (definition.info.id == kernel_id) {
      return definition;
    }
  }

  throw std::runtime_error("Unknown kernel id: " + std::to_string(kernel_id));
}

}  // namespace gemm
