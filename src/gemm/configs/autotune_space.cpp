#include <vector>

#include "gemm/configs/default_configs.hpp"

namespace gemm {

std::vector<KernelConfig> default_autotune_space() {
  return {
      {
          .workgroup_x = 256,
          .workgroup_y = 1,
          .workgroup_z = 1,
          .block_m = 128,
          .block_n = 256,
          .block_k = 16,
          .thread_m = 8,
          .thread_n = 8,
          .vector_width = 4,
          .use_shared_memory = true,
          .use_subgroup = true,
          .dispatch_order = DispatchOrder::kColumnsThenRows,
      },
      {
          .workgroup_x = 128,
          .workgroup_y = 1,
          .workgroup_z = 1,
          .block_m = 128,
          .block_n = 128,
          .block_k = 16,
          .thread_m = 8,
          .thread_n = 4,
          .vector_width = 4,
          .use_shared_memory = true,
          .use_subgroup = true,
          .dispatch_order = DispatchOrder::kColumnsThenRows,
      },
      {
          .workgroup_x = 256,
          .workgroup_y = 1,
          .workgroup_z = 1,
          .block_m = 128,
          .block_n = 256,
          .block_k = 32,
          .thread_m = 8,
          .thread_n = 8,
          .vector_width = 4,
          .use_shared_memory = true,
          .use_subgroup = true,
          .dispatch_order = DispatchOrder::kColumnsThenRows,
      },
      {
          .workgroup_x = 256,
          .workgroup_y = 1,
          .workgroup_z = 1,
          .block_m = 256,
          .block_n = 128,
          .block_k = 16,
          .thread_m = 8,
          .thread_n = 8,
          .vector_width = 4,
          .use_shared_memory = true,
          .use_subgroup = true,
          .dispatch_order = DispatchOrder::kColumnsThenRows,
      },
  };
}

}  // namespace gemm
