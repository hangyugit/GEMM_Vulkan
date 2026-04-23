#include "gemm/configs/default_configs.hpp"

#ifndef GEMM_VULKAN_SHADER_DIR
#define GEMM_VULKAN_SHADER_DIR ""
#endif

namespace gemm {

std::vector<KernelDefinition> make_default_kernel_definitions() {
  const std::filesystem::path shader_root = GEMM_VULKAN_SHADER_DIR;

  return {
      {
          .info =
              {
                  .id = 0,
                  .name = "00_naive",
                  .shader_name = "sgemm/00_naive.comp.spv",
                  .description = "One invocation computes one C element from "
                                 "global memory.",
              },
          .config =
              {
                  .workgroup_x = 32,
                  .workgroup_y = 32,
                  .workgroup_z = 1,
                  .block_m = 32,
                  .block_n = 32,
                  .block_k = 1,
                  .thread_m = 1,
                  .thread_n = 1,
                  .vector_width = 1,
                  .use_shared_memory = false,
                  .use_subgroup = false,
                  .vendor = VendorTag::kAny,
              },
          .shader_path = shader_root / "sgemm/00_naive.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 1,
                  .name = "01_coalesced",
                  .shader_name = "sgemm/01_coalesced.comp.spv",
                  .description = "Access-pattern cleanup stage.",
              },
          .config = {},
          .shader_path = shader_root / "sgemm/01_coalesced.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 2,
                  .name = "02_shared_tiling",
                  .shader_name = "sgemm/02_shared_tiling.comp.spv",
                  .description = "Shared-memory tiled block GEMM stage.",
              },
          .config =
              {
                  .block_k = 32,
                  .use_shared_memory = true,
              },
          .shader_path = shader_root / "sgemm/02_shared_tiling.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 3,
                  .name = "03_thread_tiling_1d",
                  .shader_name = "sgemm/03_thread_tiling_1d.comp.spv",
                  .description = "1D thread-tiling stage.",
              },
          .config =
              {
                  .workgroup_x = 512,
                  .workgroup_y = 1,
                  .workgroup_z = 1,
                  .block_m = 64,
                  .block_n = 64,
                  .block_k = 8,
                  .thread_m = 8,
                  .thread_n = 1,
                  .vector_width = 1,
                  .use_shared_memory = true,
                  .use_subgroup = false,
                  .dispatch_order = DispatchOrder::kColumnsThenRows,
                  .vendor = VendorTag::kAny,
              },
          .shader_path = shader_root / "sgemm/03_thread_tiling_1d.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 4,
                  .name = "04_thread_tiling_1d_shared_vec",
                  .shader_name = "sgemm/03_thread_tiling_1d_shared_vec.comp.spv",
                  .description = "1D thread-tiling stage with vec4 shared "
                                 "memory loads.",
              },
          .config =
              {
                  .workgroup_x = 512,
                  .workgroup_y = 1,
                  .workgroup_z = 1,
                  .block_m = 64,
                  .block_n = 64,
                  .block_k = 8,
                  .thread_m = 8,
                  .thread_n = 1,
                  .vector_width = 4,
                  .use_shared_memory = true,
                  .use_subgroup = false,
                  .dispatch_order = DispatchOrder::kColumnsThenRows,
                  .vendor = VendorTag::kAny,
              },
          .shader_path =
              shader_root / "sgemm/03_thread_tiling_1d_shared_vec.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 5,
                  .name = "04_thread_tiling_2d",
                  .shader_name = "sgemm/04_thread_tiling_2d.comp.spv",
                  .description = "2D thread-tiling stage.",
              },
          .config = 
              {
                  .workgroup_x = 256,
                  .workgroup_y = 1,
                  .workgroup_z = 1,
                  .block_m = 128,
                  .block_n = 128,
                  .block_k = 8,
                  .thread_m = 8,
                  .thread_n = 8,
                  .vector_width = 1,
                  .use_shared_memory = true,
                  .use_subgroup = false,
                  .dispatch_order = DispatchOrder::kColumnsThenRows,
                  .vendor = VendorTag::kAny,
              },
          .shader_path = shader_root / "sgemm/04_thread_tiling_2d.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 6,
                  .name = "05_vectorized",
                  .shader_name = "sgemm/05_vectorized.comp.spv",
                  .description = "Vectorized memory access stage.",
              },
          .config = 
              {
                  .workgroup_x = 256,
                  .workgroup_y = 1,
                  .workgroup_z = 1,
                  .block_m = 128,
                  .block_n = 128,
                  .block_k = 8,
                  .thread_m = 8,
                  .thread_n = 8,
                  .vector_width = 4,
                  .use_shared_memory = true,
                  .use_subgroup = false,
                  .dispatch_order = DispatchOrder::kColumnsThenRows,
                  .vendor = VendorTag::kAny,               
              },
          .shader_path = shader_root / "sgemm/05_vectorized.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 7,
                  .name = "06_bank_conflict_avoid",
                  .shader_name = "sgemm/06_bank_conflict_avoid.comp.spv",
                  .description = "Bank-conflict mitigation stage.",
              },
          .config = 
              {
                  .workgroup_x = 256,
                  .workgroup_y = 1,
                  .workgroup_z = 1,
                  .block_m = 128,
                  .block_n = 128,
                  .block_k = 8,
                  .thread_m = 8,
                  .thread_n = 8,
                  .vector_width = 4,
                  .use_shared_memory = true,
                  .use_subgroup = false,
                  .dispatch_order = DispatchOrder::kColumnsThenRows,
                  .vendor = VendorTag::kAny,             
              },
          .shader_path = shader_root / "sgemm/06_bank_conflict_avoid.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 8,
                  .name = "07_bank_conflict_padding",
                  .shader_name = "sgemm/07_bank_conflict_padding.comp.spv",
                  .description = "Bank-conflict mitigation using padded shared "
                                 "memory stride.",
              },
          .config =
              {
                  .workgroup_x = 256,
                  .workgroup_y = 1,
                  .workgroup_z = 1,
                  .block_m = 128,
                  .block_n = 128,
                  .block_k = 8,
                  .thread_m = 8,
                  .thread_n = 8,
                  .vector_width = 4,
                  .use_shared_memory = true,
                  .use_subgroup = false,
                  .dispatch_order = DispatchOrder::kColumnsThenRows,
                  .vendor = VendorTag::kAny,
              },
          .shader_path =
              shader_root / "sgemm/07_bank_conflict_padding.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 9,
                  .name = "08_subgroup_tiling",
                  .shader_name = "sgemm/08_subgroup_tiling.comp.spv",
                  .description = "Subgroup-centric tiling stage.",
              },
          .config = 
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
                  .vendor = VendorTag::kAny,
            },
          .shader_path = shader_root / "sgemm/08_subgroup_tiling.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 10,
                  .name = "08_subgroup_tiling_compiler_friendly",
                  .shader_name =
                      "sgemm/08_subgroup_tiling_compiler_friendly.comp.spv",
                  .description = "Subgroup tiling stage with scalarized "
                                 "compiler-friendly accumulators.",
              },
          .config =
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
                  .vendor = VendorTag::kAny,
              },
          .shader_path =
              shader_root /
              "sgemm/08_subgroup_tiling_compiler_friendly.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 11,
                  .name = "09_double_buffering",
                  .shader_name = "sgemm/09_double_buffering.comp.spv",
                  .description = "Double-buffered mainloop stage.",
              },
          .config = 
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
                  .vendor = VendorTag::kAny,
            },
          .shader_path = shader_root / "sgemm/09_double_buffering.comp.spv",
          .implemented = true,
      },
      {
          .info =
              {
                  .id = 12,
                  .name = "10_tile_swizzle",
                  .shader_name = "sgemm/10_tile_swizzle.comp.spv",
                  .description = "Tile swizzle stage using swizzled workgroup-to-C-tile mapping.",
              },
          .config =
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
                  .vendor = VendorTag::kAny,
              },
      .shader_path = shader_root / "sgemm/10_tile_swizzle.comp.spv",
      .implemented = true,
    },
    {
        .info =
            {
                .id = 13,
                .name = "11_autotuned",
                .shader_name = "sgemm/11_autotuned.comp.spv",
                .description =
                    "Autotune candidate slot based on double buffering.",
            },
        .config =
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
                .vendor = VendorTag::kAny,
        },
        .shader_path = shader_root / "sgemm/11_autotuned.comp.spv",
        .implemented = true,
    },
    {
        .info =
            {
                .id = 14,
                .name = "12_split_k",
                .shader_name = "sgemm/12_split_k_partial.comp.spv",
                .description =
                    "Two-pass Split-K GEMM using partial buffer reduction.",
            },
        .config =
            {
                .workgroup_x = 16,
                .workgroup_y = 16,
                .workgroup_z = 1,
                .block_m = 16,
                .block_n = 16,
                .block_k = 1,
                .thread_m = 1,
                .thread_n = 1,
                .vector_width = 1,
                .use_shared_memory = false,
                .use_subgroup = false,
                .dispatch_order = DispatchOrder::kColumnsThenRows,
                .vendor = VendorTag::kAny,
            },
        .shader_path = shader_root / "sgemm/12_split_k_partial.comp.spv",
        .implemented = true,
    },
    {
        .info =
            {
                .id = 15,
                .name = "13_persistent_scheduler",
                .shader_name = "sgemm/13_persistent_scheduler.comp.spv",
                .description =
                    "Persistent work-queue scheduler over 16x16 output tiles.",
            },
        .config =
            {
                .workgroup_x = 16,
                .workgroup_y = 16,
                .workgroup_z = 1,
                .block_m = 16,
                .block_n = 16,
                .block_k = 1,
                .thread_m = 1,
                .thread_n = 1,
                .vector_width = 1,
                .use_shared_memory = false,
                .use_subgroup = false,
                .dispatch_order = DispatchOrder::kColumnsThenRows,
                .vendor = VendorTag::kAny,
            },
        .shader_path = shader_root / "sgemm/13_persistent_scheduler.comp.spv",
        .implemented = true,
    },
    {
        .info =
            {
                .id = 16,
                .name = "14_stream_k",
                .shader_name = "sgemm/14_stream_k_partial.comp.spv",
                .description =
                    "Persistent Split-K work-unit scheduler with reduction pass.",
            },
        .config =
            {
                .workgroup_x = 16,
                .workgroup_y = 16,
                .workgroup_z = 1,
                .block_m = 16,
                .block_n = 16,
                .block_k = 1,
                .thread_m = 1,
                .thread_n = 1,
                .vector_width = 1,
                .use_shared_memory = false,
                .use_subgroup = false,
                .dispatch_order = DispatchOrder::kColumnsThenRows,
                .vendor = VendorTag::kAny,
            },
        .shader_path = shader_root / "sgemm/14_stream_k_partial.comp.spv",
        .implemented = true,
    },
  };
}

}  // namespace gemm
