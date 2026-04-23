file(REMOVE_RECURSE
  "CMakeFiles/gemm_vulkan_shaders"
  "shaders/sgemm/00_naive.comp.spv"
  "shaders/sgemm/01_coalesced.comp.spv"
  "shaders/sgemm/02_shared_tiling.comp.spv"
  "shaders/sgemm/03_thread_tiling_1d.comp.spv"
  "shaders/sgemm/03_thread_tiling_1d_shared_vec.comp.spv"
  "shaders/sgemm/04_thread_tiling_2d.comp.spv"
  "shaders/sgemm/05_vectorized.comp.spv"
  "shaders/sgemm/06_bank_conflict_avoid.comp.spv"
  "shaders/sgemm/07_bank_conflict_padding.comp.spv"
  "shaders/sgemm/08_subgroup_tiling.comp.spv"
  "shaders/sgemm/08_subgroup_tiling_compiler_friendly.comp.spv"
  "shaders/sgemm/09_double_buffering.comp.spv"
  "shaders/sgemm/10_tile_swizzle.comp.spv"
  "shaders/sgemm/11_autotuned.comp.spv"
  "shaders/sgemm/12_split_k_partial.comp.spv"
  "shaders/sgemm/12_split_k_reduce.comp.spv"
  "shaders/sgemm/13_persistent_scheduler.comp.spv"
  "shaders/sgemm/14_stream_k_partial.comp.spv"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/gemm_vulkan_shaders.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
