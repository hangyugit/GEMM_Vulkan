#ifndef GEMM_VULKAN_LOADS_GLSL
#define GEMM_VULKAN_LOADS_GLSL

uint row_major_index(uint row, uint col, uint stride) {
  return row * stride + col;
}

#endif
