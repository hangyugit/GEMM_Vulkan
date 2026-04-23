#ifndef GEMM_VULKAN_BOUNDS_GLSL
#define GEMM_VULKAN_BOUNDS_GLSL

bool out_of_bounds(uvec2 coord, uint rows, uint cols) {
  return coord.y >= rows || coord.x >= cols;
}

#endif
