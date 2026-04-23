#ifndef GEMM_VULKAN_COMMON_GLSL
#define GEMM_VULKAN_COMMON_GLSL

layout(push_constant) uniform PushConstants {
  uint M;
  uint N;
  uint K;
  float alpha;
  float beta;
}
pc;

#endif
