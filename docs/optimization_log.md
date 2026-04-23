# Optimization Log

## Kernel 00: Naive

- Goal: build a working end-to-end Vulkan SGEMM baseline.
- Expected bottleneck: global-memory traffic and poor arithmetic intensity.
- Success criteria:
  - matches CPU reference within tolerance
  - reports kernel time and GFLOPS
  - establishes a reproducible baseline for later kernels
