# GEMM_VULKAN Roadmap

1. Build a correctness-first FP32 SGEMM baseline.
2. Add benchmark, validation, and GPU timestamp profiling.
3. Implement the optimization ladder from naive to subgroup-tiling.
4. Split vendor-agnostic and vendor-tuned variants.
5. Refactor into a CUTLASS-like hierarchy once multiple kernels exist.
