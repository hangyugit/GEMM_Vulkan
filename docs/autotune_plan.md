# Autotune Plan

## Goal

Use `09_double_buffering` as the stable baseline and create compile-time shader
variants. Autotune should compare variants that differ only by tile shape,
thread tile shape, vector width, raster order, swizzle factor, or pipeline stage
count.

## First Variant

`shaders/sgemm/11_autotuned.comp` is intentionally a skeleton. Copy the body of
`09_double_buffering.comp` into it, then replace the constants at the top:

```glsl
const uint BM = 128;
const uint BN = 256;
const uint BK = 16;
const uint TM = 8;
const uint TN = 8;
const uint NUM_THREADS = 256;
const uint STAGES = 2;
```

Keep these as compile-time constants. Do not move them into push constants while
you are studying performance. The compiler needs fixed sizes for shared memory,
loop unrolling, register allocation, and address simplification.

## Initial Search Space

Start with a small manual space before generating many files:

- `128x256x16`, `TM=8`, `TN=8`, `threads=256`
- `128x128x16`, `TM=8`, `TN=4`, `threads=128`
- `128x256x32`, `TM=8`, `TN=8`, `threads=256`
- `256x128x16`, `TM=8`, `TN=8`, `threads=256`

Only compare variants that pass verification.

## Constraints

- `BK % vector_width == 0`
- `BN % vector_width == 0`
- `shared_bytes = STAGES * (BM * BK + BK * BN) * 4`
- `workgroup_x <= maxComputeWorkGroupInvocations`
- Register pressure usually rises when `TM * TN` rises.
- More stages can reduce occupancy because shared memory usage grows.

## Run

Dry-run the candidate list:

```bash
./tools/autotune.py --dry-run
```

Run implemented candidates:

```bash
./tools/autotune.py --exe ./build/gemm_vulkan
```

After implementing `11_autotuned.comp`, set its candidate entry to
`"implemented": true` in `tools/autotune_double_buffering.json`.
