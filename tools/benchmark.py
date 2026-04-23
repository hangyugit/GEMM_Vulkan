#!/usr/bin/env python3

import argparse
import csv
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GEMM_VULKAN benchmarks."
    )
    parser.add_argument("--exe", default="build/gemm_vulkan")
    parser.add_argument("--kernel", type=int, default=0)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048]
    )
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", default="results/benchmark/benchmark.csv")
    args = parser.parse_args()

    exe = Path(args.exe)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for size in args.sizes:
        command = [
            str(exe),
            "--kernel",
            str(args.kernel),
            "--m",
            str(size),
            "--n",
            str(size),
            "--k",
            str(size),
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--output",
            str(output),
        ]
        subprocess.run(command, check=True)
        rows.append({"size": size})

    summary_path = output.with_name(output.stem + "_sizes.csv")
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["size"])
        writer.writeheader()
        writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
