#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot GEMM_VULKAN benchmark CSV.")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="results/benchmark/plot.png")
    args = parser.parse_args()

    sizes = []
    gflops = []
    with Path(args.csv_path).open() as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            sizes.append(int(row["m"]))
            gflops.append(float(row["gflops"]))

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, gflops, marker="o")
    plt.xlabel("Matrix size")
    plt.ylabel("GFLOPS")
    plt.title("GEMM_VULKAN Benchmark")
    plt.grid(True, alpha=0.3)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
