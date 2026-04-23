#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize GEMM_VULKAN CSV results.")
    parser.add_argument("csv_path")
    args = parser.parse_args()

    best_row = None
    with Path(args.csv_path).open() as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if best_row is None or float(row["gflops"]) > float(best_row["gflops"]):
                best_row = row

    if best_row is None:
        print("No rows found.")
        return 1

    print(
        f'best kernel={best_row["kernel_name"]} '
        f'M={best_row["m"]} N={best_row["n"]} K={best_row["k"]} '
        f'GFLOPS={best_row["gflops"]}'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
