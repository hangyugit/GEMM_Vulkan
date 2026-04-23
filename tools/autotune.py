#!/usr/bin/env python3

import argparse
import itertools
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path


GFLOPS_RE = re.compile(r"^gflops:\s+([0-9.]+)", re.MULTILINE)
KERNEL_MS_RE = re.compile(
    r"^kernel_ms:\s+avg=([0-9.]+)\s+min=([0-9.]+)\s+max=([0-9.]+)\s+stddev=([0-9.]+)",
    re.MULTILINE,
)
TILE_RE = re.compile(r"^tile:\s+BM=(\d+)\s+BN=(\d+)\s+BK=(\d+)", re.MULTILINE)
VERIFY_RE = re.compile(r"^verify:\s+(\w+)", re.MULTILINE)
WORKGROUP_RE = re.compile(r"^workgroup:\s+(\d+)x(\d+)x(\d+)", re.MULTILINE)


def list_value(value):
    return value if isinstance(value, list) else [value]


def candidate_name(candidate):
    return (
        f"bm{candidate['block_m']}_bn{candidate['block_n']}_bk{candidate['block_k']}"
        f"_wm{candidate['warp_m']}_wn{candidate['warp_n']}"
        f"_wni{candidate['warp_n_iter']}_tm{candidate['thread_m']}"
        f"_tn{candidate['thread_n']}_thr{candidate['local_size_x']}"
    )


def expand_candidates(config):
    if "candidates" in config:
        for candidate in config["candidates"]:
            candidate = dict(candidate)
            candidate.setdefault("name", candidate_name(candidate))
            yield candidate
        return

    search = config["search_space"]
    keys = [
        "block_m",
        "block_n",
        "block_k",
        "warp_m",
        "warp_n",
        "warp_n_iter",
        "thread_m",
        "thread_n",
        "local_size_x",
        "stages",
        "dispatch_order",
    ]
    values = [list_value(search[key]) for key in keys]
    for combination in itertools.product(*values):
        candidate = dict(zip(keys, combination))
        candidate["name"] = candidate_name(candidate)
        yield candidate


def validate_candidate(candidate, limits):
    bm = int(candidate["block_m"])
    bn = int(candidate["block_n"])
    bk = int(candidate["block_k"])
    wm = int(candidate["warp_m"])
    wn = int(candidate["warp_n"])
    wniter = int(candidate["warp_n_iter"])
    tm = int(candidate["thread_m"])
    tn = int(candidate["thread_n"])
    threads = int(candidate["local_size_x"])
    stages = int(candidate["stages"])

    if stages != 2:
        return "only stages=2 is supported by the current double-buffer kernel"
    if bk % 4 != 0:
        return "BK must be divisible by vec4 width"
    if bn % 4 != 0:
        return "BN must be divisible by vec4 width"
    if threads % 2 != 0:
        return "local_size_x must be divisible by 2 for double-buffer load split"
    if threads % 32 != 0:
        return "local_size_x must be divisible by subgroup size 32"
    if threads > int(limits["max_workgroup_invocations"]):
        return "local_size_x exceeds max_workgroup_invocations"
    if bm % wm != 0:
        return "BM must be divisible by WM"
    if bn % wn != 0:
        return "BN must be divisible by WN"
    if (threads // 32) != (bm // wm) * (bn // wn):
        return "subgroup count must equal (BM/WM) * (BN/WN)"
    if wniter == 0:
        return "WNITER must be greater than zero"
    if (wn // wniter) == 0 or (wn // wniter) % tn != 0:
        return "WSUBN must be divisible by TN"

    numerator = wm * wn
    denominator = 32 * tm * tn * wniter
    if denominator == 0 or numerator % denominator != 0:
        return "WMITER must be an integer"
    wmiter = numerator // denominator
    if wmiter == 0:
        return "WMITER must be greater than zero"
    if wm % wmiter != 0:
        return "WM must be divisible by WMITER"

    wsubm = wm // wmiter
    wsubn = wn // wniter
    if (wsubm // tm) * (wsubn // tn) != 32:
        return "thread tile layout must map exactly to one subgroup"

    half_threads = threads // 2
    if ((half_threads * 4) % bk) != 0:
        return "A row stride must be integral"
    if (half_threads * 4) // bk == 0:
        return "A row stride must be greater than zero"
    if half_threads < (bn // 4) or half_threads % (bn // 4) != 0:
        return "B row stride must be integral and greater than zero"

    shared_bytes = stages * (bm * bk + bk * bn) * 4
    if shared_bytes > int(limits["max_shared_bytes"]):
        return "shared memory usage exceeds limit"

    thread_results = wmiter * tm * wniter * tn
    if thread_results > int(limits["max_thread_results"]):
        return "thread result register tile exceeds limit"

    return ""


def build_command(exe, kernel_id, candidate, benchmark):
    command = [
        str(exe),
        "--kernel",
        str(kernel_id),
        "--m",
        str(benchmark["m"]),
        "--n",
        str(benchmark["n"]),
        "--k",
        str(benchmark["k"]),
        "--alpha",
        str(benchmark["alpha"]),
        "--beta",
        str(benchmark["beta"]),
        "--warmup",
        str(benchmark["warmup"]),
        "--iters",
        str(benchmark["iters"]),
        "--spec-local-size-x",
        str(candidate["local_size_x"]),
        "--spec-bm",
        str(candidate["block_m"]),
        "--spec-bn",
        str(candidate["block_n"]),
        "--spec-bk",
        str(candidate["block_k"]),
        "--spec-wm",
        str(candidate["warp_m"]),
        "--spec-wn",
        str(candidate["warp_n"]),
        "--spec-wniter",
        str(candidate["warp_n_iter"]),
        "--spec-tm",
        str(candidate["thread_m"]),
        "--spec-tn",
        str(candidate["thread_n"]),
        "--spec-stages",
        str(candidate["stages"]),
        "--dispatch-order",
        str(candidate["dispatch_order"]),
    ]
    if not benchmark.get("verify", False):
        command.append("--no-verify")
    return command


def parse_output(output):
    gflops_match = GFLOPS_RE.search(output)
    kernel_ms_match = KERNEL_MS_RE.search(output)
    tile_match = TILE_RE.search(output)
    verify_match = VERIFY_RE.search(output)
    workgroup_match = WORKGROUP_RE.search(output)

    result = {
        "gflops": float(gflops_match.group(1)) if gflops_match else None,
        "verify": verify_match.group(1) if verify_match else None,
    }
    if kernel_ms_match:
        result["kernel_ms"] = {
            "avg": float(kernel_ms_match.group(1)),
            "min": float(kernel_ms_match.group(2)),
            "max": float(kernel_ms_match.group(3)),
            "stddev": float(kernel_ms_match.group(4)),
        }
    if tile_match:
        result["tile"] = {
            "block_m": int(tile_match.group(1)),
            "block_n": int(tile_match.group(2)),
            "block_k": int(tile_match.group(3)),
        }
    if workgroup_match:
        result["workgroup"] = {
            "x": int(workgroup_match.group(1)),
            "y": int(workgroup_match.group(2)),
            "z": int(workgroup_match.group(3)),
        }
    return result


def run_candidate(exe, kernel_id, candidate, benchmark):
    command = build_command(exe, kernel_id, candidate, benchmark)
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    parsed = parse_output(completed.stdout)
    verify_requested = benchmark.get("verify", False)
    ok_for_best = (
        completed.returncode == 0
        and parsed["gflops"] is not None
        and (not verify_requested or parsed["verify"] == "PASS")
    )
    return {
        "candidate": candidate["name"],
        "config": candidate,
        "command": command,
        "returncode": completed.returncode,
        "status": "OK" if ok_for_best else "FAILED",
        "ok_for_best": ok_for_best,
        "output": completed.stdout,
        **parsed,
    }


def select_candidates(valid_candidates, args):
    candidates = list(valid_candidates)
    if args.limit_random is not None:
        rng = random.Random(args.seed)
        rng.shuffle(candidates)
        candidates = candidates[: args.limit_random]
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="Autotune GEMM_VULKAN 11_autotuned specialization constants."
    )
    parser.add_argument("--exe", default="build/gemm_vulkan")
    parser.add_argument("--config", default="tools/autotune_double_buffering.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--limit-random", type=int)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    exe = Path(args.exe)
    kernel_id = int(config.get("autotuned_kernel_id", 13))
    benchmark = config["benchmark"]
    limits = config["limits"]

    skipped = []
    valid = []
    for candidate in expand_candidates(config):
        reason = validate_candidate(candidate, limits)
        if reason:
            skipped.append(
                {
                    "candidate": candidate["name"],
                    "config": candidate,
                    "status": "SKIPPED_INVALID",
                    "skip_reason": reason,
                    "ok_for_best": False,
                }
            )
        else:
            valid.append(candidate)

    selected = select_candidates(valid, args)

    if args.dry_run:
        reason_counts = Counter(item["skip_reason"] for item in skipped)
        print(f"total_candidates: {len(valid) + len(skipped)}")
        print(f"valid_candidates: {len(valid)}")
        print(f"selected_candidates: {len(selected)}")
        print(f"skipped_candidates: {len(skipped)}")
        for reason, count in reason_counts.most_common():
            print(f"skip_reason: {count} {reason}")
        print("selected:")
        for candidate in selected:
            print(f"  {candidate['name']}")
        return 0

    results = list(skipped)
    for index, candidate in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {candidate['name']}")
        result = run_candidate(exe, kernel_id, candidate, benchmark)
        results.append(result)
        print(
            f"  status={result['status']} verify={result.get('verify')} "
            f"gflops={result.get('gflops')}"
        )

    ranked = sorted(
        (result for result in results if result.get("ok_for_best")),
        key=lambda result: result["gflops"],
        reverse=True,
    )
    best = ranked[0] if ranked else None

    raw_path = Path(config["outputs"]["raw"])
    best_path = Path(config["outputs"]["best"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(results, indent=2))
    best_path.write_text(
        json.dumps(
            {
                "benchmark": benchmark,
                "best": best,
                "top10": ranked[:10],
                "selected_count": len(selected),
                "valid_count": len(valid),
                "skipped_count": len(skipped),
            },
            indent=2,
        )
    )

    if best is None:
        print("best: none")
        return 1

    print("best:")
    print(f"  candidate: {best['candidate']}")
    print(f"  gflops: {best['gflops']}")
    print(f"  kernel_ms_avg: {best.get('kernel_ms', {}).get('avg')}")
    print("top10:")
    for result in ranked[:10]:
        print(f"  {result['gflops']:.3f} GFLOPS  {result['candidate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
