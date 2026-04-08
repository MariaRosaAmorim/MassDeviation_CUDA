"""
Usage:
    python benchmark/benchmark.py [--target cuda|cpu|both]

  --target  Which back-end to benchmark (default: both)

Dependencies: pandas  (pip install pandas)
"""

import argparse
import itertools
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths – all relative to this script's location so the script can be called
# from any working directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
CUDA_SOURCE = str(SCRIPT_DIR / "gpu_test.cu")
CPU_SOURCE = str(SCRIPT_DIR / "cpu_test.c")
BUILD_DIR = SCRIPT_DIR / "build"  # compiled binaries land here
RESULTS_DIR = SCRIPT_DIR / "results"  # simulation results land here
OUTPUT_CSV = SCRIPT_DIR / "benchmark_results.csv"

# ---------------------------------------------------------------------------
# Default parameter grid – edit these lists to define the sweep
# ---------------------------------------------------------------------------

NL_VALUES = [10, 100, 200]
NH_VALUES = [10, 100, 200]
NT_VALUES = [1_000, 10_000]

# Radius: auto-derived as nL // 4 (keeps the droplet proportional to domain).
# Set R_OVERRIDE = [25, 50, ...] to fix explicit values instead.
R_OVERRIDE = None

# ---------------------------------------------------------------------------
# Compiler settings
# ---------------------------------------------------------------------------
NVCC = "nvcc"
GXX = "g++"
OPT_FLAGS = ["-O3"]

# NVCC_EXTRA = ["-Xcompiler", "-fopenmp", "-lm"]
# GXX_EXTRA = ["-fopenmp", "-lm"]
NVCC_EXTRA = ["-Xcompiler", "-lm"]
GXX_EXTRA = ["-lm"]

# Timeout per execution run in seconds (None = no limit)
RUN_TIMEOUT = 3600


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def radius_for(nL: int, index: int) -> int:
    if R_OVERRIDE:
        return R_OVERRIDE[index % len(R_OVERRIDE)]
    return nL // 4


def compile_variant(
    source: str,
    binary: str,
    nL: int,
    nH: int,
    nt: int,
    R: int,
    tag: str,
    use_nvcc: bool,
) -> tuple[bool, str]:
    """Compile *source* → *binary* with the given macro values and tag."""
    defines = [
        f"-DnL={nL}",
        f"-DnH={nH}",
        f"-Dnt={nt}",
        f"-DR={R}",
        f'-DOUTPUT_TAG="{tag}"',  # printed in the simulation report
    ]
    if use_nvcc:
        cmd = [NVCC] + OPT_FLAGS + defines + NVCC_EXTRA + [source, "-o", binary]
    else:
        cmd = [GXX] + OPT_FLAGS + defines + GXX_EXTRA + [source, "-o", binary]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, ""
    except FileNotFoundError as exc:
        return False, f"Compiler not found: {exc}"
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out"


def run_variant(binary: str, output_dir: Path) -> tuple[float | None, str]:
    """
    Run *binary* with cwd=*output_dir* so all simulation output files land
    inside the tag-specific subfolder. Simulation stdout is forwarded directly
    to the console; only stderr is captured for error reporting.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        t_start = time.perf_counter()
        result = subprocess.run(
            [binary],
            stdout=None,  # inherit parent stdout → visible on console
            stderr=subprocess.PIPE,  # capture stderr for error reporting only
            timeout=RUN_TIMEOUT,
            cwd=str(output_dir),
        )
        elapsed = time.perf_counter() - t_start

        if result.returncode != 0:
            msg = f"exit {result.returncode}: {result.stderr.strip()[:300]}"
            return None, msg

        return elapsed, ""
    except subprocess.TimeoutExpired:
        return None, f"Timed out after {RUN_TIMEOUT}s"
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------
def run_benchmark(targets: list[str]) -> pd.DataFrame:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    combos = list(itertools.product(NL_VALUES, NH_VALUES, NT_VALUES))
    total = len(combos) * len(targets)
    done = 0

    print(f"\n{'='*62}")
    print(
        f"  Benchmark – {total} run(s) over {len(combos)} combo(s), targets: {targets}"
    )
    print(f"  Output root : {SCRIPT_DIR}")
    print(f"{'='*62}\n")

    for idx, (nL, nH, nt) in enumerate(combos):
        R = radius_for(nL, idx)

        for target in targets:
            done += 1
            use_nvcc = target == "cuda"
            source = CUDA_SOURCE if use_nvcc else CPU_SOURCE
            tag = f"nL{nL}_nH{nH}_nt{nt}_R{R}_{target}"
            binary = str(BUILD_DIR / f"sim_{tag}")
            output_dir = RESULTS_DIR / tag  # simulation files land here

            print(f"[{done}/{total}] {tag}")
            print(f"         Compiling ...", end="", flush=True)

            ok, err = compile_variant(source, binary, nL, nH, nt, R, tag, use_nvcc)
            if not ok:
                print(f"  FAILED\n         {err[:200]}")
                records.append(
                    {
                        "target": target,
                        "nL": nL,
                        "nH": nH,
                        "nt": nt,
                        "R": R,
                        "grid_size": nL * nH,
                        "compile_ok": False,
                        "execution_time": float("nan"),
                        "mlups": float("nan"),
                        "output_dir": "",
                        "error": err[:300],
                    }
                )
                continue

            print(f"  OK  →  running (output → {tag}/) ...", end="", flush=True)

            elapsed, err = run_variant(binary, output_dir)
            if elapsed is None:
                print(f"  FAILED\n         {err}")
            else:
                mlups = (nL * nH * nt) / elapsed / 1e6
                print(f"  {elapsed:.2f}s  ({mlups:.1f} MLUP/s)")

            records.append(
                {
                    "target": target,
                    "nL": nL,
                    "nH": nH,
                    "nt": nt,
                    "R": R,
                    "grid_size": nL * nH,
                    "compile_ok": True,
                    "execution_time": elapsed if elapsed is not None else float("nan"),
                    "mlups": mlups if mlups is not None else float("nan"),
                    "output_dir": str(output_dir) if elapsed is not None else "",
                    "error": err[:300] if err else "",
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LBM CUDA/CPU benchmark")
    parser.add_argument(
        "--target",
        choices=["cuda", "cpu", "both"],
        default="both",
        help="Back-end to benchmark (default: both)",
    )
    args = parser.parse_args()

    targets = ["cuda", "cpu"] if args.target == "both" else [args.target]

    df = run_benchmark(targets)

    print(f"\n{'='*62}")
    print("  Results")
    print(f"{'='*62}")
    print(df.to_string(index=False))

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}\n")

    # Quick summary of successful runs
    valid = df[df["execution_time"].notna() & (df["compile_ok"] == True)]
    if len(valid) > 1:
        fastest = valid.loc[valid["execution_time"].idxmin()]
        slowest = valid.loc[valid["execution_time"].idxmax()]
        print(
            f"Fastest: {fastest['tag'] if 'tag' in fastest else fastest[['target','nL','nH']].to_dict()}  →  {fastest['execution_time']:.2f}s"
        )
        print(
            f"Slowest: {slowest['tag'] if 'tag' in slowest else slowest[['target','nL','nH']].to_dict()}  →  {slowest['execution_time']:.2f}s"
        )


if __name__ == "__main__":
    main()
