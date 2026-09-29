#!/usr/bin/env python3
"""
Sample N records per benchmark with a fixed random seed and write fixretina-format
parquet files for RL held-out evaluation.

Usage:
    python sample_heldout_eval.py --output_dir /path/to/output [--n 50] [--seed 42]
    python sample_heldout_eval.py --output_dir ./heldout_50 --n 50 --seed 42

Output:
    {output_dir}/{benchmark_name}_heldout_50.parquet  (one file per benchmark)
    Optionally: {output_dir}/all_heldout_50.parquet   (concatenated)
"""
import argparse
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

# Add current directory so we can import datasets
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import (
    HRBenchDataset,
    VStarDataset,
    MMERealWorldLiteDataset,
    VisualProbeDataset,
    GeneralVQADataset,
)
from sample_to_fixretina_row import sample_to_fixretina_row

# Default base paths (vlmpaper root = 2 levels up from this file: eval_w_tool -> eval -> vlmpaper)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_VLMPAPER_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
DEFAULT_BENCH_BASE = os.path.join(_VLMPAPER_ROOT, "data", "benchmarks")
DEFAULT_VISUALPROBE_BASE = os.path.join(_VLMPAPER_ROOT, "data", "rl", "VisualProbe")

BENCHMARK_CONFIG = [
    ("vstar_bench", "vstar_bench", os.path.join(DEFAULT_BENCH_BASE, "vstar_bench"), None),
    ("hrbench4k", "HRBench4K", os.path.join(DEFAULT_BENCH_BASE, "hrbench", "hr_bench_4k.tsv"), os.path.join(DEFAULT_BENCH_BASE, "hrbench", "images_4k")),
    ("hrbench8k", "HRBench8K", os.path.join(DEFAULT_BENCH_BASE, "hrbench", "hr_bench_8k.tsv"), os.path.join(DEFAULT_BENCH_BASE, "hrbench", "images_8k")),
    ("mmerealworldlite", "mme_realworld_lite", os.path.join(DEFAULT_BENCH_BASE, "mmerealworldlite"), None),
    ("realworldqa", "realworldqa", os.path.join(DEFAULT_BENCH_BASE, "realworldqa"), None),
    ("visualprobe_easy", "visualprobe", os.path.join(DEFAULT_VISUALPROBE_BASE, "VisualProbe_Easy"), None),
    ("visualprobe_medium", "visualprobe", os.path.join(DEFAULT_VISUALPROBE_BASE, "VisualProbe_Medium"), None),
    ("visualprobe_hard", "visualprobe", os.path.join(DEFAULT_VISUALPROBE_BASE, "VisualProbe_Hard"), None),
]


def load_dataset(benchmark_name: str, dataset_name: str, dataset_path: str, image_dir: str | None):
    """Instantiate and load data for one benchmark."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if dataset_name.startswith("HRBench"):
        if not image_dir:
            raise ValueError(f"HRBench requires image_dir for {benchmark_name}")
        os.makedirs(image_dir, exist_ok=True)
        ds = HRBenchDataset(dataset_path, image_dir=image_dir)
    elif dataset_name == "vstar_bench":
        ds = VStarDataset(dataset_path)
    elif dataset_name == "mme_realworld_lite":
        ds = MMERealWorldLiteDataset(dataset_path)
    elif dataset_name == "visualprobe":
        ds = VisualProbeDataset(dataset_path)
    else:
        ds = GeneralVQADataset(dataset_path)
    return ds.load_data()


def main():
    parser = argparse.ArgumentParser(description="Sample held-out eval records per benchmark")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for parquet outputs")
    parser.add_argument("--n", type=int, default=50, help="Number of records to sample per benchmark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--concat", action="store_true", help="Also write one concatenated parquet (all_heldout_<n>.parquet)")
    parser.add_argument("--bench_base", type=str, default=None, help="Override base path for benchmarks (vstar, hrbench, mme, realworldqa)")
    parser.add_argument("--vp_base", type=str, default=None, help="Override base path for VisualProbe dirs")
    args = parser.parse_args()

    rng = __import__("random").Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    configs = list(BENCHMARK_CONFIG)
    if args.bench_base:
        base = args.bench_base.rstrip("/")
        configs = [
            (bn, dname, dpath.replace(DEFAULT_BENCH_BASE, base), imgdir.replace(DEFAULT_BENCH_BASE, base) if imgdir else None)
            for bn, dname, dpath, imgdir in configs
        ]
    if args.vp_base:
        base = args.vp_base.rstrip("/")
        configs = [
            (bn, dname, dpath.replace(DEFAULT_VISUALPROBE_BASE, base), imgdir.replace(DEFAULT_VISUALPROBE_BASE, base) if imgdir else None)
            for bn, dname, dpath, imgdir in configs
        ]

    all_rows = []
    for benchmark_name, dataset_name, dataset_path, image_dir in configs:
        try:
            data = load_dataset(benchmark_name, dataset_name, dataset_path, image_dir)
        except Exception as e:
            print(f"[{benchmark_name}] Skip: {e}")
            continue
        n_total = len(data)
        k = min(args.n, n_total)
        indices = rng.sample(range(n_total), k)
        samples = [data[i] for i in indices]
        rows = []
        for s in samples:
            try:
                row = sample_to_fixretina_row(s, data_source=benchmark_name)
                rows.append(row)
            except Exception as e:
                print(f"[{benchmark_name}] Skip sample {s.get('index')}: {e}")
        if not rows:
            print(f"[{benchmark_name}] No rows written")
            continue
        out_path = os.path.join(args.output_dir, f"{benchmark_name}_heldout_{k}.parquet")
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_path)
        print(f"[{benchmark_name}] Wrote {len(rows)} rows -> {out_path}")
        all_rows.extend(rows)

    if args.concat and all_rows:
        concat_path = os.path.join(args.output_dir, f"all_heldout_{args.n}.parquet")
        pq.write_table(pa.Table.from_pylist(all_rows), concat_path)
        print(f"Concatenated {len(all_rows)} rows -> {concat_path}")


if __name__ == "__main__":
    main()
