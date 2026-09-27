#!/usr/bin/env python
"""Populate AMBER's dataset-preparation cache in parallel.

AMBER prepares its dataset eagerly at every launch: per data point it imports
the STEP geometry through OCC, generates the initial coarse mesh with gmsh and
reads the (large) expert mesh. That is single-threaded and costs ~13 min for
our 975+ data points, and it is repeated on every restart.

GrefemDatasetPreparator now caches the resulting vertex/element arrays to
<dataset>/_prep_cache/<split>_<idx>.npz and reloads them instead (geom_fn is a
cheap closure and is rebuilt on load, so meshing during training still works).
This script fills that cache with a process pool, so the one-off cost is
divided across cores and every later run starts in seconds.

Usage:
    python build_amber_cache.py --task grefem --splits train val test --workers 48
"""
import argparse
import functools
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, "/data/1bali/GReFEM/baselines/AMBER")

print = functools.partial(print, flush=True)

AMBER_ROOT = "/data/1bali/GReFEM/baselines/AMBER"


def _one(args):
    idx, split, task = args
    os.chdir(AMBER_ROOT)
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from src.tasks.grefem_dataset_preparator import GrefemDatasetPreparator

    with initialize_config_dir(config_dir=os.path.join(AMBER_ROOT, "config"),
                              version_base=None):
        cfg = compose(config_name="training_config",
                      overrides=["+_runs/neurips_submission/amber=_amber",
                                 f"task={task}", "exp_name=cache",
                                 "logger.wandb.enabled=False"])
    with open_dict(cfg):
        cfg.task.num_data_points.train = 1
        cfg.task.num_data_points.val = 1
        cfg.task.num_data_points.test = 1
    prep = GrefemDatasetPreparator(algorithm_config=cfg.algorithm,
                                   task_config=cfg.task)
    if os.path.exists(prep.cache_path(idx, split)):
        return (idx, split, "cached")
    try:
        prep._prepare_source_and_mesh(data_idx=idx, dataset_mode=split)
        return (idx, split, "ok")
    except (Exception, SystemExit) as e:
        return (idx, split, f"fail: {type(e).__name__}: {str(e)[:70]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="grefem")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(AMBER_ROOT, "config", "task",
                                      f"{args.task}.yaml"))
    counts = {s: int(cfg.num_data_points[s]) for s in args.splits}
    print(f"task {args.task}: {counts}")

    tasks = [(i, s, args.task) for s in args.splits
             for i in range(counts[s])]
    n_ok = n_cached = n_fail = 0
    with Pool(args.workers) as pool:
        for k, (idx, split, status) in enumerate(
                pool.imap_unordered(_one, tasks, chunksize=1)):
            if status == "ok":
                n_ok += 1
            elif status == "cached":
                n_cached += 1
            else:
                n_fail += 1
                print(f"[{split} {idx+1}] {status}")
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(tasks)} | new {n_ok} cached {n_cached} "
                      f"fail {n_fail}")
    print(f"\ndone: new {n_ok}, already cached {n_cached}, failed {n_fail}")
    cache_dir = os.path.join(AMBER_ROOT, "data", args.task, "_prep_cache")
    if os.path.isdir(cache_dir):
        n = len(os.listdir(cache_dir))
        size = sum(os.path.getsize(os.path.join(cache_dir, f))
                   for f in os.listdir(cache_dir)) / 1e9
        print(f"cache: {n} files, {size:.1f} GB in {cache_dir}")


if __name__ == "__main__":
    main()
