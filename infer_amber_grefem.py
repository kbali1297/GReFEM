#!/usr/bin/env python
"""Run a trained AMBER checkpoint on the held-out GReFEM evaluation units and
write budget-matched refined meshes that plug into the existing eval pipeline.

For each (object, load_case):
  * build AMBER's initial coarse mesh from the ORIGINAL STEP geometry
  * run `inference_steps` iterative predict-and-remesh steps
  * calibrate AMBER's own last-step sizing-field scale (`last_step_damping`)
    so the tet count matches that unit's GReFEM budget within tol -- the same
    budget convention used for the heuristic and oracle candidates
  * write <object>/refined_mesh/{load_case}_amber_refined.msh

No oracle information is used at inference: the expert-mesh slot of AmberData
(which only feeds metrics, never the prediction) is filled with the initial
coarse mesh.

Usage:
    python infer_amber_grefem.py --ckpt <path.ckpt> --bounds amber_bounds.json \
        --shard 0 --nshards 40 --out_csv amber_infer_shards/a0.csv
"""
import argparse
import csv
import functools
import glob
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, "/data/1bali/GReFEM/baselines/AMBER")
sys.path.insert(0, "/data/1bali/GReFEM")

print = functools.partial(print, flush=True)

LCS = ["bending", "compression", "torsion",
       "bending_compression", "torsion_compression"]
SRC_DIR = {"orig": "test_meshes", "extra": "test_meshes_extra",
           "dense": "test_meshes_dense"}
FIELDS = ["object", "load_case", "src", "target_cells", "n_cells",
          "damping", "iters", "out_msh", "t_s", "status", "level"]
SWEEP_LEVELS = ["h100", "h125", "h160", "h200"]


def append_eval_queue(path, line):
    """Atomically append a finished unit to the eval work queue."""
    import fcntl
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line + "\n")


def build_units(units_csv, budget_method="grefem_max"):
    """Eval units + per-unit element budget from the existing results table."""
    import pandas as pd
    df = pd.read_csv(units_csv, dtype={"object": str})
    df = df[df["method"] == budget_method]
    units = []
    for _, r in df.iterrows():
        units.append(dict(object=r["object"], load_case=r["load_case"],
                          src=r.get("src", "extra"),
                          target_cells=float(r["n_cells"])))
    units.sort(key=lambda u: (u["object"], u["load_case"]))
    return units


def run_sweep(args, algo, prep, T, units, done, writer, fout, count_tets,
              geom_fn_from_file, get_bounding_box, mesh_from_geometry_fn,
              MeshWrapper, SourceData, GrefemFeatureProvider, AmberData):
    """Multi-level budget sweep: per unit, run steps 1..T-1 once, then
    calibrate the final sizing-field scale separately for each h-level's
    GReFEM budget. Writes <lc>_<suffix><level>_refined.msh per level."""
    for u in units:
        obj, lc, src = u["object"], u["load_case"], u["src"]
        obj_dir = os.path.join(SRC_DIR.get(src, src), obj)
        out_paths = {lv: os.path.join(
            obj_dir, "refined_mesh",
            f"{lc}_{args.suffix}{lv[1:]}_refined.msh") for lv in SWEEP_LEVELS}
        pending = []
        for lv in SWEEP_LEVELS:
            if (obj, lc, lv) in done:
                continue
            if os.path.exists(out_paths[lv]) and count_tets(out_paths[lv]) > 0:
                writer.writerow(dict(object=obj, load_case=lc, src=src,
                                     level=lv, out_msh=out_paths[lv],
                                     target_cells=int(u["targets"][lv]),
                                     n_cells=count_tets(out_paths[lv]),
                                     t_s=0.0, iters=0, damping="",
                                     status="ok"))
                fout.flush()
                continue
            pending.append(lv)
        if not pending:
            if args.eval_queue:
                append_eval_queue(args.eval_queue, f"{src},{obj},{lc}")
            continue

        t0 = time.time()
        try:
            steps = glob.glob(os.path.join(obj_dir, "*.step"))
            if not steps:
                raise RuntimeError("no STEP file")
            gf = geom_fn_from_file(steps[0])
            bb = get_bounding_box(geometry_fn=gf)
            dim = len(bb) // 2
            ext = np.asarray(bb[dim:], float) - np.asarray(bb[:dim], float)
            vol = float(np.prod(ext)) / 3000.0
            init, init_err = None, None
            for attempt in range(4):
                try:
                    init = mesh_from_geometry_fn(
                        geometry_fn=gf,
                        max_initial_element_volume=vol / (8 ** attempt), dim=dim)
                    break
                except (Exception, SystemExit) as e:
                    init_err = e
            if init is None:
                raise RuntimeError(f"initial mesh failed at all sizes: {init_err}")
            init.geom_fn = gf
            init_w = MeshWrapper(init)

            feats = [1.0 if lc == l else 0.0 for l in LCS]
            feats.append(float(np.log10(max(np.linalg.norm(ext), 1e-12))))
            ftxt = out_paths[SWEEP_LEVELS[0]].replace(".msh", "_features.txt")
            os.makedirs(os.path.dirname(ftxt), exist_ok=True)
            with open(ftxt, "w") as f:
                f.write("\n".join(str(v) for v in feats) + "\n")
            names = [k for k, v in
                     prep.task_config.grefem_features.vertex.items() if v]
            fp = GrefemFeatureProvider(
                features_file=ftxt,
                observation_features={"vertex": names, "element": names})
            src_data = SourceData(expert_mesh=init_w, initial_mesh=init_w,
                                  feature_provider=fp)
            data = AmberData(mesh=init_w, source_data=src_data,
                             **prep._data_init_kwargs)

            # shared steps 1..T-1 (budget-independent)
            algo.sizing_field_damping.last_step_damping = 1.0
            for _ in range(max(T - 1, 0)):
                out = algo._inference_step(data)
                if not out.refinement_okay:
                    break
                data = AmberData.from_reference(reference=data,
                                                new_mesh=out.output_mesh)
        except (Exception, SystemExit) as e:
            for lv in pending:
                writer.writerow(dict(
                    object=obj, load_case=lc, src=src, level=lv,
                    out_msh=out_paths[lv], t_s=round(time.time() - t0, 1),
                    status=f"error: {type(e).__name__}: {str(e)[:90]}"))
            fout.flush()
            print(f"[fail] {obj}/{lc}: {e}")
            traceback.print_exc()
            if args.eval_queue:
                append_eval_queue(args.eval_queue, f"{src},{obj},{lc}")
            continue

        import meshio
        damping, prev_target = 1.0, None
        for lv in pending:
            t1 = time.time()
            target = u["targets"][lv]
            row = dict(object=obj, load_case=lc, src=src, level=lv,
                       target_cells=int(target), out_msh=out_paths[lv])
            try:
                if prev_target is not None:
                    # warm start: n ~ damping^-3 roughly; same sqrt rule
                    damping = damping * (prev_target / target) ** 0.5
                best, n = None, None
                for it in range(args.max_iter):
                    algo.sizing_field_damping.last_step_damping = float(damping)
                    out = algo._inference_step(data)
                    mesh = out.output_mesh
                    n = int(mesh.nelements)
                    gap = abs(n - target) / target
                    if best is None or gap < best[0]:
                        best = (gap, n, damping, mesh)
                    if gap <= args.tol:
                        break
                    damping = damping * (n / target) ** 0.5
                gap, n, damping, mesh = best
                prev_target = target
                skm = mesh.mesh if hasattr(mesh, "mesh") else mesh
                _tets = skm.t.T.copy()
                _ones = np.ones(len(_tets), dtype=np.int32)
                m = meshio.Mesh(points=skm.p.T.copy(),
                                cells=[("tetra", _tets)],
                                cell_data={"gmsh:physical": [_ones],
                                           "gmsh:geometrical": [_ones.copy()]},
                                field_data={"volume": np.array([1, 3])})
                meshio.write(out_paths[lv], m, file_format="gmsh22",
                             binary=False)
                n = count_tets(out_paths[lv])
                if n <= 0:
                    status = "fail_write"
                elif gap > 0.5:
                    status = f"budget_miss_{gap:.0%}"
                else:
                    status = "ok"
                row.update(n_cells=n, damping=f"{damping:.4g}", iters=it + 1,
                           t_s=round(time.time() - t1, 1), status=status)
                print(f"{obj}/{lc}/{lv}: {n} tets (target {int(target)}, "
                      f"gap {gap:.2%}, damping {damping:.3g}, {row['t_s']}s)")
            except (Exception, SystemExit) as e:
                row.update(t_s=round(time.time() - t1, 1),
                           status=f"error: {type(e).__name__}: {str(e)[:90]}")
                print(f"[fail] {obj}/{lc}/{lv}: {e}")
            writer.writerow(row)
            fout.flush()
        if args.eval_queue:
            append_eval_queue(args.eval_queue, f"{src},{obj},{lc}")
        print(f"[unit done] {obj}/{lc} in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bounds", default="amber_bounds.json",
                    help="JSON with min/max_sizing_field + max_mesh_elements "
                         "computed once from the full training set")
    ap.add_argument("--units_csv", default="combined_200obj_table.csv")
    ap.add_argument("--budget_method", default="grefem_max")
    ap.add_argument("--stub_train_points", type=int, default=8)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--max_iter", type=int, default=4)
    ap.add_argument("--suffix", default="amber")
    ap.add_argument("--task", default="grefem",
                    help="hydra task config: grefem (adapted) or grefem_default")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--queue_file", default=None,
                    help="shared work queue: lines 'object,load_case'; workers "
                         "pop atomically (flock). Overrides --shard/--nshards.")
    ap.add_argument("--sweep_csv", default=None,
                    help="multi-level sweep mode: CSV with columns object,"
                         "load_case,src,h100,h125,h160,h200 (per-level target "
                         "cells). Shares inference steps 1..T-1 across levels; "
                         "only the calibrated final step is redone per level. "
                         "Outputs <lc>_<suffix><level-digits>_refined.msh.")
    ap.add_argument("--eval_queue", default=None,
                    help="file to append 'src,object,load_case' after a unit "
                         "finishes all levels (feeds the solve+QoI workers)")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    from src.algorithm import create_algorithm
    from src.algorithm.dataloader.source_data import SourceData
    from src.tasks.grefem_dataset_preparator import GrefemDatasetPreparator
    from src.tasks.domains.gmsh_util import geom_fn_from_file, get_bounding_box
    from src.tasks.domains.mesh_wrapper import MeshWrapper
    from src.tasks.expert_geometry_dataset_preparator import mesh_from_geometry_fn
    from src.tasks.features.grefem_feature_provider import GrefemFeatureProvider

    def count_tets(msh_path):
        """Tet count of a mesh file; 0 if unreadable/truncated (failed gmsh run).

        Inlined from compute_local_error_tables.count_tets to avoid importing
        that module's fenics/mpi4py dependency chain (absent in AMBER_neurips).
        """
        import meshio as _meshio
        try:
            m = _meshio.read(msh_path)
        except (Exception, SystemExit):
            return 0
        return sum(len(b.data) for b in m.cells if b.type == "tetra")

    with initialize_config_dir(config_dir="/data/1bali/GReFEM/baselines/AMBER/config",
                              version_base=None):
        cfg = compose(config_name="training_config",
                      overrides=["+_runs/neurips_submission/amber=_amber",
                                 f"task={args.task}", "exp_name=infer",
                                 "logger.wandb.enabled=False"])
    # tiny stub training set: only needed for network input shapes; the
    # normalizer statistics and weights come from the checkpoint, and the
    # sizing-field clip bounds are injected from --bounds below.
    with open_dict(cfg):
        cfg.task.num_data_points.train = args.stub_train_points
        cfg.task.num_data_points.val = 1
        cfg.task.num_data_points.test = 1
    prep = GrefemDatasetPreparator(algorithm_config=cfg.algorithm,
                                   task_config=cfg.task)
    datasets = prep()
    algo = create_algorithm(algorithm_config=cfg.algorithm,
                            train_dataset=datasets["train"], loading=True,
                            checkpoint_path=args.ckpt)
    algo.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    algo.to(dev)

    bounds = json.load(open(args.bounds))
    algo.gmsh_kwargs["min_sizing_field"] = bounds["min_sizing_field"]
    algo.gmsh_kwargs["max_sizing_field"] = bounds["max_sizing_field"]
    algo.max_mesh_elements = bounds.get("max_mesh_elements", 1e7)
    OmegaConf.set_struct(algo.sizing_field_damping, False)
    T = algo.inference_steps
    print(f"loaded {args.ckpt}: inference_steps={T}, "
          f"clip=({algo.gmsh_kwargs['min_sizing_field']:.4g}, "
          f"{algo.gmsh_kwargs['max_sizing_field']:.4g}), dev={dev}")

    if args.sweep_csv:
        all_units = []
        with open(args.sweep_csv, newline="") as f:
            for r in csv.DictReader(f):
                all_units.append(dict(
                    object=r["object"], load_case=r["load_case"],
                    src=r["src"],
                    targets={lv: float(r[lv]) for lv in SWEEP_LEVELS}))
        all_units.sort(key=lambda u: (u["object"], u["load_case"]))
        print(f"sweep mode: {len(all_units)} units x {len(SWEEP_LEVELS)} levels")
    else:
        all_units = build_units(args.units_csv, args.budget_method)
    if args.queue_file:
        import fcntl

        unit_by_key = {(u["object"], u["load_case"]): u for u in all_units}

        def pop_unit():
            with open(args.queue_file, "r+") as qf:
                fcntl.flock(qf, fcntl.LOCK_EX)
                lines = [l.strip() for l in qf.readlines() if l.strip()]
                if not lines:
                    return None
                head, rest = lines[0], lines[1:]
                qf.seek(0)
                qf.truncate()
                qf.write("\n".join(rest) + ("\n" if rest else ""))
            obj, lc = head.split(",")
            return unit_by_key.get((obj, lc))

        def unit_iter():
            while True:
                u = pop_unit()
                if u is None:
                    return
                yield u

        units = unit_iter()
        print(f"queue mode: pulling from {args.queue_file}")
    else:
        units = [u for i, u in enumerate(all_units)
                 if i % args.nshards == args.shard]
        print(f"shard {args.shard}/{args.nshards}: {len(units)} units")

    done = set()
    if os.path.exists(args.out_csv):
        with open(args.out_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add((row["object"], row["load_case"],
                              row.get("level", "")))
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    write_header = not os.path.exists(args.out_csv)
    fout = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=FIELDS)
    if write_header:
        writer.writeheader()

    from src.algorithm.dataloader.amber_data import AmberData

    if args.sweep_csv:
        run_sweep(args, algo, prep, T, units, done, writer, fout, count_tets,
                  geom_fn_from_file, get_bounding_box, mesh_from_geometry_fn,
                  MeshWrapper, SourceData, GrefemFeatureProvider, AmberData)
        fout.close()
        print(f"Done. {args.out_csv}")
        return

    for ui, u in enumerate(units):
        obj, lc = u["object"], u["load_case"]
        if (obj, lc, "") in done:
            continue
        t0 = time.time()
        obj_dir = os.path.join(SRC_DIR.get(u["src"], u["src"]), obj)
        out_msh = os.path.join(obj_dir, "refined_mesh",
                               f"{lc}_{args.suffix}_refined.msh")
        row = dict(object=obj, load_case=lc, src=u["src"],
                   target_cells=int(u["target_cells"]), out_msh=out_msh)
        if os.path.exists(out_msh) and count_tets(out_msh) > 0:
            row.update(n_cells=count_tets(out_msh), status="ok",
                       t_s=0.0, iters=0, damping="")
            writer.writerow(row); fout.flush(); continue
        try:
            steps = glob.glob(os.path.join(obj_dir, "*.step"))
            if not steps:
                raise RuntimeError("no STEP file")
            gf = geom_fn_from_file(steps[0])
            bb = get_bounding_box(geometry_fn=gf)
            dim = len(bb) // 2
            ext = np.asarray(bb[dim:], float) - np.asarray(bb[:dim], float)
            vol = float(np.prod(ext)) / 3000.0
            # some cohort geometries fail with "Invalid boundary mesh
            # (overlapping facets)" at coarse resolution; retry finer.
            init = None
            init_err = None
            for attempt in range(4):
                try:
                    init = mesh_from_geometry_fn(
                        geometry_fn=gf,
                        max_initial_element_volume=vol / (8 ** attempt), dim=dim)
                    break
                except (Exception, SystemExit) as e:
                    init_err = e
            if init is None:
                raise RuntimeError(f"initial mesh failed at all sizes: {init_err}")
            init.geom_fn = gf
            init_w = MeshWrapper(init)

            feats = [1.0 if lc == l else 0.0 for l in LCS]
            feats.append(float(np.log10(max(np.linalg.norm(ext), 1e-12))))
            ftxt = out_msh.replace(".msh", "_features.txt")
            os.makedirs(os.path.dirname(ftxt), exist_ok=True)
            with open(ftxt, "w") as f:
                f.write("\n".join(str(v) for v in feats) + "\n")
            names = [k for k, v in
                     prep.task_config.grefem_features.vertex.items() if v]
            fp = GrefemFeatureProvider(
                features_file=ftxt,
                observation_features={"vertex": names, "element": names})

            # expert slot = the coarse mesh itself: it only feeds metrics,
            # never the prediction, so no oracle information leaks in.
            src_data = SourceData(expert_mesh=init_w, initial_mesh=init_w,
                                  feature_provider=fp)
            data = AmberData(mesh=init_w, source_data=src_data,
                             **prep._data_init_kwargs)

            # steps 1..T-1 unchanged
            for _ in range(max(T - 1, 0)):
                out = algo._inference_step(data)
                if not out.refinement_okay:
                    break
                data = AmberData.from_reference(reference=data,
                                                new_mesh=out.output_mesh)

            # final step: calibrate the global sizing-field scale so the tet
            # count matches this unit's GReFEM budget (n_cells ~ h^-3; use the
            # same damped exponent as generate_budget_matched_mesh)
            target = u["target_cells"]
            damping, best = 1.0, None
            n = None
            for it in range(args.max_iter):
                algo.sizing_field_damping.last_step_damping = float(damping)
                out = algo._inference_step(data)
                mesh = out.output_mesh
                n = int(mesh.nelements)
                gap = abs(n - target) / target
                if best is None or gap < best[0]:
                    best = (gap, n, damping, mesh)
                if gap <= args.tol:
                    break
                damping = damping * (n / target) ** 0.5
            gap, n, damping, mesh = best
            import meshio
            skm = mesh.mesh if hasattr(mesh, "mesh") else mesh
            _tets = skm.t.T.copy()
            _ones = np.ones(len(_tets), dtype=np.int32)
            m = meshio.Mesh(points=skm.p.T.copy(),
                            cells=[("tetra", _tets)],
                            cell_data={"gmsh:physical": [_ones],
                                       "gmsh:geometrical": [_ones.copy()]},
                            field_data={"volume": np.array([1, 3])})
            meshio.write(out_msh, m, file_format="gmsh22", binary=False)
            n = count_tets(out_msh)
            if n <= 0:
                status = "fail_write"
            elif gap > 0.5:
                # refinement collapsed (e.g. remeshing failures) - the mesh is
                # nowhere near the matched budget; must not enter the table
                status = f"budget_miss_{gap:.0%}"
            else:
                status = "ok"
            row.update(n_cells=n, damping=f"{damping:.4g}", iters=it + 1,
                       t_s=round(time.time() - t0, 1), status=status)
            ntot = len(units) if isinstance(units, list) else "?"
            print(f"[{ui+1}/{ntot}] {obj}/{lc}: {n} tets "
                  f"(target {int(target)}, gap {gap:.2%}, damping {damping:.3g}, "
                  f"{row['t_s']}s)")
        except (Exception, SystemExit) as e:
            row.update(status=f"error: {type(e).__name__}: {str(e)[:90]}",
                       t_s=round(time.time() - t0, 1))
            print(f"[fail] {obj}/{lc}: {e}")
            traceback.print_exc()
        writer.writerow(row)
        fout.flush()

    fout.close()
    print(f"Done. {args.out_csv}")


if __name__ == "__main__":
    main()
