"""Prepare brand-new ABC objects for infer_meshpoints.py.

Samples N unseen objects from ABC_CAD_Dataset_small2_augmented (excluding
ortho-view training objects, the val split and the existing test set), then per object:
  1. copy .obj/.step into <dest>/<obj>/
  2. render renders_pyvista_mesh_initial (110 ortho views)
  3. render renders_pyvista_mesh_<lc> for all 5 load cases (110 ortho views each)
  4. run the ortho-view selector NN -> pred_ortho_views2.log (top 10)

Usage:
  python prep_new_objects.py --dest test_meshes_extra --n 10 --seed 0 --workers 20
"""
import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ABC_DIR = '/data/1bali/GReFEM/ABC_CAD_Dataset_small2_augmented'
TRAIN_LOG = '/data/1bali/GReFEM/train_orthoviews.log'
VAL_LOG = '/data/1bali/GReFEM/val_7.04.2026.txt'
TEST_DIR = '/data/1bali/GReFEM/test_meshes'
LOAD_CASES = ['compression', 'bending', 'torsion', 'bending_compression', 'torsion_compression']
ELEVATIONS = [-90, -72, -54, -36, -18, 0, 18, 36, 54, 72, 90]


def excluded_ids():
    excl = set()
    for f in (TRAIN_LOG, VAL_LOG):
        with open(f) as fr:
            excl.update(os.path.basename(l.strip()) for l in fr if l.strip())
    excl.update(os.listdir(TEST_DIR))
    return excl


def find_files(folder):
    obj = step = None
    for f in os.listdir(folder):
        if f.endswith('.obj'):
            obj = os.path.join(folder, f)
        elif f.endswith('.step'):
            step = os.path.join(folder, f)
    return obj, step


def is_valid(obj_path):
    try:
        import trimesh
        m = trimesh.load(obj_path, force='mesh')
        if len(m.faces) < 200:
            return False
        ext = m.bounds[1] - m.bounds[0]
        if np.min(ext) <= 0 or np.max(ext) / max(np.min(ext), 1e-12) > 200:
            return False
        return True
    except Exception:
        return False


def sample_objects(n, seed, extra_excl=()):
    excl = excluded_ids()
    excl.update(extra_excl)
    cands = sorted(set(os.listdir(ABC_DIR)) - excl)
    rng = np.random.default_rng(seed)
    rng.shuffle(cands)
    chosen = []
    for cid in cands:
        folder = os.path.join(ABC_DIR, cid)
        if not os.path.isdir(folder):
            continue
        obj, step = find_files(folder)
        if obj is None or step is None:
            continue
        if not is_valid(obj):
            print(f'[skip-invalid] {cid}', flush=True)
            continue
        chosen.append(cid)
        print(f'[sampled] {len(chosen)}/{n}: {cid}', flush=True)
        if len(chosen) == n:
            break
    return chosen


def render_task(args):
    cid, dest, lc = args
    from generate_renders import render_mesh_views, render_mesh_views_with_load
    mesh_obj = f'{dest}/{cid}/renders_pyvista/{cid}.obj'
    try:
        if lc is None:
            out = f'{dest}/{cid}/renders_pyvista_mesh_initial'
            if os.path.isdir(out) and len([f for f in os.listdir(out) if f.endswith('.png')]) >= 110:
                return f'[skip] {cid}/initial'
            render_mesh_views(mesh_obj, output_dir=out, n_azimuth=12,
                              n_elevation=ELEVATIONS, orthographic=True, add_axes=False)
            return f'[done] {cid}/initial'
        out = f'{dest}/{cid}/renders_pyvista_mesh_{lc}'
        if os.path.isdir(out) and len([f for f in os.listdir(out) if f.endswith('.png')]) >= 110:
            return f'[skip] {cid}/{lc}'
        render_mesh_views_with_load(mesh_obj, output_dir_prefix=out, n_azimuth=12,
                                    n_elevation=ELEVATIONS, orthographic=True,
                                    add_axes=False, loading_type=lc, verbose=False)
        return f'[done] {cid}/{lc}'
    except Exception as e:
        return f'[fail] {cid}/{lc}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dest', default='/data/1bali/GReFEM/test_meshes_extra')
    ap.add_argument('--n', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--ids', nargs='*', default=None,
                    help='explicit object ids (skips sampling)')
    ap.add_argument('--list_name', default='pilot_objects.txt')
    ap.add_argument('--exclude_existing', action='store_true',
                    help='also exclude objects already present in dest')
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)

    extra_excl = set(os.listdir(args.dest)) if args.exclude_existing else set()
    ids = args.ids if args.ids else sample_objects(args.n, args.seed, extra_excl)
    print(f'objects: {ids}', flush=True)
    with open(os.path.join(args.dest, args.list_name), 'w') as fw:
        fw.write('\n'.join(ids) + '\n')

    # 1. copy source files
    for cid in ids:
        obj, step = find_files(os.path.join(ABC_DIR, cid))
        tgt = os.path.join(args.dest, cid)
        os.makedirs(f'{tgt}/renders_pyvista', exist_ok=True)
        shutil.copy(obj, f'{tgt}/{os.path.basename(obj)}')
        shutil.copy(step, f'{tgt}/{os.path.basename(step)}')
        shutil.copy(obj, f'{tgt}/renders_pyvista/{cid}.obj')
    print('[progress] source files copied', flush=True)

    # 2+3. renders (initial + 5 load cases), parallel
    tasks = [(cid, args.dest, None) for cid in ids]
    tasks += [(cid, args.dest, lc) for cid in ids for lc in LOAD_CASES]
    ndone = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(render_task, t) for t in tasks]
        for f in as_completed(futs):
            ndone += 1
            print(f.result(), flush=True)
            print(f'[progress] {ndone}/{len(tasks)} render tasks done', flush=True)

    # 4. ortho view selector
    from infer import infer_NN
    for i, cid in enumerate(ids):
        mesh_obj = f'{args.dest}/{cid}/renders_pyvista/{cid}.obj'
        chosen = infer_NN(mesh_obj, './model_saves/ep5_val0.0338.pth',
                          choose_top=10, min_angle_diff=30.0)
        with open(f'{args.dest}/{cid}/pred_ortho_views2.log', 'w') as fw:
            for v in chosen:
                fw.write(f'{v}\n')
        print(f'[progress] ortho preds {i + 1}/{len(ids)} ({cid})', flush=True)

    print('[progress] ALL PREP DONE', flush=True)


if __name__ == '__main__':
    main()
