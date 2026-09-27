"""Run infer_meshpoints.py sweeps for the pilot/extra object set.

Usage:
  python run_experiments_extra.py --parent /data/1bali/GReFEM/test_meshes_extra \
      --objects_file /data/1bali/GReFEM/test_meshes_extra/pilot_objects.txt \
      --num_views 5 --max_jobs 12
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

PYTHON_EXEC = '/data/1bali/miniforge3/envs/multi_view_3DQA/bin/python'
SCRIPT = '/data/1bali/GReFEM/infer_meshpoints.py'
LLM = 'google/gemini-3-flash-preview'
PROMPTS = ['geo_max', 'geo_mid', 'geo_none']
LOAD_CASES = ['compression', 'bending', 'torsion', 'bending_compression', 'torsion_compression']
GRID = 11
RUN = 1


def run_job(job):
    obj, parent, lc, prompt, num_views = job
    short_llm = LLM.split('/')[-1]
    exp = f'{lc}_{short_llm}_{prompt}prompt_ortho_{num_views}views_{GRID}grid_{RUN}run'
    exp_dir = f'{parent}/{obj}/{exp}'
    final_npy = f'{exp_dir}/refinement_points_final.npy'
    prefilt_npy = f'{exp_dir}/refinement_points_prefilt.npy'
    if os.path.exists(final_npy) or os.path.exists(prefilt_npy):
        return f'[skip] {obj}/{exp}'
    os.makedirs(exp_dir, exist_ok=True)
    cmd = [PYTHON_EXEC, '-u', SCRIPT,
           '--mesh_path', f'{parent}/{obj}/renders_pyvista/{obj}.obj',
           '--view_selection_strategy', 'ortho',
           '--prompt_type', prompt,
           '--num_views', str(num_views),
           '--grid_size', str(GRID),
           '--LLM_name', LLM,
           '--run', str(RUN),
           '--load_case', lc]
    with open(f'{exp_dir}/inference.log', 'w') as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=1800)
    return f'[{"ok" if r.returncode == 0 else "fail"}] {obj}/{exp}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parent', default='/data/1bali/GReFEM/test_meshes_extra')
    ap.add_argument('--objects_file', default=None)
    ap.add_argument('--num_views', type=int, default=5)
    ap.add_argument('--max_jobs', type=int, default=12)
    args = ap.parse_args()

    if args.objects_file:
        with open(args.objects_file) as fr:
            objs = [l.strip() for l in fr if l.strip()]
    else:
        objs = sorted(d for d in os.listdir(args.parent)
                      if os.path.isdir(os.path.join(args.parent, d)))

    jobs = [(o, args.parent, lc, p, args.num_views)
            for o in objs for lc in LOAD_CASES for p in PROMPTS]
    print(f'{len(jobs)} jobs over {len(objs)} objects', flush=True)

    ndone = 0
    with ThreadPoolExecutor(max_workers=args.max_jobs) as ex:
        futs = [ex.submit(run_job, j) for j in jobs]
        for f in as_completed(futs):
            ndone += 1
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print(f'[fail] job error: {e}', flush=True)
            print(f'[progress] {ndone}/{len(jobs)} objects done', flush=True)

    print('[progress] ALL JOBS DONE', flush=True)


if __name__ == '__main__':
    main()
