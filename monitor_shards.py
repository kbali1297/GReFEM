#!/usr/bin/env python
"""Live tqdm progress bar aggregated over parallel shard workers
(compute_local_error_tables.py / solve_fine_reference.py).

Usage:
    python monitor_shards.py hs15_shards            # all logs in dir
    python monitor_shards.py 'hs15_shards/bending_*.log' torsion_shards
Counts per-worker "[progress] i/N objects done" lines, sums them into one
bar, and exits when no worker process is running and the bar is full
(or nothing moved for a while after workers exited).

A central status file <first_dir>/PROGRESS.txt is rewritten every refresh
with percent, rate, ETA and per-worker activity, so progress can be checked
any time with:  cat <shard_dir>/PROGRESS.txt
"""
import datetime
import glob
import os
import re
import subprocess
import sys
import time

from tqdm import tqdm

PROG_RE = re.compile(r"\[progress\] (\d+)/(\d+) objects done")
ACT_RE = re.compile(r"\[(mesh|solve|done|skip|ok|fail)\] (\S+)")


def expand_logs(args):
    logs = []
    for a in args:
        if os.path.isdir(a):
            logs += glob.glob(os.path.join(a, "*.log"))
        else:
            logs += glob.glob(a)
    logs = [l for l in logs if os.path.basename(l) != "monitor.log"]
    return sorted(set(logs))


def scan(logs):
    """Return (done, total, per_worker) summed over worker logs.
    per_worker maps name -> (done, total, last_activity_str)."""
    done = total = 0
    per_worker = {}
    for lf in logs:
        try:
            with open(lf, errors="ignore") as f:
                txt = f.read()
        except OSError:
            continue
        matches = PROG_RE.findall(txt)
        if matches:
            d, t = map(int, matches[-1])
        else:
            # worker started but no object finished yet; parse total from
            # the object headers if present
            heads = re.findall(r"\[(\d+)/(\d+)\] OBJECT", txt)
            d, t = 0, (int(heads[-1][1]) if heads else 0)
        acts = ACT_RE.findall(txt)
        act = f"[{acts[-1][0]}] {acts[-1][1]}" if acts else "(starting)"
        done += d
        total += t
        per_worker[os.path.basename(lf).replace(".log", "")] = (d, t, act)
    return done, total, per_worker


def write_status(path, done, total, per_worker, alive, t_start, d_start):
    now = time.time()
    elapsed = now - t_start
    rate = (done - d_start) / elapsed if elapsed > 0 else 0.0  # obj/s
    remaining = max(total - done, 0)
    if rate > 0:
        eta_s = remaining / rate
        eta = str(datetime.timedelta(seconds=int(eta_s)))
        eta_at = datetime.datetime.now() + datetime.timedelta(seconds=eta_s)
        eta += f" (finish ~{eta_at:%H:%M})"
    else:
        eta = "unknown (no completions yet since monitor start)"
    pct = 100.0 * done / total if total else 0.0
    nbar = 40
    filled = int(nbar * pct / 100)
    lines = [
        f"updated  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"progress [{'#' * filled}{'.' * (nbar - filled)}] "
        f"{done}/{total} ({pct:.1f}%)",
        f"workers  {alive} alive",
        f"rate     {rate * 3600:.1f} obj/h (since monitor start)",
        f"ETA      {eta}",
        "",
    ]
    for w, (d, t, act) in sorted(per_worker.items()):
        lines.append(f"  {w:24s} {d}/{t}  {act}")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)


def workers_alive():
    n = 0
    for proc in ("compute_local_error_tables.py", "solve_fine_reference.py",
                 "run_experiments_extra.py"):
        try:
            out = subprocess.run(["pgrep", "-fc", proc],
                                 capture_output=True, text=True)
            n += int(out.stdout.strip() or 0)
        except Exception:
            pass
    return n


def main():
    args = sys.argv[1:] or ["."]
    total_override = 0
    for a in list(args):
        if a.startswith("--total="):
            total_override = int(a.split("=")[1])
            args.remove(a)
    logs = expand_logs(args)
    if not logs:
        sys.exit(f"no logs matching {args}")
    first_dir = os.path.dirname(logs[0]) or "."
    status_path = os.path.join(first_dir, "PROGRESS.txt")
    done, total, per_worker = scan(logs)
    total = max(total, total_override)
    t_start, d_start = time.time(), done
    write_status(status_path, done, total, per_worker,
                 workers_alive(), t_start, d_start)
    bar = tqdm(total=max(total, 1), initial=done, unit="obj",
               desc=f"{len(logs)} workers", dynamic_ncols=True)
    idle_after_exit = 0
    while True:
        time.sleep(10)
        done, total, per_worker = scan(logs)
        total = max(total, total_override)
        if total != bar.total and total > 0:
            bar.total = total
        bar.n = done
        alive = workers_alive()
        write_status(status_path, done, total, per_worker,
                     alive, t_start, d_start)
        lag = [w for w, (d, t, _) in per_worker.items() if t and d < t]
        bar.set_postfix_str(
            f"alive={alive}" + (f" pending={','.join(lag[:3])}"
                                + ("..." if len(lag) > 3 else "")
                                if lag else ""))
        bar.refresh()
        if total and done >= total:
            break
        if alive == 0:
            idle_after_exit += 1
            if idle_after_exit >= 3:  # 30s grace after workers exit
                break
        else:
            idle_after_exit = 0
    bar.close()
    done, total, per_worker = scan(logs)
    write_status(status_path, done, total, per_worker,
                 workers_alive(), t_start, d_start)
    incomplete = {w: dt[:2] for w, dt in per_worker.items() if dt[0] < dt[1]}
    if incomplete:
        print("workers that did not finish:")
        for w, (d, t) in sorted(incomplete.items()):
            print(f"  {w}: {d}/{t}")
    else:
        print("all workers finished.")


if __name__ == "__main__":
    main()
