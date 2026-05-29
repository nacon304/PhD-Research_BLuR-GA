import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_one(task_id: int, tasks_csv: str, log_dir: Path, threads_per_task: int, dry_run: bool = False):
    env = os.environ.copy()

    # Avoid CPU oversubscription when running many GA tasks locally.
    env["OMP_NUM_THREADS"] = str(threads_per_task)
    env["MKL_NUM_THREADS"] = str(threads_per_task)
    env["OPENBLAS_NUM_THREADS"] = str(threads_per_task)
    env["NUMEXPR_NUM_THREADS"] = str(threads_per_task)
    env["MPLBACKEND"] = "Agg"

    cmd = [
        sys.executable,
        "run_blur_ga.py",
        "run-task",
        "--tasks",
        tasks_csv,
        "--task-id",
        str(task_id),
    ]

    out_path = log_dir / f"local_task_{task_id}.out"
    err_path = log_dir / f"local_task_{task_id}.err"

    if dry_run:
        print(" ".join(cmd))
        return task_id, 0

    with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, stdout=out, stderr=err, env=env)

    return task_id, proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads-per-task", type=int, default=1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    with tasks_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n_tasks = len(rows)
    end = n_tasks if args.end is None else min(args.end, n_tasks)
    task_ids = list(range(args.start, end))

    if args.log_dir is None:
        log_dir = Path("local_logs") / tasks_path.stem
    else:
        log_dir = Path(args.log_dir)

    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tasks: {tasks_path}")
    print(f"Running task_id {args.start} to {end - 1}")
    print(f"Total selected tasks: {len(task_ids)}")
    print(f"Workers: {args.workers}")
    print(f"Threads per task: {args.threads_per_task}")
    print(f"Logs: {log_dir}")

    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(
                run_one,
                task_id,
                str(tasks_path),
                log_dir,
                args.threads_per_task,
                args.dry_run,
            )
            for task_id in task_ids
        ]

        for fut in as_completed(futures):
            task_id, code = fut.result()
            if code == 0:
                print(f"[OK] task {task_id}")
            else:
                print(f"[FAIL] task {task_id} exit={code}")
                failures.append(task_id)

    if failures:
        fail_path = log_dir / "failed_tasks.txt"
        fail_path.write_text("\n".join(map(str, failures)) + "\n", encoding="utf-8")
        print(f"Failed tasks written to: {fail_path}")
        raise SystemExit(1)

    print("All selected tasks finished successfully.")


if __name__ == "__main__":
    main()

"""
$env:DATA_DIR = "../Dataset/prepared_feature_selection"
$env:RESULTS_ROOT = "../Results/results_fs_uniform_positive_medium_highdim"
$env:TASKS_OUT = "hpc/tasks_fs_uniform_positive_medium_highdim.csv"

bash hpc/make_fs_uniform_positive_tasks.sh

python scripts/run_tasks_local.py --tasks hpc/tasks_fs_uniform_positive_medium_highdim.csv --workers 4 --threads-per-task 1

"""