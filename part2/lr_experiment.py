"""Run a learning-rate-only comparison using the archived baseline source."""
import argparse
import multiprocessing as mp
import os
from pathlib import Path
import importlib
import sys
import traceback
import zipfile
import time


def sql_worker(queries, sender):
    try:
        # A spawned process imports the unmodified course implementation.
        from utils import compute_records
        sender.send((True, compute_records(queries)))
    except BaseException:
        sender.send((False, traceback.format_exc()))
    finally:
        sender.close()


def isolated_compute_records(queries):
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=sql_worker, args=(queries, sender))
    process.start()
    sender.close()
    try:
        # Keep the original 120-second SQL timeout inside the worker.
        if not receiver.poll(300):
            raise TimeoutError("SQL worker did not return; no partial score accepted.")
        success, result = receiver.recv()
        if not success:
            raise RuntimeError(result)
        return result
    finally:
        receiver.close()
        process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()


def prepare(root, run_name):
    run = root / "runs" / run_name
    run.mkdir(parents=True, exist_ok=False)
    names = ["train_t5.py", "load_data.py", "t5_utils.py", "utils.py"]
    with zipfile.ZipFile(root / "baseline_backup.zip") as archive:
        for name in names:
            content = archive.read(name)
            compile(content, name, "exec")
            (run / name).write_bytes(content)
    for name in ["results", "records"]:
        (run / name).mkdir()
    data = root / "data"
    reference = root / "records/ground_truth_dev.pkl"
    for name in ["train.nl", "train.sql", "dev.nl", "dev.sql", "test.nl", "flight_database.db"]:
        assert (data / name).is_file(), f"Missing input: {name}"
    assert reference.is_file()
    (run / "data").symlink_to(data, target_is_directory=True)
    (run / "records/ground_truth_dev.pkl").symlink_to(reference)
    for name in ["load_data.py", "t5_utils.py", "train_t5.py"]:
        assert "tokenizer_utils" not in (run / name).read_text(), "Expected original baseline source"
    print(f"Run directory: {run}", flush=True)
    print("Original baseline source restored; input files checked.", flush=True)
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-sql-worker", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    if args.check_sql_worker:
        run_name = f"lr1e4_sql_check_{time.time_ns()}"
    elif args.prepare_only:
        run_name = "lr1e4_preflight"
    else:
        job = os.environ.get("SLURM_JOB_ID")
        if not job:
            raise RuntimeError("Submit the training run through sbatch.")
        run_name = f"lr1e4_{job}"
    run = prepare(root, run_name)
    if args.prepare_only:
        return
    os.chdir(run)
    sys.path.insert(0, str(run))
    if args.check_sql_worker:
        import utils
        utils.compute_records = isolated_compute_records
        # Use the real archived training module and its imported scoring function.
        training = importlib.import_module("train_t5")
        training.save_queries_and_records(
            ["SELECT 1", "SELECT 2"], "results/check.sql", "records/check.pkl"
        )
        import pickle
        with open("records/check.pkl", "rb") as file:
            records, errors = pickle.load(file)
        assert records == [[(1,)], [(2,)]] and errors == ["", ""]
        print("SQL worker integration check: PASS", flush=True)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    import utils
    # Preserve metric formulas and SQL execution settings, but clean up timed-out workers.
    utils.compute_records = isolated_compute_records
    sys.argv = ["train_t5.py", "--finetune", "--experiment_name", "lr1e4",
                "--learning_rate", "1e-4", "--weight_decay", "0",
                "--batch_size", "8", "--test_batch_size", "8",
                "--scheduler_type", "linear", "--num_warmup_epochs", "0",
                "--max_n_epochs", "20", "--patience_epochs", "5"]
    # Keep this launcher as __main__ so spawn can locate sql_worker.
    training = importlib.import_module("train_t5")
    training.main()
    print("Learning-rate experiment completed.", flush=True)


if __name__ == "__main__":
    main()
