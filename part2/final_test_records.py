"""Save test execution records with the unchanged course SQL helper."""
from pathlib import Path
import pickle
import subprocess
import sys


def main():
    sql_path = Path("results/t5_ft_experiment_test.sql")
    record_path = Path("records/t5_ft_experiment_test.pkl")
    questions = Path("data/test.nl").read_text().splitlines()
    queries = sql_path.read_text().splitlines()
    if len(queries) != len(questions) or not all(q.strip() for q in queries):
        raise ValueError("Test questions and nonempty SQL predictions must align.")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    worker = r'''
import os
from pathlib import Path
from utils import save_queries_and_records
sql_path = "results/t5_ft_experiment_test.sql"
record_path = "records/t5_ft_experiment_test.pkl"
queries = Path(sql_path).read_text().splitlines()
save_queries_and_records(queries, sql_path, record_path)
print("Test records saved.", flush=True)
# The output file is closed; stop any lingering SQL worker threads.
os._exit(0)
'''
    subprocess.run([sys.executable, "-u", "-c", worker], check=True, timeout=300)
    with record_path.open("rb") as stream:
        records, errors = pickle.load(stream)
    assert len(records) == len(errors) == len(queries)
    print("SQL count:", len(queries))
    print("Record count:", len(records))
    print("Execution errors:", sum(bool(e) for e in errors))
    print("Timeouts:", sum("timed out" in str(e).lower() for e in errors))
    print("Submission file checks: PASS")


if __name__ == "__main__":
    main()
