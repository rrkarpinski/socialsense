import subprocess
import time
import json
import os
from datetime import datetime

# NUM_WORKERS_LIST = [4, 6]              # for 'normal...'
# PERSISTENT_LIST = [True]
# SCENARIOS = ['normal.onlysoc']         # normal scenario(s) you want

# Then, for HDF5 part:
NUM_WORKERS_LIST = [0]
PERSISTENT_LIST = [False]
SCENARIOS = ['hdf5.onlysoc']


# NUM_WORKERS_LIST = [0, 2, 4, 6]
# PERSISTENT_LIST = [False, True]
# SCENARIOS = ['normal.onlysoc', 'hdf5.onlysoc']
CONFIG_PATH = "train_config.py"
MODEL_SCRIPT = "train_models.py"
LOG_DIR = "logs"
RESULTS_FILE = "benchmark_results.json"
TEMP_CONFIG = "temp_config.py"

os.makedirs(LOG_DIR, exist_ok=True)
results = []

def ts():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

def write_temp_config(scenario, num_workers, persistent_workers):
    with open(TEMP_CONFIG, "w") as f0, open(CONFIG_PATH, "r") as f1:
        f0.write(f1.read())
        f0.write(
            f"\nnum_workers = {num_workers}\n"
            f"persistent_workers = {persistent_workers}\n"
            f"selected_scenario = '{scenario}'\n"
        )

def run_one(scenario, num_workers, persistent_workers):
    print(f"{ts()} [START] {scenario=} {num_workers=} {persistent_workers=}")
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"}
    log_file = f"{LOG_DIR}/run_{scenario}_w{num_workers}_p{persistent_workers}.log"
    write_temp_config(scenario, num_workers, persistent_workers)
    print(f"{ts()} [created logfile] {log_file=}")
    t0 = time.time()
    status, error = "OK", None
    print(f"{ts()} [starting script] ...")
    try:
        with open(log_file, "w") as log:
            p = subprocess.run(
                ["python", MODEL_SCRIPT, TEMP_CONFIG],
                stdout=log,
                stderr=subprocess.PIPE,
                env=env,
                check=True,
                encoding='utf-8',
            )
    except subprocess.CalledProcessError as e:
        status, error = "FAILED", e.stderr
        print(f"{ts()} [ERROR] {scenario=} {num_workers=} {persistent_workers=}")
        print(f"{ts()} {e.stderr or str(e)}")
        print(f"{ts()} Check log: {log_file}")
    else:
        if p.stderr:
            print(f"{ts()} [WARNING] stderr output during successful run:")
            print(p.stderr)
    duration = time.time() - t0
    print(f"{ts()} [DONE] {scenario} {num_workers=} {persistent_workers=} status={status} time={duration:.1f}s")
    results.append({
        "scenario": scenario,
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
        "status": status,
        "error": error,
        "duration": duration,
        "log_file": log_file,
    })
    save_results()
    # Optionally remove temp config after each run
    try: os.remove(TEMP_CONFIG)
    except Exception: pass
    print('results saved, temp confing cleaned')

def load_resume_point():
    if not os.path.exists(RESULTS_FILE):
        return None
    try:
        with open(RESULTS_FILE) as f:
            data = json.load(f)
        if not data:
            return None
        last = data[-1]
        return (
            SCENARIOS.index(last["scenario"]),
            NUM_WORKERS_LIST.index(last["num_workers"]),
            PERSISTENT_LIST.index(last["persistent_workers"]),
        )
    except Exception:
        return None

def save_results():
    with open(RESULTS_FILE, "w") as out:
        json.dump(results, out, indent=2)

def main():
    global results
    resume_from = load_resume_point()
    if resume_from:
        print(f"{ts()} Resuming from {resume_from}")
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    for i_s, scenario in enumerate(SCENARIOS):
        for i_w, num_workers in enumerate(NUM_WORKERS_LIST):
            for i_p, persistent in enumerate(PERSISTENT_LIST):
                if num_workers == 0 and persistent:
                    continue
                key = (i_s, i_w, i_p)
                if resume_from and key <= resume_from:
                    continue
                run_one(scenario, num_workers, persistent)
    print(f"\n{ts()} All benchmark results saved.")

if __name__ == "__main__":
    main()
