import argparse
import os
import re
import socket
import subprocess
import sys
import time
from datetime import timedelta

try:
    import pynvml
except ImportError:
    pynvml = None


cache_dir = os.environ.get("CACHEDIR", os.path.expanduser("~/.cache"))
job_id = os.environ["SLURM_JOB_ID"]


class Handle:
    def __init__(self, i):
        self.i = i

    def __enter__(self):
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(self.i)

    def __exit__(self, type, value, traceback):
        pynvml.nvmlShutdown()


def gpu_stats(i):
    job_id = os.environ["SLURM_JOB_ID"]
    job_info = subprocess.check_output(
        f"scontrol show job {job_id} -o".split(), text=True
    )
    start_time = re.search(r".*StartTime=([^\s]+).*", job_info).group(1)
    start_time = int(time.mktime(time.strptime(start_time, "%Y-%m-%dT%H:%M:%S"))) * 1e6

    with Handle(i) as h:
        stats = [
            pynvml.nvmlDeviceGetAccountingStats(h, p)
            for p in pynvml.nvmlDeviceGetAccountingPids(h)
        ]
        total_mem = pynvml.nvmlDeviceGetMemoryInfo(h).total

    stats = [s for s in stats if s.startTime >= start_time]

    timesplits = sorted(
        [s.startTime for s in stats] + [s.startTime + s.time for s in stats]
    )

    gpu_util = 0
    peak_mem = 0
    for start, stop in zip(timesplits[0:-1], timesplits[1:]):
        gpu_util_split = 0
        peak_mem_split = 0
        for s in stats:
            if s.startTime <= start and s.startTime + s.time >= stop:
                gpu_util_split += s.gpuUtilization
                peak_mem_split += s.maxMemoryUsage

        gpu_util += gpu_util_split * (stop - start)
        peak_mem = max(peak_mem, peak_mem_split)

    if len(timesplits) == 0:
        gpu_util = 0
    else:
        gpu_util = gpu_util / 100 / max(timesplits[-1] - timesplits[0], 1)

    return gpu_util, peak_mem, total_mem


def memory_stats():
    with open(
        f"/sys/fs/cgroup/memory/slurm/uid_{os.getuid()}/job_{job_id}/memory.max_usage_in_bytes"
    ) as f:
        memory_max = int(f.read().strip())

    with open(
        f"/sys/fs/cgroup/memory/slurm/uid_{os.getuid()}/job_{job_id}/memory.limit_in_bytes"
    ) as f:
        memory_limit = int(f.read().strip())

    return memory_max, memory_limit


def cpu_stats():
    with open(
        f"/sys/fs/cgroup/cpuset/slurm/uid_{os.getuid()}/job_{job_id}/cpuset.cpus"
    ) as f:
        line = f.read().strip()
        cpuset = []
        for c in line.split(","):
            if "-" in c:
                start, stop = map(int, c.split("-"))
                cpuset.extend(range(start, stop + 1))
            else:
                cpuset.append(int(c))

    hostname = socket.gethostname()

    with open(f"{cache_dir}/sprofile.{job_id}.{hostname}") as f:
        usage_percpu_old = list(map(int, f.read().strip().split()))

    os.remove(f"{cache_dir}/sprofile.{job_id}.{hostname}")

    with open(f"/sys/fs/cgroup/cpuacct/cpuacct.usage_percpu") as f:
        usage_percpu = list(map(int, f.read().strip().split()))

    cpu_times = [usage_percpu[i] - usage_percpu_old[i] for i in cpuset]

    job_info = subprocess.check_output(
        f"scontrol show job {job_id} -o".split(), text=True
    )
    run_time = re.search(
        r".*RunTime=(([0-9]+)-)?([0-9]+):([0-9]+):([0-9]+).*", job_info
    )
    _, days, hours, minutes, seconds = map(int, run_time.groups(default="0"))
    run_time = days * 86400 + hours * 3600 + minutes * 60 + seconds

    return [t / (run_time * 1e9) for t in cpu_times]


def start():
    job_id = os.environ["SLURM_JOB_ID"]
    hostname = socket.gethostname()
    os.makedirs(cache_dir, exist_ok=True)
    subprocess.run(
        [
            "cp",
            "/sys/fs/cgroup/cpuacct/cpuacct.usage_percpu",
            f"{cache_dir}/sprofile.{job_id}.{hostname}",
        ]
    )


def print_report():
    job_info = subprocess.check_output(
        f"scontrol show job {job_id} -o".split(), text=True
    )

    hostname = socket.gethostname()
    print(f"-- sprofile report ({hostname}) --")

    rsv_time = re.search(
        r".*TimeLimit=(([0-9]+)-)?([0-9]+):([0-9]+):([0-9]+).*", job_info
    )
    _, days, hours, minutes, seconds = map(int, rsv_time.groups(default="0"))
    rsv_time = timedelta(seconds=days * 86400 + hours * 3600 + minutes * 60 + seconds)

    run_time = re.search(
        r".*RunTime=(([0-9]+)-)?([0-9]+):([0-9]+):([0-9]+).*", job_info
    )
    _, days, hours, minutes, seconds = map(int, run_time.groups(default="0"))
    run_time = timedelta(seconds=days * 86400 + hours * 3600 + minutes * 60 + seconds)

    print(f"Time:  {str(run_time):>12s}  /  {str(rsv_time):s}")

    cpu_avg_load = cpu_stats()

    print(f"CPU load:      {sum(cpu_avg_load):4.1f}  /  {len(cpu_avg_load):4.1f}")

    mem_peak, mem_resv = memory_stats()

    print(
        f"RAM peak mem:  {mem_peak / 1024 ** 3:3.0f}G  /  {mem_resv / 1024 ** 3:3.0f}G"
    )

    if pynvml is not None:
        pynvml.nvmlInit()
        device_ids = list(range(pynvml.nvmlDeviceGetCount()))
        pynvml.nvmlShutdown()

        gpu_avg_load, gpu_peak_mem, gpu_total_mem = zip(
            *[gpu_stats(i) for i in device_ids]
        )

        print(f"GPU load:      {sum(gpu_avg_load):4.1f}  /  {len(gpu_avg_load):4.1f}")
        print(
            f"GPU peak mem:  {max(gpu_peak_mem) / 1024 ** 3:3.0f}G  /  {gpu_total_mem[0] / 1024 ** 3:3.0f}G"
        )


def main():
    if os.environ["SLURM_LOCALID"] != "0":
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop"])
    args = parser.parse_args()

    if args.action == "start":
        start()

    elif args.action == "stop":
        while True:
            try:
                os.open(f"{cache_dir}/sprofile.{job_id}.lock", os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                time.sleep(0.1)
            else:
                print_report()
                os.remove(f"{cache_dir}/sprofile.{job_id}.lock")
                break

    else:
        raise ValueError("invalid action argument")


if __name__ == "__main__":
    main()
