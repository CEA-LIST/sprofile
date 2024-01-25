import argparse
import os
import re
import socket
import subprocess
import time
from datetime import timedelta
import shelve
from abc import ABC
from typing import Any

try:
    import pynvml
except ImportError:
    pynvml = None


cache_dir = os.environ.get("CACHEDIR", os.path.expanduser("~/.cache"))
job_id = os.environ["SLURM_JOB_ID"]
job_info = subprocess.check_output(f"scontrol show job {job_id} -o".split(), text=True)
hostname = socket.gethostname()


def num_gpus():
    if pynvml is None:
        return 0
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount()
    finally:
        pynvml.nvmlShutdown()


class Semaphore:
    def __init__(self, lock_file, timeout=10) -> None:
        self.lock_file = lock_file
        self.timeout = timeout

    def __enter__(self):
        timeout = self.timeout
        while timeout >= 0:
            try:
                os.open(self.lock_file, os.O_CREAT | os.O_EXCL)
            except FileExistsError:
                time.sleep(0.1)
                timeout -= 0.1
            else:
                return

        raise TimeoutError

    def __exit__(self, type, value, traceback):
        os.remove(self.lock_file)


class NVMLHandle:
    def __init__(self, i):
        self.i = i

    def __enter__(self):
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(self.i)

    def __exit__(self, type, value, traceback):
        pynvml.nvmlShutdown()


class Monitor(ABC):
    @staticmethod
    def start(db: shelve.Shelf):
        raise NotImplementedError

    @staticmethod
    def stop(db: shelve.Shelf) -> Any:
        raise NotImplementedError


class TimeStats(Monitor):
    @staticmethod
    def start(db):
        pass

    @staticmethod
    def stop(db):
        rsv_time = re.search(
            r".*TimeLimit=(([0-9]+)-)?([0-9]+):([0-9]+):([0-9]+).*", job_info
        )
        _, days, hours, minutes, seconds = map(int, rsv_time.groups(default="0"))
        rsv_time = timedelta(
            seconds=days * 86400 + hours * 3600 + minutes * 60 + seconds
        )

        run_time = re.search(
            r".*RunTime=(([0-9]+)-)?([0-9]+):([0-9]+):([0-9]+).*", job_info
        )
        _, days, hours, minutes, seconds = map(int, run_time.groups(default="0"))
        run_time = timedelta(
            seconds=days * 86400 + hours * 3600 + minutes * 60 + seconds
        )

        return run_time, rsv_time


class NVMLStats(Monitor):
    @staticmethod
    def usage_stats(h):
        # if pynvml.nvmlVgpuInstanceGetAccountingMode(h) != 1:
        #     print("WARN: accounting mode required for sprofile GPU usage statistics.")

        start_time = re.search(r".*StartTime=([^\s]+).*", job_info).group(1)
        start_time = (
            int(time.mktime(time.strptime(start_time, "%Y-%m-%dT%H:%M:%S"))) * 1e6
        )

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

    @staticmethod
    def start(db):
        energy_usage_old = []
        for i in range(num_gpus()):
            with NVMLHandle(i) as h:
                energy_usage_old.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))

        db["energy_usage_old"] = energy_usage_old

    @staticmethod
    def stop(db):
        """Gather GPU metrics using NVML API.

        :return: a tuple with the following fields:
            - total load
            - number of GPUs
            - peak GPU memory in GB
            - min memory of involved GPUs in GB
            - total energy usage in kWh
        """
        energy_usage_old = db["energy_usage_old"]

        energy_usage = []
        gpu_avg_load = []
        gpu_peak_mem = []
        gpu_total_mem = []
        for i in range(num_gpus()):
            with NVMLHandle(i) as h:
                energy_usage.append(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))

                l, p, t = NVMLStats.usage_stats(h)
                gpu_avg_load.append(l)
                gpu_peak_mem.append(p)
                gpu_total_mem.append(t)

        energy_used = sum(
            (n - o) / 3.6e9 for n, o in zip(energy_usage, energy_usage_old)
        )

        return (
            sum(gpu_avg_load),
            len(gpu_avg_load),
            max(gpu_peak_mem) / 1024**3,
            min(gpu_total_mem) / 1024**3,
            energy_used,
        )


class CGroupStats(Monitor):
    @staticmethod
    def start(db):
        with open("/sys/fs/cgroup/cpuacct/cpuacct.usage_percpu") as f:
            usage_percpu_old = list(map(int, f.read().strip().split()))

        db["usage_percpu_old"] = usage_percpu_old

    @staticmethod
    def stop(db):
        with open(
            f"/sys/fs/cgroup/memory/slurm/uid_{os.getuid()}/job_{job_id}/memory.max_usage_in_bytes"
        ) as f:
            memory_max = int(f.read().strip())

        with open(
            f"/sys/fs/cgroup/memory/slurm/uid_{os.getuid()}/job_{job_id}/memory.limit_in_bytes"
        ) as f:
            memory_limit = int(f.read().strip())

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

        with open(f"/sys/fs/cgroup/cpuacct/cpuacct.usage_percpu") as f:
            usage_percpu = list(map(int, f.read().strip().split()))

        usage_percpu_old = db["usage_percpu_old"]

        cpu_times = [usage_percpu[i] - usage_percpu_old[i] for i in cpuset]

        run_time = re.search(
            r".*RunTime=(([0-9]+)-)?([0-9]+):([0-9]+):([0-9]+).*", job_info
        )
        _, days, hours, minutes, seconds = map(int, run_time.groups(default="0"))
        run_time = days * 86400 + hours * 3600 + minutes * 60 + seconds

        cpu_load = [t / (run_time * 1e9) for t in cpu_times]

        return (
            sum(cpu_load),
            len(cpu_load),
            memory_max / 1024**3,
            memory_limit / 1024**3,
        )


def main():
    if os.environ["SLURM_LOCALID"] != "0":
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop"])
    args = parser.parse_args()

    os.makedirs(cache_dir, exist_ok=True)

    with shelve.open(f"{cache_dir}/sprofile.{job_id}.{hostname}") as db:
        if args.action == "start":
            TimeStats.start(db)
            CGroupStats.start(db)
            if num_gpus() > 0:
                NVMLStats.start(db)

        elif args.action == "stop":
            with Semaphore(f"{cache_dir}/sprofile.{job_id}.lock"):
                print(f"-- sprofile report ({hostname}) --")

                run_time, rsv_time = TimeStats.stop(db)
                print(f"  Time:  {str(run_time):>12s}  /  {str(rsv_time):s}")

                cpu_load, num_cpus, memory_max, memory_limit = CGroupStats.stop(db)
                print(f"  CPU load:      {cpu_load:4.1f}  /  {num_cpus:4.1f}")
                print(f"  RAM peak:      {memory_max:3.0f}G  /  {memory_limit:3.0f}G")

                if num_gpus() > 0:
                    (
                        gpu_avg_load,
                        _,
                        gpu_peak_mem,
                        gpu_avail_mem,
                        energy_used,
                    ) = NVMLStats.stop(db)

                    print(f"  GPU load:      {gpu_avg_load:4.1f}  /  {num_gpus():4.1f}")
                    print(
                        f"  GPU peak mem:  {gpu_peak_mem:3.0f}G  /  {gpu_avail_mem:3.0f}G"
                    )
                    print(f"  GPU energy:    {energy_used:4.1f}kWh")

        else:
            raise ValueError("invalid action argument")


if __name__ == "__main__":
    main()
