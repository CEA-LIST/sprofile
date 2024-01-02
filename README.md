# SProfile

Sprofile prints the consumed CPU, RAM and GPU resources at the end of a slurm job.
It parses readily available resource usage information and therefore incurs no overhead.

Sprofile can be installed from pypi or from source:

```sh
pip install sprofile
```

For CPU and RAM statistics, slurm must be configured to use the cgroup plugin.
For GPU resource informations, accounting mode must be unabled in the nvidia driver (`nvidia-smi --accounting-mode=1`).

In order to use sprofile, add the following lines at the beginning and the end of the slurm script:

```sh
#!/usr/bin/env sh

...

srun sprofile start

...

srun sprofile stop
```

The last command will print actual resource utilization:

```
-- sprofile report (node03) --
Time:       0:00:03  /  1:00:00
CPU load:       2.0  /   4.0
RAM peak mem:    7G  /    8G
GPU load:       0.2  /   2.0
GPU peak mem:    7G  /   40G
```
