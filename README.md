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
-- sprofile report (node27) --
  Time:       0:00:25  /  1:00:00
  CPU load:       0.9  /   2.0
  RAM peak:        3G  /    8G
  GPU load:       0.9  /   1.0
  GPU peak mem:    3G  /   32G
  GPU energy:     0.0kWh
```
