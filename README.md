# SProfile

Sprofile prints the consumed CPU, RAM and GPU resources at the end of a slurm job.

It has the following requirements:

- slurm configured with cgroup for resource limit management (you might need to adjust the cgroup parsing code to your cluster configuration)
- accounting mode enabled for GPU resource logging (`nvidia-smi --accounting-mode=1`)
- pynvml python3 package for GPU resource reading

In order to use sprofile, add the following lines at the beginning of the slurm script:

```sh
srun --ntasks-per-node=1 sprofile start
trap "srun --ntasks-per-node=1 sprofile stop" EXIT
```

This will cause the script to print the following content before exiting:

```
<nodename>
  avg CPU:       ??%
  peak RAM:      ??G /  ??G ( ??%)
  avg GPU load:  ??%
  peak GPU mem:  ??G /  ??G
```
