# Running `torch.distributed` trainings on HiPerGator AI

This repository contains example scripts and supporting code to launch a [`torch.distributed`](https://pytorch.org/tutorials/beginner/dist_overview.html) training on HiPerGator AI.  Our experience has been that large distributed runs, especially from singularity containers, sometimes fail because initialization on one or more nodes fails.  A key feature of these scripts is that launch failures on individual nodes can automatically be detected and restarted to prevent the entire job from failing.

To use these scripts, you will need to customize `pt_multinode_launch.sh` for your computing task, then launch `pt_multinode_launch.sh` as the main task via sbatch.  The other scripts should not require modification.


