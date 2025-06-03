#!/bin/bash

#SBATCH -p RM-shared
#SBATCH -t 15:00:00
#SBATCH --cpus-per-task=32
#SBATCH -n 1
#SBATCH -o /ocean/projects/cis240030p/canovill/amelia/AmeliaScenes/logs/slurm_logs/%j.out


airport=$1

set -x
. /jet/home/canovill/.bashrc

conda activate amelia

cd /ocean/projects/cis240030p/canovill/amelia/AmeliaScenes

srun  python amelia_scenes/run_processor.py airport=${airport} traj_version=a42v01 graph_version=a42v01