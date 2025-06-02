#!/bin/bash

#SBATCH -p RM-shared
#SBATCH -t 8:00:00
#SBATCH --cpus-per-task=32
#SBATCH -n 1
#SBATCH -o /ocean/projects/cis240030p/canovill/Amelia/AmeliaScenes/logs/slurm_logs/%j.out

airport=$1

set -x
. /jet/home/canovill/.bashrc

conda activate amelia

cd /ocean/projects/cis220039p/jaypat/AMELIA_NEURIPS/AmeliaSWIM

srun python process_parrallel.py data=${data} airports=${airport}