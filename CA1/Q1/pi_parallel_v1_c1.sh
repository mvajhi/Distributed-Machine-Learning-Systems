#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
# Max job duration
#SBATCH --time=00:02:00
#SBATCH --job-name=pi-v1-vajhi
#SBATCH --output=pi-v1.out

srun python pi_parallel_v1.py