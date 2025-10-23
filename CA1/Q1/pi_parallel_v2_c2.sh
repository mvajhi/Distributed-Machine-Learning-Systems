#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
# Max job duration
#SBATCH --time=00:02:00
#SBATCH --job-name=pi-v2-vajhi
#SBATCH --output=pi-v2.out

srun python pi_parallel_v2.py