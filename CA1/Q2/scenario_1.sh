#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=256m
# Max job duration
#SBATCH --time=00:02:00
#SBATCH --job-name=scenario_1-vajhi
#SBATCH --output=scenario_1.out

srun python logreg_fedavg.py --epoch 10 --round 1