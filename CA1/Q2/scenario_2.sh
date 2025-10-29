#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=256m
# Max job duration
#SBATCH --time=00:02:00
#SBATCH --job-name=scenario_2-vajhi
#SBATCH --output=scenario_2.out

srun python logreg_fedavg.py --epoch 1 --round 10