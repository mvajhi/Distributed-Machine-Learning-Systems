#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=256m
# Max job duration
#SBATCH --time=00:02:00
#SBATCH --job-name=malicious-vajhi
#SBATCH --output=malicious.out

srun python logreg_fedavg.py --epoch 1 --round 10 --malicious