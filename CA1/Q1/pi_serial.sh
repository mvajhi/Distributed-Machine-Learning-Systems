#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Max job duration
#SBATCH --time=00:02:00
#SBATCH --job-name=pi-ser-vajhi
#SBATCH --output=pi-ser.out

srun python pi_serial.py