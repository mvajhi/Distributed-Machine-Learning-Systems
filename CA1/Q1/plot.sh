#!/bin/bash

for ((i = 0 ; i < 5 ; i++)); do
    sbatch pi_parallel_v2_c2.sh
    sbatch pi_parallel_v2_c1.sh
    sbatch pi_parallel_v1_c2.sh
    sbatch pi_parallel_v1_c1.sh
    sbatch pi_serial.sh
done
