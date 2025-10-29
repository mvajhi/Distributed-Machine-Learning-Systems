#!/bin/bash

RAND=$RANDOM

echo "benchmark with seed $RAND" 

bash ./bench.sh det 10000 $RAND
bash ./bench.sh det 20000 $RAND
bash ./bench.sh inv 10000 $RAND
bash ./bench.sh inv 20000 $RAND