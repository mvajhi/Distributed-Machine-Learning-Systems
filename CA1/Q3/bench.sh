#!/bin/bash

python bench.py --action $1 --size $2 --seed $3 2> /dev/null
source .venv_mkl/bin/activate
python bench.py --action $1 --size $2 --seed $3 2> /dev/null
deactivate