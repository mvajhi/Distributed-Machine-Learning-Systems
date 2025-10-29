import numpy as np
import time
import argparse

blas = np.show_config('dicts')['Build Dependencies']['blas']['name']

parser = argparse.ArgumentParser(description="Bench args")
parser.add_argument("--action", type=str, default='det', help="Set matrix action", choices=['det', 'inv'])
parser.add_argument("--size", type=int, default=5000, help="Size of matrix")
parser.add_argument("--seed", type=int, default=0, help="Set random seed")

args = parser.parse_args()

SIZE = args.size
SEED = args.seed
ACTION = args.action

np.random.seed(SEED)
m = np.random.rand(SIZE,SIZE)

stime = time.perf_counter()

if ACTION == 'det':
    det = np.linalg.det(m)
elif ACTION == 'inv':
    inv = np.linalg.inv(m)

etime = time.perf_counter()
exec_time = etime - stime

space = '\t\t' if len(blas) < 10 else '\t'
print(f'time \tfor {ACTION} \t{SIZE/1000}k*{SIZE/1000}k \twith {blas}: {space}{exec_time:0.2f}')
