import fcntl
from mpi4py import MPI
import math
import time
import csv
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
count = comm.Get_size()

N = 500_000

def compute_term(k):
    s = 0
    for i in range(k // 1000):
        s += i * i
    return 1.0 / (k * k)

def get_sum_k():
    sum_k = 0
    for i in range(rank+1, N+1, count):
        sum_k += compute_term(i)
    return sum_k

def calc_result(sum_k):
    for _ in range(count-1):
        sum_k += comm.recv(source=MPI.ANY_SOURCE, tag=1)
    
    result = math.sqrt(6*sum_k)
    
    print(result)
    
def main():
    sum_k = get_sum_k()
    if rank == 0:
        calc_result(sum_k)
    else:
        comm.send(sum_k, dest=0, tag=1)
        
def store_time(exec_time):
    filename = f"time_p2_{count}.csv"
    
    with open(filename, 'a') as csvfile:
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([rank, exec_time])
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)

if __name__ == "__main__":
    stime = time.perf_counter()
    main()
    etime = time.perf_counter()
    exec_time = etime - stime
    store_time(exec_time)
