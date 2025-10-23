import math
import time
import csv
import os
import fcntl

N = 500_000

def compute_term(k):
    s = 0
    for i in range(k // 1000):
        s += i * i
    return 1.0 / (k * k)

def main():
    sum_k = 0
    for i in range(1, N+1):
        sum_k += compute_term(i)
    result = math.sqrt(6*sum_k)
    print(result)

def store_time(exec_time):
    filename = "time_s.csv"
    rank = 0
    
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