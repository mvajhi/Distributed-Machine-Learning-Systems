from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import time
from mpi4py import MPI
import math
import csv
import fcntl

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
count = comm.Get_size()

PATH = "Data/"
EPOCH = 2
ROUND = 3

X, y = None, None
X_train, X_test, y_train, y_test = [None]*4

n_features = 50
n_classes = 1
initial_coeffs = np.zeros((n_classes, n_features))
initial_intercept = np.zeros((n_classes,))

model = None

def read_data():
    output = []
    for i in range(1,4):
        d = np.load(PATH + f"data{i}.npy")
        l = np.load(PATH + f"labels{i}.npy")
        output.append((d,l))
    return output

def load_data():
    data = None
    if rank == 0:
        data = read_data()
        data = [(np.zeros((1,1)), np.zeros(1))] + data
    data = comm.scatter(data, root=0)
    X, y = data
    if rank != 0:
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(f"rank {rank}, data: {X.shape}, {y.shape}")

def train():
    global model
    model = SGDClassifier(
        loss='log_loss',
        learning_rate='constant',
        eta0=0.01,
        max_iter=EPOCH,
        random_state=0
    ).fit(X_train, y_train, coef_init=initial_coeffs, intercept_init=initial_intercept)

def send_weight():
    global initial_coeffs, initial_intercept
    initial_coeffs = model.coef_ if model != None else np.zeros((n_classes, n_features))
    initial_intercept = model.intercept_ if model != None else np.zeros((n_classes,))
    print(f'pre rank {rank}, weight {initial_coeffs.sum():0.2f}, {initial_intercept.sum():0.2f}')
    # print(f'post rank {rank}, weight {initial_coeffs.shape}, {initial_intercept.shape}')
    initial_coeffs = comm.gather(initial_coeffs, root=0)
    initial_intercept = comm.gather(initial_intercept, root=0)
    if rank != 0 : 
        return
    initial_coeffs = sum(initial_coeffs) / (count-1)
    initial_intercept = sum(initial_intercept) / (count-1)
    # print(f'post rank {rank}, weight {initial_coeffs.shape}, {initial_intercept.shape}')
    print(f'post rank {rank}, weight {initial_coeffs.sum():0.2f}, {initial_intercept.sum():0.2f}')
    # print(f'post rank {rank}, weight {initial_coeffs}, {initial_intercept}')

def resive_weight():
    global initial_coeffs, initial_intercept, model
    initial_coeffs = comm.bcast(initial_coeffs, root=0)
    initial_intercept = comm.bcast(initial_intercept, root=0)
    
    if rank == 0:
        return
    
    model.coef_ = initial_coeffs
    model.intercept_ = initial_intercept

def test():
    global model
    acc = 0
    if rank != 0:
        acc = model.score(X_test, y_test)
        print(f'rank {rank}, acc {acc}')
    acc = comm.reduce(acc, MPI.SUM, root=0)
    if rank == 0:
        acc /= (count - 1)
        print(f'rank {rank}, acc {acc}')
        
    
def main():
    load_data()
    for i in range(ROUND):
        if rank != 0:
            train()
        send_weight()
        resive_weight()
    test()
  
def store_time(exec_time, filename):
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
    store_time(exec_time, f"time_fed_{ROUND}_{EPOCH}.csv")
