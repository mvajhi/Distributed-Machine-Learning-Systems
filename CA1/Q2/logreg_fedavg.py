import numpy as np
from sklearn.model_selection import train_test_split
import time
from mpi4py import MPI
import math
import csv
import fcntl
import torch
import torch.nn as nn
import argparse


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
count = comm.Get_size()

# COLORS = [
#     "\033[91m",  # Red
#     "\033[92m",  # Green
#     "\033[93m",  # Yellow
#     "\033[94m",  # Blue
#     "\033[95m",  # Magenta
#     "\033[96m",  # Cyan
#     "\033[90m",  # Gray
#     "\033[97m",  # White
# ]

COLORS = [
    "\033[0m",  # Default (Rank 0)
    "\033[90m",  # Gray
    "\033[90m",  # Gray
    "\033[90m",  # Gray
]
RESET = "\033[0m"

color = COLORS[rank % len(COLORS)]

log_prefix = f"[Rank {rank:02d}]"

PATH = "Data/"
EPOCH = 10
ROUND = 1
LR = 0.01
SEED = 0
malicious = False

np.random.seed(SEED)
torch.manual_seed(SEED)

X, y = None, None
X_train, X_test, y_train, y_test = [None]*4

n_features = 50

train_time = 0

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        self.sigmoid = nn.Sigmoid()
        
        torch.nn.init.constant_(self.linear.weight, 0)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
model = LogisticRegression(n_features)
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
weights = model.state_dict()

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
        print(f"{color}{log_prefix} | Init  | Coordinator node. Reading and scattering data...{RESET}")
        data = read_data()
        data = [(np.zeros((1,1)), np.zeros(1))] + data
    data = comm.scatter(data, root=0)
    X, y = data
    if rank != 0:
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        X_train = torch.from_numpy(X_train.astype(np.float32))
        X_test = torch.from_numpy(X_test.astype(np.float32))
        y_train = torch.from_numpy(y_train.astype(np.float32))
        y_test = torch.from_numpy(y_test.astype(np.float32))

        y_train = y_train.view(y_train.shape[0], 1)
        y_test = y_test.view(y_test.shape[0], 1)
    
    if rank == 0:
        print(f"{color}{log_prefix} | Init  | Data scatter complete.{RESET}")
    else:
        print(f"{color}{log_prefix} | Init  | Data loaded. Train: {X_train.shape[0]}, Test: {X_test.shape[0]}{RESET}")

def train():
    global model, train_time, EPOCH
    stime = time.perf_counter()
    
    for epoch in range(EPOCH):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # if (epoch + 1) % EPOCH == 0 or (epoch + 1) % 5 == 0:
        #      print(f'{color}{log_prefix} | Train | Epoch: {epoch+1}/{EPOCH}, Loss: {loss.item():.4f}{RESET}')
    
    etime = time.perf_counter()
    exec_time = etime - stime
    train_time += exec_time

def test():
    global model
    acc = 0
    if rank != 0:
        with torch.no_grad():
            y_predicted = model(X_test)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
        print(f'{color}{log_prefix} | Test  | Local Accuracy: {acc:.4f}{RESET}')
        
    acc = comm.reduce(acc, MPI.SUM, root=0)
    
    if rank == 0:
        global_acc = acc / (count - 1)
        print(f'{color}{log_prefix} | Test  | === Global Avg Accuracy: {global_acc:.4f} ==={RESET}')

def get_avg_weights(weights_all):
    global weights
    avg_weights = model.state_dict()
    for i in avg_weights.keys():
        all_tensors = [w[i] for w in weights_all]
        stacked_tensors = torch.stack(all_tensors)
        avg_weights[i] = torch.mean(stacked_tensors, dim=0)
    
    print(f'{color}{log_prefix} | Agg   | global weights, bias: {avg_weights['linear.weight'][0][:3]}, {avg_weights["linear.bias"].item()}{RESET}')
    weights = avg_weights

def send_weight():
    global model, weights
    weights = model.state_dict()
    print(f'{color}{log_prefix} | Sync  | Sending weights. Local weights, bias: {weights['linear.weight'][0][:3]}, {weights["linear.bias"].item()}{RESET}')
    
    weights = comm.gather(weights, root=0)
    
    if rank == 0:
        weights = weights[1:]

def receive_weight():
    global model, weights
    weights = comm.bcast(weights, root=0)
    
    if rank == 0:
        return
    
    model.load_state_dict(weights)
    print(f'{color}{log_prefix} | Sync  | Received new global weights.{RESET}')
    
    
def main():
    global weights
    if rank == 0:
        print(f"{color}{log_prefix} | Main  | Starting Federated Learning with {count} processes.{RESET}")
        
    load_data()
    
    for _ in range(ROUND):
        if rank != 0:
            train()
        send_weight()
        if rank == 0:
            get_avg_weights(weights)
        receive_weight()
    test()

def store_time(exec_time, filename):
    with open(filename, 'a') as csvfile:
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([rank, exec_time])
        fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)

def set_malicious():
    global LR, optimizer, malicious
    if rank == 1:
        print(f"{color}{log_prefix} | BadBoy| Change LR to 0.5.{RESET}")
        LR = 0.5
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    malicious = True

def read_arg():
    global ROUND, EPOCH
    parser = argparse.ArgumentParser(description="Bench args")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--malicious", action="store_true")
    args = parser.parse_args()
    EPOCH = args.epoch
    ROUND = args.round
    malicious = args.malicious
    if args.malicious:
        set_malicious()

if __name__ == "__main__":
    read_arg()
    stime = time.perf_counter()
    main()
    etime = time.perf_counter()
    exec_time = etime - stime
    print(f"{color}{log_prefix} | Done  | Training time: {train_time:.2f} seconds.{RESET}")
    print(f"{color}{log_prefix} | Done  | Total execution time: {exec_time:.2f} seconds.{RESET}")
    malicious_text = '_malicious' if malicious else ''
    store_time(exec_time, f"time_exec_fed_{ROUND}_{EPOCH}{malicious_text}.csv")
    store_time(train_time, f"time_train_fed_{ROUND}_{EPOCH}{malicious_text}.csv")