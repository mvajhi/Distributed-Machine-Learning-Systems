import numpy as np
from sklearn.model_selection import train_test_split
import time
from mpi4py import MPI
import csv
import fcntl
import torch
import torch.nn as nn

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
count = comm.Get_size()

PATH = "Data/"
EPOCH = 2
ROUND = 3
LR = 0.01
SEED = 0

np.random.seed(SEED)
torch.manual_seed(SEED)

X, y = None, None
X_train, X_test, y_train, y_test = [None]*4

n_features = 50

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
    print(f"rank {rank}, data: {X.shape}, {y.shape}")

def train():
    global model
    for epoch in range(EPOCH):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print(f'Rank {rank} Epoch: {epoch+1}, Loss: {loss.item():.4f}')

def test():
    global model
    acc = 0
    if rank != 0:
        with torch.no_grad():
            y_predicted = model(X_test)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
        print(f'rank {rank}, acc {acc:.4f}')
    acc = comm.reduce(acc, MPI.SUM, root=0)
    if rank == 0:
        acc /= (count - 1)
        print(f'rank {rank}, acc {acc:.4f}')

def get_avg_weights(weights_all):
    avg_weights = model.state_dict()
    for i in avg_weights.keys():
        all_tensors = [w[i] for w in weights_all]
        stacked_tensors = torch.stack(all_tensors)
        avg_weights[i] = torch.mean(stacked_tensors, dim=0)
    
    print(f'AVG rank {rank}, weight {avg_weights['linear.weight'][0][:3]}, {avg_weights['linear.bias']}')
    return avg_weights

def send_weight():
    global model, weights
    weights = model.state_dict()
    print(f'pre rank {rank}, weight {weights['linear.weight'][0][:3]}, {weights['linear.bias']}')
    
    all_weights = comm.gather(weights, root=0)
    
    if rank == 0:
        return all_weights[1:]

def receive_weight():
    global model, weights
    weights = comm.bcast(weights, root=0)
    
    if rank == 0:
        return
    
    model.load_state_dict(weights)       
    
def main():
    global weights
    load_data()
    for i in range(ROUND):
        if rank != 0:
            train()
        all_weights = send_weight()
        if rank == 0:
            weights = get_avg_weights(all_weights)
        receive_weight()
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
