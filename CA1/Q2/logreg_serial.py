import numpy as np
from sklearn.model_selection import train_test_split
import time
import torch
import torch.nn as nn

PATH = "Data/"
SEED = 0

np.random.seed(SEED)
torch.manual_seed(SEED)

X, y = None, None

for i in range(1,4):
    d = np.load(PATH + f"data{i}.npy")
    l = np.load(PATH + f"labels{i}.npy")
    if X is None:
        X, y = d, l
    else:
        X = np.concatenate((X, d), axis=0)
        y = np.concatenate((y, l), axis=0)
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

n_features = X.shape[1]

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


stime = time.perf_counter()

for epoch in range(30):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1) % 5 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

etime = time.perf_counter()
exec_time = etime - stime

with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
print('time', exec_time)
