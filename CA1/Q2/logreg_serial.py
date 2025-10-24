from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import time

PATH = "Data/"

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

n_features = X.shape[1]
n_classes = 1
initial_coeffs = np.zeros((n_classes, n_features))
initial_intercept = np.zeros((n_classes,))

stime = time.perf_counter()
model = SGDClassifier(
        loss='log_loss',
        learning_rate='constant',
        eta0=0.01,
        max_iter=30,
        random_state=0
    ).fit(X_train, y_train, coef_init=initial_coeffs, intercept_init=initial_intercept)
etime = time.perf_counter()
exec_time = etime - stime

print('acc',model.score(X_test, y_test))
print('time', exec_time)
