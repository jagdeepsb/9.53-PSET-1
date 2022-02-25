import numpy as np

from plot import plot_data, plot_accuracies
from data import DataSet
from algos import Perceptron, Adaline

def get_percent_acc(model, dset):
    correct = 0
    for x, y, c in dset.data:
        y_hat = model.call(x)
        if y == 0:
            if y_hat == 0:
                correct += 1
        if y_hat/y > 0:
            correct += 1
    return correct / len(dset.data) 
        
# Problem 1
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
dset = DataSet(set([classes[0]]), set([classes[1], classes[2]]))
model = Perceptron(4, lr=1e-3, batch_size=1)

# visualize data
plot_data(dset)

# train a simple model and visualize output
acc = []
for i in range(10):
    for x, y, c in dset.get_randomized():
        model.update(x, y)
        acc.append(get_percent_acc(model, dset))

plot_accuracies([acc], [1e-3])
plot_data(dset, model.theta)
print("THETA", model.theta)

# train a simple model with absolute correction
model = Perceptron(4, lr=1e-3, batch_size=1)
acc = []
for i in range(1):
    for x, y, c in dset.get_randomized():
        model.absolute_correction_update(x, y)
        acc.append(get_percent_acc(model, dset))

plot_accuracies([acc], [1e-3])
plot_data(dset, model.theta)
print("THETA", model.theta)

# test the effect of learning rate
lrs = 2**(-np.linspace(1, 5, 5))
accs = []
for lr in lrs:
    accs.append([])
    model = Perceptron(4, lr=lr, batch_size=1)
    for i in range(2):
        for x, y, c in dset.get_randomized():
            model.update(x, y)
            accs[-1].append(get_percent_acc(model, dset))
        
plot_accuracies(accs, lrs)

# test the effect of batch size
lrs = 2**(-np.linspace(1, 5, 5))
accs = []
for lr in lrs:
    accs.append([])
    model = Perceptron(4, lr=lr, batch_size=10)
    for i in range(2):
        for x, y, c in dset.get_randomized():
            model.update(x, y)
            accs[-1].append(get_percent_acc(model, dset))
        
plot_accuracies(accs, lrs)



