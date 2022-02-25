import numpy as np

from plot import plot_data_adaline, plot_losses_adaline
from data import DataSet
from algos import Adaline

def get_loss(model, dset):
    loss = 0
    for x, y, c in dset.data:
        y_hat = model.call(x)
        loss += np.sum((y-y_hat)**2)
    return loss / len(dset.data) 
        
# Problem 3
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
dset = DataSet(set([classes[1]]), set([classes[2]]))
model = Adaline(4, lr=1e-3)

# plot data
plot_data_adaline(dset, labels_to_plot=[0, 1])

# train a simple model and visualize output
loss = []
for i in range(10):
    for x, y, c in dset.get_randomized():
        model.update(x, y)
        loss.append(get_loss(model, dset))

plot_losses_adaline([loss], [1e-3])
plot_data_adaline(dset, model.theta, labels_to_plot=[0, 1])
print("THETA", model.theta)

# test the effect of learning rate
lrs = 2**(-np.linspace(1+5, 5+9, 5))
losses = []
for lr in lrs:
    losses.append([])
    model = Adaline(4, lr=lr)
    for i in range(10):
        for x, y, c in dset.get_randomized():
            model.update(x, y)
            losses[-1].append(get_loss(model, dset))
        
plot_losses_adaline(losses, lrs)
