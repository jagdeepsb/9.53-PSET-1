import matplotlib.pyplot as plt
import numpy as np

def plot_data(dset, theta=None, labels_to_plot=[0,1,2]):

    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    data = []
    for label in labels:
        data.append([])

    for x, y, c in dset.data:
        data[c].append(list(x))

    for i in range(len(data)):
        data[i] = np.array(data[i])

    ax = plt.axes(projection='3d')

    colors = ['red', 'green', 'blue']
    axis = [0, 1, 2]
    for i in labels_to_plot:
        ax.scatter3D(data[i][:, axis[0]], data[i][:, axis[1]], data[i][:, axis[2]], c=colors[i], label=labels[i])
    ax.legend()

    if not theta is None:
        def f(x, y):
            return -(theta[axis[0]]*x + theta[axis[1]]*y + theta[-1])/theta[axis[2]]

        x = np.linspace(0, 7, 30)
        y = np.linspace(0, 7, 30)

        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        # ax.contour3D(X, Y, Z, 50, cmap='binary')
        ax.plot_surface(X, Y, Z, cmap='binary')
        
    ax.legend()

    ax.set_xlabel('sepal length')
    ax.set_ylabel('sepal width')
    ax.set_zlabel('petal length')


    plt.show()

def plot_data_adaline(dset, theta=None, labels_to_plot=[0,1,2]):

    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    data = []
    for label in labels:
        data.append([])

    for x, y, c in dset.data:
        data[c].append(list(x))

    for i in range(len(data)):
        data[i] = np.array(data[i])

    ax = plt.axes(projection='3d')

    colors = ['red', 'green', 'blue']
    axis = [2, 3, 0]
    for i in labels_to_plot:
        ax.scatter3D(data[i][:, axis[0]], data[i][:, axis[1]], data[i][:, axis[2]], c=colors[i], label=labels[i])
    ax.legend()

    if not theta is None:
        def f(x, y):
            return -(theta[axis[0]]*x + theta[axis[1]]*y + theta[-1])/theta[axis[2]]

        x = np.linspace(0, 7, 30)
        y = np.linspace(0, 7, 30)

        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)
        # ax.contour3D(X, Y, Z, 50, cmap='binary')
        ax.plot_surface(X, Y, Z, cmap='binary')
        
    ax.legend()

    ax.set_xlabel('sepal length')
    ax.set_ylabel('sepal width')
    ax.set_zlabel('petal length')


    plt.show()

def plot_accuracies(accs, lrs):
    fig, ax = plt.subplots()
    for acc, label in zip(accs, lrs):
        ax.plot(np.linspace(1, len(acc), len(acc)), acc, label=label)

    
    ax.set_xlabel('iters')
    ax.set_ylabel('acc')
    ax.legend()
    plt.show()

def plot_losses_adaline(losses, lrs):
    fig, ax = plt.subplots()
    for acc, label in zip(losses, lrs):
        ax.plot(np.linspace(1, len(acc), len(acc)), acc, label=label)

    
    ax.set_xlabel('iters')
    ax.set_ylabel('percent accuracy')
    ax.legend()
    plt.show()

def plot_losses_adaline(losses, lrs):
    fig, ax = plt.subplots()
    for acc, label in zip(losses, lrs):
        ax.plot(np.linspace(1, len(acc), len(acc)), acc, label=label)

    
    ax.set_xlabel('iters')
    ax.set_ylabel('error')
    ax.legend()
    plt.show()
    

