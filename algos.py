import numpy as np

class Perceptron():
    def __init__(self, input_dim, lr=1e-3, batch_size=1):
        # basic
        self.theta = np.ones((input_dim+1,))
        self.lr = lr

        #for handling batches
        self.batch_size = batch_size
        self.count = 0
        self.theta_new = self.theta.copy()

    # get pred
    def call(self, x):
        xp =  np.concatenate((x, np.array([1.0])))
        out = np.sum(xp * self.theta)
        return -1 if out <= 0 else 1

    # run pred and update model
    def update(self, x, y):
        xp =  np.concatenate((x, np.array([1.0])))
        y_hat = self.call(x)

        # if incorrect
        if y_hat/y < 0:
            if y_hat > 0:
                self.theta_new = self.theta_new - self.lr*xp
            if y_hat < 0:
                self.theta_new = self.theta_new + self.lr*xp

        # manage batches
        self.count += 1
        if self.count % self.batch_size == 0:
            self.theta = self.theta_new.copy()

    # update with absolute correction
    def absolute_correction_update(self, x, y):
        xp =  np.concatenate((x, np.array([1.0])))
        y_hat = self.call(x)

        # if incorrect
        if y_hat/y < 0:
            lr = np.sum(xp * xp)/y_hat
            if lr <= 0:
                lr = -lr
            if y_hat > 0:
                self.theta_new = self.theta_new - lr*xp
            if y_hat < 0:
                self.theta_new = self.theta_new + lr*xp

        # manage batches
        self.count += 1
        if self.count % self.batch_size == 0:
            self.theta = self.theta_new.copy()
        

class Adaline():
    def __init__(self, input_dim, lr=1e-3):
        self.theta = np.ones((input_dim+1,))
        self.lr = lr

    # get pred
    def call(self, x):
        xp =  np.concatenate((x, np.array([1.0])))
        out = np.sum(xp * self.theta)
        return out #-1 if out <= 0 else 1

    # run pred and update model
    def update(self, x, y):
        xp =  np.concatenate((x, np.array([1.0])))
        y_hat = self.call(x)
        self.theta = self.theta + self.lr*(y-y_hat)*xp