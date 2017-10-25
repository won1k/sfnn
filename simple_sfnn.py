'''
Simple Stochastic Feedforward Neural Network
W. Ryan Lee

Implementation of a simple MLP using stochastic binary units.
Uses data augmentation (latent h, weights W) for training/sampling.
'''

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from itertools import izip

# Stochastic layer definition
class BinaryLayer:
    def __init__(self, dim, prevdim):
        self.h = np.array([0]*dim)
        self.p = np.array([0]*dim)
        self.Z = np.array([0]*dim)
        # Glorot uniform initialization of W
        r = np.sqrt(6.0/(dim + prevdim))
        self.W = np.random.uniform(-r, r, (prevdim, dim))

    def forward(self, prev):
        self.Z = np.dot(prev, self.W)
        self.p = expit(self.Z)
        self.h = np.random.binomial(1, self.p)
        return self.h

    def det_forward(self, prev):
        self.Z = np.dot(prev, self.W)
        self.p = expit(self.Z)
        return self.p

class OutputLayer:
    def __init__(self, prevdim):
        self.Z = np.array([0])
        r = np.sqrt(6.0/(1 + prevdim))
        self.W = np.random.uniform(-r, r, (prevdim,))

    def forward(self, prev):
        self.Z = np.dot(prev, self.W)
        return self.Z

    def det_forward(self, prev):
        self.Z = np.dot(prev, self.W)
        return self.Z

# Model definition
class SimpleModel:
    def __init__(self, input_dim = 200, train_size = 100, hid_dim = 100, layers = 3, step_size = 0.01, sigma = 1):
        # Initialize hyperparams
        self.nlayer = layers # total layers = nlayer + 1 (output)
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.step_size = step_size
        self.sigma = sigma
        # Initialize layers
        self.layers = [BinaryLayer(hid_dim, input_dim)] # Input layer
        for i in range(layers - 1):
            self.layers.append(BinaryLayer(hid_dim, hid_dim)) # Hidden layers
        self.layers.append(OutputLayer(hid_dim)) # Output layer
    
    def forward(self, x):
        prev = self.layers[0].forward(x)
        for i in range(1, layers):
            prev = self.layers[i].forward(prev)
        self.output = self.layers[layers].forward(prev)
        return self.output

    def backpropAndReject(self, x, y):
        mse = MSE()
        ent = CrossEntropy()
        for l in range(self.nlayer - 1, -1, -1):
            # Backprop
            if l > 0:
                prev = self.layers[l].det_forward(self.layers[l-1].h)
                #print("Length: " + str(len(prev)))
            else:
                prev = self.layers[l].det_forward(x)
                #print("The length is " + str(len(prev)))
            pred = self.layers[l+1].det_forward(prev)
            if l == self.nlayer - 1:
                grad = mse.backward(pred, y) * self.layers[l+1].W # negative gradient
                m = len(grad)
            else:
                grad = np.array([0]*self.hid_dim)
                i = 0
                for p, q in izip(pred, self.layers[l+1].h):
                    res = ent.backward(p, q)
                    grad = grad + res * p * (1-p) * self.layers[l+1].W[:,i]
                    i += 1
                #res = ent.backward(pred, self.layers[l+1].h)
                #m = len(res)
                #grad = np.array([0]*m)
                #for k in range(m):
                #    grad = grad + res[k] * pred[k] * (1-pred[k]) * self.layers[l+1].W[:,k]
                grad = (-1) * grad
            prev = [min(max(t, 0.001), 0.999) for t in prev + self.step_size * grad]
            # Proposal
            prop = np.random.binomial(1, prev)
            # Accept-Reject
            curr = self.layers[l].h
            proppred = self.layers[l+1].det_forward(prop)
            pred = self.layers[l+1].det_forward(curr)
            if l == self.nlayer - 1:
                logalpha = np.log(norm(proppred, self.sigma).pdf(y) / norm(pred, self.sigma).pdf(y))
            else:
                next = self.layers[l+1].h
                logalpha = 0
                for j in range(m):
                    logalpha = logalpha + next[j]*np.log(proppred[j]/pred[j]) + (1-next[j])*np.log((1-proppred[j])/(1-pred[j]))
            for j in range(m):
                    logalpha = logalpha + (prop[j] - curr[j])*np.log(prev[j]) + (curr[j] - prop[j])*np.log(1-prev[j])
            if np.log(np.random.uniform()) < logalpha:
                self.layers[l].h = prop

    def logisticAndReject(X, y):
        return True
            
            
        

# Loss functions
class MSE:
    def forward(self, pred, true):
        return (pred - true)**2

    def backward(self, pred, true):
        return 2*(pred-true)

class CrossEntropy:
    def forward(self, pred, true):
        return -sum(true * np.log(pred) + (1-true) * np.log(1-pred))
    
    def backward(self, pred, true):
        return (true - pred) / (pred * (1-pred))


'''
Sample model execution
'''

# Data loading                                                                                                                          
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

# Model initialization
train_size = len(X_train)
epochs = 100
layers = 3
input_dim = 50
hid_dim = 30

model = SimpleModel(input_dim, train_size, hid_dim, layers) # Model init

# MCMC
# Initial h sample
model.forward(X_train[0])
# Gibbs
for t in range(epochs):
    print "Epoch: " + str(t)
    for i in range(train_size):
#        print("Obs: " + str(i))
        model.backpropAndReject(X_train[i], y_train[i]) # h update for each obs
    #model.logisticAndReject(X_train, y_train) # W update using all obs

# Prediction






