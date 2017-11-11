'''
Simple Stochastic Feedforward Neural Network
W. Ryan Lee

Implementation of a simple MLP using stochastic binary units.
Uses data augmentation (latent h, weights W) for training/sampling.
'''

import numpy as np
from pypolyagamma import PyPolyaGamma
from scipy.special import expit
from scipy.stats import norm
from itertools import izip
from sklearn.linear_model import LinearRegression, LogisticRegression


# Prob definitions
def lognorm(x, mu, sigma):
    return -(x-mu)**2/(2*sigma**2)



# Stochastic layer definition
class BinaryLayer:
    def __init__(self, dim, prevdim, n): # n = training set size
        self.h = np.array([0]*dim*n).reshape((n, dim)) # n x hid_dim array
        self.p = np.array([0.0]*dim*n).reshape((n, dim))
        self.Z = np.array([0.0]*dim*n).reshape((n, dim))
        # Glorot uniform initialization of W
        r = np.sqrt(6.0/(dim + prevdim))
        self.W = np.random.uniform(-r, r, (prevdim, dim))

    def forward(self, prev, i):
        self.Z[i] = np.dot(prev, self.W)
        self.p[i] = expit(self.Z[i])
        self.h[i] = np.random.binomial(1, self.p[i])
        return self.h[i]

    def det_forward(self, prev, i):
        self.Z[i] = np.dot(prev, self.W)
        self.p[i] = expit(self.Z[i])
        return self.p[i]

    def pred_forward(self, prev):
        Z = np.dot(prev, self.W)
        p = expit(Z)
        h = np.random.binomial(1, p)
        return h

class OutputLayer:
    def __init__(self, prevdim, n):
        self.Z = np.array([0.0]*n)
        r = np.sqrt(6.0/(1 + prevdim))
        self.W = np.random.uniform(-r, r, (prevdim,))

    def forward(self, prev, i):
        self.Z[i] = np.dot(prev, self.W)
        return self.Z[i]

    def det_forward(self, prev, i):
        #print(self.W.shape)
        self.Z[i] = np.dot(prev, self.W)
        return self.Z[i]

    def pred_forward(self, prev):
        Z = np.dot(prev, self.W)
        return Z

# Model definition
class SimpleModel:
    def __init__(self, input_dim = 200, train_size = 100, hid_dim = 100, layers = 3, step_size = 0.01, sigma = 1):
        # Initialize hyperparams
        self.nlayer = layers # total layers = nlayer + 1 (output)
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.step_size = step_size
        self.sigma = sigma
        self.train_size = train_size
        # Initialize layers
        self.layers = [BinaryLayer(hid_dim, input_dim, train_size)] # Input layer
        for i in range(layers - 1):
            self.layers.append(BinaryLayer(hid_dim, hid_dim, train_size)) # Hidden layers
        self.layers.append(OutputLayer(hid_dim, train_size)) # Output layer
    
    def forward(self, x, i):
        prev = self.layers[0].forward(x, i)
        for l in range(1, self.nlayer):
            prev = self.layers[l].forward(prev, i)
        self.output = self.layers[self.nlayer].forward(prev, i)
        return self.output

    def pred_forward(self, x):
        prev = self.layers[0].pred_forward(x)
        for l in range(1, self.nlayer):
            prev = self.layers[l].pred_forward(prev)
        output = self.layers[self.nlayer].pred_forward(prev)
        return output

    def backpropAndReject(self, x, y, i):
        mse = MSE()
        ent = CrossEntropy()
        accept = 0 # for diagnostics
        for l in range(self.nlayer - 1, -1, -1):
            # Backprop
            if l > 0:
                prev = self.layers[l].det_forward(self.layers[l-1].h[i], i)
            else:
                prev = self.layers[l].det_forward(x, i)
            pred = self.layers[l+1].det_forward(prev, i)
            if l == self.nlayer - 1:
                grad = mse.backward(pred, y) * self.layers[l+1].W # negative gradient
                m = len(grad)
            else:
                grad = np.array([0]*self.hid_dim)
                j = 0
                for p, q in izip(pred, self.layers[l+1].h[i]):
                    res = ent.backward(p, q)
                    grad = grad + res * p * (1-p) * self.layers[l+1].W[:,j]
                    j += 1
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
            curr = self.layers[l].h[i]
            proppred = self.layers[l+1].det_forward(prop, i)
            pred = self.layers[l+1].det_forward(curr, i)
            if l == self.nlayer - 1:
                logalpha = lognorm(proppred, y, self.sigma) - lognorm(pred, y, self.sigma)
            else:
                next = self.layers[l+1].h[i]
                logalpha = 0
                for j in range(m):
                    logalpha = logalpha + next[j]*np.log(proppred[j]/pred[j]) + (1-next[j])*np.log((1-proppred[j])/(1-pred[j]))
            for j in range(m):
                    logalpha = logalpha + (prop[j] - curr[j])*np.log(prev[j]) + (curr[j] - prop[j])*np.log(1-prev[j])
            if np.log(np.random.uniform()) < logalpha:
                accept += 1
                self.layers[l].h[i] = prop
        return accept

    def logisticAndReject(self, X, Y):
        pg = PyPolyaGamma() # use N(0, I) prior
        n = X.shape[0]
        # Output layer
        out_fit = LinearRegression(fit_intercept = False).fit(self.layers[self.nlayer-1].h, Y)
        self.layers[self.nlayer].W = out_fit.coef_
        # Hidden layers
        for l in range(self.nlayer - 1, 0, -1):
        #    for j in range(self.hid_dim):
                # Draw prior beta
                #prior = np.random.normal(0, 1, size = self.hid_dim)
                # Draw latent w
                #w = np.zeros(n)
                #for k in range(n):
                #    w[k] = pg.pgdraw(1, np.dot(self.layers[l-1].h[k,:], prior))
                # Draw posterior beta
                #kappa = self.layers[l].h[:,j] - 0.5
                #omega = np.diag(w)
                #Vw = np.linalg.inv(np.dot(np.dot(np.transpose(self.layers[l].h), omega), self.layers[l].h) + np.eye(self.hid_dim))
                #mw = np.dot(Vw, np.dot(np.transpose(self.layers[l].h), kappa))
                #self.layers[l].W[:,j] = np.random.multivariate_normal(mw, Vw)
                
            # Propose
            propW = np.zeros(self.layers[l].W.shape)
            logalpha = 0
            for j in range(self.hid_dim):
                hid_fit = LogisticRegression(fit_intercept = False).fit(self.layers[l-1].h, self.layers[l].h[:,j])
                propW[:,j] = hid_fit.coef_ + np.random.normal(size = len(propW[:,j]))
                prop_hW = expit(np.dot(self.layers[l-1].h, propW[:,j]))
                curr_hW = expit(np.dot(self.layers[l-1].h, self.layers[l].W[:,j]))
                # Accept-Reject
                logalpha = sum(self.layers[l].h[:,j] * np.log(prop_hW/curr_hW) + (1-self.layers[l].h[:,j]) * np.log((1-prop_hW)/(1-curr_hW)))
                if np.log(np.random.uniform()) < logalpha:
                    self.layers[l].W[:,j] = propW[:,j]
                               

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
epochs = 300
layers = 3
input_dim = 50
hid_dim = 30

model = SimpleModel(input_dim, train_size, hid_dim, layers) # Model init

# Training/Sampling
# Initial h sample
model.forward(X_train[0], 0)
# Gibbs
for t in range(epochs):
    print "Epoch: " + str(t)
    
    # Training
    a = 0 # number of accepts each epoch
    for i in range(train_size):
        a += model.backpropAndReject(X_train[i], y_train[i], i) # h update for each obs
    model.logisticAndReject(X_train, y_train) # W update using all obs
    print "H Acceptance Proportion: " + str(float(a)/(layers*train_size))
    
    # Prediction
    bary = np.mean(y_train)
    ssr = 0.0
    sse = 0.0
    sst = 0.0
    for i in range(train_size):
        pred = model.pred_forward(X_train[i])
        ssr += (pred - bary)**2
        sse += (y_train[i] - pred)**2
        sst += (y_train[i] - bary)**2
    print "Prediction R2: " + str(ssr/sst)
    print "Prediction Error: " + str(sse/sst)
