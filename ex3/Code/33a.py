import matplotlib.pyplot as plt
import numpy as np

def plotArray(X, c):

    for i in np.arange(0, X.shape[0]):
        plt.plot(X[i,1], X[i,2], c)

    return

def normalize(X):
    plotArray(X, 'bo')

    m = np.array([np.mean(X, 0)])
    m = np.repeat(m, X.shape[0], 0)
    X = np.subtract(X, m)
    plotArray(X, 'ro')

    var = np.array([np.var(X, 0)])
    var = np.repeat(var, X.shape[0], 0)
    X = np.divide(X, var)
    plotArray(X, 'go')
    plt.show()

    return X

data = np.loadtxt('../dataSets/iris.txt', delimiter=',')
data = normalize(data)




print data
