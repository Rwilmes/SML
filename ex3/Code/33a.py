import matplotlib.pyplot as plt
import numpy as np


def plotArray(X, c):
    for i in np.arange(0, X.shape[0]):
        plt.plot(X[i, 1], X[i, 2], c)

    return


def normalize(X):
    # Subtract mean
    m = np.array([np.mean(X, 0)])
    m = np.repeat(m, X.shape[0], 0)
    X = np.subtract(X, m)

    # Devide by standard deviation
    dev = np.array([np.std(X, 0)])
    X = np.divide(X, dev)

    return X


def pca(X):
    covar = np.cov(X, rowvar=False)
    [eval, evec] = np.linalg.eig(covar)
    return eval, evec


data = np.loadtxt('../dataSets/iris.txt', delimiter=',')
data = normalize(data)
[eval, evec] = pca(data)

totalVariance = np.sum(eval)

componentsNeeded = 0
explainedVariance = np.cumsum(eval)/np.sum(eval)
print "Explained: ", explainedVariance
i = 0
while explainedVariance[i] < 0.95:
    i += 1

datap = np.dot(evec, data.T).T
datap = datap[:, np.arange(0, 2)]
plt.plot(datap[:, 0], datap[:, 1], '.')
plt.plot(data[:, 0], data[:, 1], 'x')

plt.show()

print "Necessary components"
print i
print "Variance explained", explainedVariance / totalVariance
print "Eigen values: "
print eval
print "Evec: "
print evec
